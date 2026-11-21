"""Run the miner API and the queue worker in one process (one evaluation at a time).

    python scripts/intake_server.py --period pilot-1 --detectors its_ai --dev-any-hotkey \
        --replay-map var/intake-dev/replay.json --state-dir var/intake-dev

Production: no --period (the period named in var/periods/CURRENT), crown detectors, the metagraph
snapshot with commitments (--metagraph, written by scripts/refresh_metagraph.py every few minutes),
bundles on the miners' Hugging Face repos pinned by those commitments, runs on Lium (--lium); a crown
counts once the bundle is in <state>/archive. 48 h after a coronation GET /v1/kings links the king's bundle:
the miner's repo at the pinned commit if anyone can download it, else our copy in --mirror-repo.
Development: --replay-map {bundle_url: saved outputs name in the period dir} runs no pod; --dev-any-hotkey
skips the registration check. Stripe must be in test mode unless --live-stripe (not before launch).
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from humanizer_eval.billing import Accounts, BillingService, PriceCard, StripeAPI
from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.detectors.registry import CROWN_DETECTORS, build_detector
from humanizer_eval.evaluation import EvalConfig, LiumRunner, ReplayRunner, SubmissionStore, load_period
from humanizer_eval.gates.judges import GateRunner
from humanizer_eval.intake import (
    CurrentPeriod,
    IntakeLimits,
    IntakeService,
    MetagraphSnapshot,
    NonceStore,
    QueueWorker,
    SubmissionQueue,
    Verifier,
    WorkerEnv,
)
from humanizer_eval.intake.api import create_app
from humanizer_eval.intake.archive import Archive
from humanizer_eval.intake.hf import HubClient
from humanizer_eval.llm.cache import CachedLLM
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, load_or_create_key
from humanizer_eval.pool.store import PoolStore
from humanizer_eval.publish import HFMirror, KingPublisher, build_payload, load_or_create_signing_key, sign
from humanizer_eval.runner.lium import LiumClient, Registry
from humanizer_eval.scoring.crown import KingState


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8032)
    ap.add_argument("--public-url", default="http://127.0.0.1:8032", help="base for Checkout return URLs")
    ap.add_argument("--period", default=None, help="fixed period dir name (default: periods/CURRENT)")
    ap.add_argument("--pool", default="pool-full-v2,pool-full")
    ap.add_argument("--allow-missing-human", action="store_true")
    ap.add_argument("--detectors", default=",".join(CROWN_DETECTORS))
    ap.add_argument("--state-dir", type=Path, default=None, help="queue, billing, king and records")
    ap.add_argument("--metagraph", type=Path, default=None)
    ap.add_argument("--dev-any-hotkey", action="store_true")
    ap.add_argument("--replay-map", type=Path, default=None)
    ap.add_argument("--lium", action="store_true", help="really rent Lium pods for bundles")
    ap.add_argument("--cheap-gpu", action="store_true", help="tests only: a 3090/4090-class pod, not an H100")
    ap.add_argument("--live-stripe", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-worker", action="store_true")
    ap.add_argument("--netuid", type=int, default=32)
    ap.add_argument(
        "--mirror-repo",
        default=None,
        help="our public HF repo <org>/<repo> for copies of kings whose repo is closed (HF_MIRROR_TOKEN)",
    )
    ap.add_argument("--publish-every", type=float, default=600.0, help="seconds between publication passes")
    ap.add_argument(
        "--mode",
        choices=("prod", "test"),
        default="prod",
        help="test: shown as TEST everywhere, Stripe test keys only, a daily evaluation cap",
    )
    ap.add_argument(
        "--network", default="finney", help="the chain the metagraph snapshot comes from (shown in /v1/info)"
    )
    ap.add_argument(
        "--max-evals-per-day", type=int, default=None, help="cap for all miners together (test: 30)"
    )
    ap.add_argument(
        "--parallel", type=int, default=1, help="evaluations at once, one pod each (prod: own Lium)"
    )
    ap.add_argument(
        "--closed",
        action="store_true",
        help="read-only: no new submissions (info, results, kings stay public)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if (args.metagraph is None) == (not args.dev_any_hotkey):
        raise SystemExit("give exactly one of --metagraph, --dev-any-hotkey")
    if (args.replay_map is None) == (not args.lium):
        raise SystemExit("give exactly one of --replay-map, --lium")

    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    state = args.state_dir or (var_dir() / "intake")
    state.mkdir(parents=True, exist_ok=True)
    base = OpenRouter(s.openrouter_api_key, led, pricing=ModelPricing(var_dir() / "openrouter_models.json"),
                      log_path=var_dir() / "llm_calls.jsonl")  # fmt: skip
    llm = CachedLLM(base, var_dir() / "calibration" / "llm_cache.sqlite", key=key, cache_all=True)
    dets = [d for d in args.detectors.split(",") if d]
    detectors = {d: build_detector(d, ledger=led) for d in dets}
    tag = "-".join(sorted(dets))
    pools = [PoolStore.load(var_dir() / p) for p in args.pool.split(",")]

    def loader(pdir: Path):
        data = json.loads(decrypt_at_rest(key, (pdir / "dataset.enc").read_bytes()))
        want = {h["id"] for h in data["human_ref"]}
        texts = {r["text_id"]: r["text"] for p in pools for r in p.rows if r["text_id"] in want}
        return load_period(pdir, key, human_texts=texts, detectors=detectors, facts_llm=llm,
                           workers=args.workers, allow_missing_human=args.allow_missing_human,
                           human_cache=pdir / f"human_scores.{tag}.enc")  # fmt: skip

    periods = var_dir() / "periods"
    if args.period:
        fixed = loader(periods / args.period)

        def period():
            return fixed
    else:
        period = CurrentPeriod(periods, loader)

    stores: dict[str, SubmissionStore] = {}

    def store_for(pid: str) -> SubmissionStore:
        return stores.setdefault(pid, SubmissionStore(state / "periods" / pid / "submissions", key))

    api = StripeAPI(s.stripe_secret_key)
    if not api.test_mode and (args.mode == "test" or not args.live_stripe):
        raise SystemExit(
            "Stripe key is live; refusing (test mode never takes a live key; prod needs --live-stripe)"
        )
    if args.mode == "test" and args.max_evals_per_day is None:
        args.max_evals_per_day = 30
    billing = BillingService(
        PriceCard.load("configs/price_card.json"), Accounts(state / "billing.sqlite"), api
    )
    queue = SubmissionQueue(state / "queue.sqlite")
    if args.dev_any_hotkey:  # every hotkey counts as registered at block 0 (one submission each)

        def registration(hk):
            return 0

        def is_registered(hk, reg):
            return True
    else:
        snap = MetagraphSnapshot(args.metagraph, netuid=args.netuid)
        registration, is_registered = snap.registration, snap.is_registered
    hub = HubClient()
    commitment = None if args.metagraph is None else snap.commitment
    secret_key = load_or_create_key(var_dir() / "keys" / "intake_secrets.key")
    limits = IntakeLimits(allow_raw_urls=bool(args.replay_map),  # raw URLs only with the dev replay map
                          max_evals_per_day=args.max_evals_per_day, submissions_open=not args.closed)  # fmt: skip
    svc = IntakeService(queue, billing, period, store_for, registration,
                        success_url=f"{args.public_url}/v1/checkout/done",
                        cancel_url=f"{args.public_url}/v1/checkout/cancelled", limits=limits,
                        commitment=commitment, hub=hub, secret_key=secret_key,
                        about={"mode": args.mode, "network": args.network, "netuid": args.netuid})  # fmt: skip

    if args.replay_map:
        rmap = json.loads(args.replay_map.read_text())

        def runner_for(sub):
            ctx = period()
            name = rmap.get(sub.bundle_url)
            if name is None:
                raise RuntimeError("dev server: bundle_url not in the replay map")
            outs = json.loads(decrypt_at_rest(key, (ctx.dir / f"{name}.enc").read_bytes()))
            return ReplayRunner(outs)
    else:
        # one pod per evaluation, plus one for the king's run on the next period (prepare_loop)
        client = LiumClient(s.lium_api_key, led, Registry(var_dir() / "lium" / "registry.json"), enabled=True,
                            floor_hours=s.lium_balance_floor_hours, max_price_per_hour=2.6, max_hours=2,
                            max_concurrent=max(1, args.parallel) + 1)  # fmt: skip
        cheap = None
        if args.cheap_gpu:
            from humanizer_eval.evaluation.runners import rank_executors

            def cheap(executors):
                return rank_executors(executors, 0.4, ("3090", "4090", "A4000", "A5000"))

        lium = LiumRunner(client, Path.home() / ".ssh" / "humeval_lium_ed25519", rank=cheap)

        def runner_for(sub):
            return lium

    gates_cfg = json.loads(Path("configs/gates.json").read_text())
    archive = Archive(state / "archive")
    env = WorkerEnv(
        period=period,
        store_for=store_for,
        llm=llm,
        gate_factory=lambda m: GateRunner.from_config(m, gates_cfg),
        detectors=detectors,
        runner_for=runner_for,
        king_run_path=lambda ctx: state / "periods" / ctx.period_id / f"king_run.{tag}.enc",
        king_state_path=state / f"king_state.{tag}.json",
        key=key,
        cfg=EvalConfig(workers=args.workers, crown_kw={"n_boot": args.n_boot}),
        is_registered=is_registered,
        hub=hub,
        archive=archive,
        secret_key=secret_key,
    )
    from humanizer_eval.ops.alerts import from_settings as alerter_from_settings

    alerter = alerter_from_settings(state_path=var_dir() / "ops" / "alerts_state.json")
    worker = QueueWorker(queue, svc, billing, env, concurrency=max(1, args.parallel), alerter=alerter)
    stop = threading.Event()
    if not args.no_worker:
        threading.Thread(target=worker.serve, args=(stop,), kwargs={"idle_s": 5}, daemon=True).start()
    if args.mirror_repo and not s.hf_mirror_token:
        raise SystemExit("--mirror-repo needs HF_MIRROR_TOKEN")
    mirror = HFMirror(args.mirror_repo, s.hf_mirror_token) if args.mirror_repo else None
    publisher = KingPublisher(state / "king_publications.json", env.king_state_path, queue.get, hub, archive,
                              mirror, alerter=alerter)  # fmt: skip

    def publish_loop() -> None:
        while not stop.is_set():
            try:
                publisher.run_once()
                publisher.prune(is_registered)
            except Exception:
                logging.getLogger("publish").exception("publication pass failed")
            stop.wait(args.publish_every)

    def prepare_loop() -> None:
        """The next period (NEXT, written by rotate_period.py): load it here and run the king on it."""
        while not stop.is_set():
            try:
                nxt = period.upcoming() if isinstance(period, CurrentPeriod) else None
                if nxt is not None:
                    worker.prepare_next(nxt)
            except Exception as e:
                logging.getLogger("prepare").exception("preparing the next period failed")
                alerter.send(f"preparing the next period failed: {type(e).__name__}: {e}", dedupe="prepare")
            stop.wait(300)

    if not args.no_worker:
        threading.Thread(target=publish_loop, daemon=True).start()
        threading.Thread(target=prepare_loop, daemon=True).start()
    verifier = Verifier(NonceStore(state / "nonces.sqlite", ttl=600), max_skew=300)
    lb_key = load_or_create_signing_key(var_dir() / "keys" / "leaderboard_ed25519.key")
    lb_cache: dict[str, Any] = {}

    def leaderboard() -> dict[str, Any]:
        """Re-signed at most once a minute; the king state changes only on a coronation."""
        now = time.time()
        if lb_cache.get("t", 0) < now - 60:
            sp = env.king_state_path
            st = KingState.from_json(sp.read_text()) if sp.exists() else None
            lb_cache.update(
                t=now,
                env=sign(build_payload(st, netuid=args.netuid, now=now, is_registered=is_registered), lb_key),
            )
        return lb_cache["env"]

    app = create_app(svc, verifier, webhook_secret=s.stripe_webhook_secret or None, leaderboard=leaderboard,
                     kings=publisher.public, periods_dir=periods,
                     reveal_dir=var_dir() / "public" / "periods")  # fmt: skip

    import uvicorn

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
