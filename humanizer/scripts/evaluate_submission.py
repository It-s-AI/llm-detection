"""Evaluate one submission on a period, end to end: quote -> hold -> run -> gates -> detectors -> score and
crown test -> statement -> settlement (src/humanizer_eval/evaluation).

Outputs come from a bundle on a fresh Lium pod (--bundle / --bundle-url, needs --yes) or from a saved
pilot submission (--replay NAME reads var/periods/<period>/NAME.enc). The king is the period's king run;
with none yet, the owner's identity baseline (inputs unchanged) is scored first and becomes the king.

--detectors its_ai runs the whole cycle on the free proxy detector (a demo: It's AI never decides the
crown, D-33); king state and king run then live in files tagged with the detector set.
--billing test uses Stripe TEST mode (a test-card top-up when the balance is short); off = owner run.

Usage:
  python scripts/evaluate_submission.py --replay ref_v2 --detectors its_ai --billing test
  python scripts/evaluate_submission.py --bundle var/lium/identity.tar --cheap-gpu --yes --detectors its_ai
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from humanizer_eval.billing import Accounts, BillingService, PriceCard, StripeAPI
from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.detectors.registry import CROWN_DETECTORS, build_detector
from humanizer_eval.evaluation import (
    EvalConfig,
    InsufficientBalance,
    KingRun,
    LiumRunner,
    ReplayRunner,
    Submission,
    SubmissionStore,
    evaluate_submission,
    load_period,
)
from humanizer_eval.evaluation.runners import rank_executors
from humanizer_eval.gates.judges import GateRunner
from humanizer_eval.llm.cache import CachedLLM
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, load_or_create_key
from humanizer_eval.pool.store import PoolStore
from humanizer_eval.runner.lium import LiumClient, Registry

OWNER_HOTKEY = "owner-baseline"
CHEAP = ("3090", "4090", "A4000", "A5000")


def cheap_gpus(executors):
    return rank_executors(executors, 0.4, CHEAP)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--replay", help="saved outputs in the period dir, e.g. ref_v2")
    src.add_argument("--bundle", type=Path, help="local bundle .tar (uploaded to the pod)")
    src.add_argument("--bundle-url", help="https URL the pod downloads (with --bundle-sha256)")
    ap.add_argument("--bundle-sha256")
    ap.add_argument("--period", default="pilot-1")
    ap.add_argument(
        "--pool", default="pool-full-v2,pool-full", help="pool dirs in var/ to look H texts up in"
    )
    ap.add_argument("--allow-missing-human", action="store_true", help="pilot only: H texts not in any pool")
    ap.add_argument("--hotkey", default="5Fpilot-miner")
    ap.add_argument("--submission-id", default=None)
    ap.add_argument("--detectors", default=",".join(CROWN_DETECTORS))
    ap.add_argument("--billing", choices=("test", "off"), default="test")
    ap.add_argument("--no-update-king", action="store_true")
    ap.add_argument(
        "--king-only",
        action="store_true",
        help="score the human texts and the owner's baseline king on this period, then stop",
    )
    ap.add_argument("--yes", action="store_true", help="really rent a Lium pod")
    ap.add_argument("--cheap-gpu", action="store_true", help="smoke runs: a 3090/4090-class GPU, not an H100")
    ap.add_argument("--max-price", type=float, default=2.6)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if not args.king_only and not (args.replay or args.bundle or args.bundle_url):
        ap.error("give --replay, --bundle or --bundle-url (or --king-only)")

    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    spent0 = {k: v["spent"] for k, v in led.summary().items()}
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    base = OpenRouter(s.openrouter_api_key, led, pricing=ModelPricing(var_dir() / "openrouter_models.json"),
                      log_path=var_dir() / "llm_calls.jsonl")  # fmt: skip
    llm = CachedLLM(base, var_dir() / "calibration" / "llm_cache.sqlite", key=key, cache_all=True)
    pdir = var_dir() / "periods" / args.period
    dets = [d for d in args.detectors.split(",") if d]
    detectors = {d: build_detector(d, ledger=led) for d in dets}
    data = json.loads(decrypt_at_rest(key, (pdir / "dataset.enc").read_bytes()))
    want = {h["id"] for h in data["human_ref"]}
    texts: dict[str, str] = {}
    for pool in args.pool.split(","):
        rows = PoolStore.load(var_dir() / pool).rows
        texts.update({r["text_id"]: r["text"] for r in rows if r["text_id"] in want})
    ctx = load_period(pdir, key, human_texts=texts, detectors=detectors, facts_llm=llm, workers=args.workers,
                      allow_missing_human=args.allow_missing_human)  # fmt: skip
    tag = "-".join(sorted(dets))
    store = SubmissionStore(pdir / "submissions", key)
    king_run_path = pdir / f"king_run.{tag}.enc"
    king_state_path = var_dir() / "crown" / f"king_state.{args.period}.{tag}.json"
    gates_cfg = json.loads(Path("configs/gates.json").read_text())

    def gate_factory(m):
        return GateRunner.from_config(m, gates_cfg)

    from humanizer_eval.evaluation.conditions import scoring_fingerprint

    fp = scoring_fingerprint(gates_cfg, detectors, {"n_boot": args.n_boot})
    cfg = EvalConfig(workers=args.workers, update_king=not args.no_update_king, crown_kw={"n_boot": args.n_boot},
                     fingerprint=fp)  # fmt: skip
    common = {"llm": llm, "gate_factory": gate_factory, "detectors": detectors, "store": store, "cfg": cfg}

    king = KingRun.load(king_run_path, key)
    if king is not None and king.fingerprint != fp:
        if not king.submission_id.startswith("baseline"):
            raise SystemExit("the king's run was scored under other conditions: rescore it first")
        king = None  # the baseline is rescored below, under the current conditions
    if king is None:  # the owner's baseline: inputs unchanged (D-13), scored like any submission, owner pays
        bsid = f"baseline-identity-{tag}"
        r = evaluate_submission(Submission(bsid, OWNER_HOTKEY, name="identity"), ctx,
                                runner=ReplayRunner([it["text"] for it in ctx.items]), **common)  # fmt: skip
        sc = {d: [float("inf") if x is None else x for x in v] for d, v in r.private["scores"].items()}
        king = KingRun(bsid, OWNER_HOTKEY, ctx.input_ids, r.private["levels"], sc, fingerprint=fp)
        king.save(king_run_path, key)
        print("baseline king:", json.dumps(r.public))
    if args.king_only:
        spent = {k: round(v["spent"] - spent0.get(k, 0.0), 3) for k, v in led.summary().items()
                 if v["spent"] - spent0.get(k, 0.0) > 0}  # fmt: skip
        print(json.dumps({"period": args.period, "n_inputs": len(ctx.items), "n_human": ctx.meta["n_human"],
                          "missing_human": ctx.meta["missing_human"], "king": king.submission_id,
                          "owner_spent": spent}, indent=1))  # fmt: skip
        return 0

    billing = None
    if args.billing == "test":
        api = StripeAPI(s.stripe_secret_key)  # refuses live keys
        if not api.test_mode:
            raise SystemExit("--billing test needs a Stripe test key")
        billing = BillingService(PriceCard.load("configs/price_card.json"),
                                 Accounts(var_dir() / "billing" / "pilot.sqlite"), api)  # fmt: skip

    if args.replay:
        outs = json.loads(decrypt_at_rest(key, (pdir / f"{args.replay}.enc").read_bytes()))
        runner = ReplayRunner(outs)
    else:
        client = LiumClient(s.lium_api_key, led, Registry(var_dir() / "lium" / "registry.json"), enabled=args.yes,
                            floor_hours=s.lium_balance_floor_hours, max_price_per_hour=args.max_price,
                            max_hours=2)  # fmt: skip
        runner = LiumRunner(client, Path.home() / ".ssh" / "humeval_lium_ed25519", max_price=args.max_price,
                            rank=cheap_gpus if args.cheap_gpu else None)  # fmt: skip
    sid = args.submission_id or f"sub-{uuid.uuid4().hex[:10]}"
    sub = Submission(sid, args.hotkey, bundle=args.bundle, bundle_url=args.bundle_url,
                     bundle_sha256=args.bundle_sha256, name=args.replay or "")  # fmt: skip
    kw = {**common, "runner": runner, "king": king, "billing": billing, "king_state_path": king_state_path,
          "king_run_path": king_run_path}  # fmt: skip
    try:
        res = evaluate_submission(sub, ctx, **kw)
    except InsufficientBalance as e:
        print(f"balance short by {e.amount_due_cents} cents: Stripe TEST top-up")
        billing.test_topup(args.hotkey, e.amount_due_cents, ref=f"{e.quote['id']}:testcard")
        res = evaluate_submission(sub, ctx, **kw)
    p = res.private
    summary = {
        "public": res.public,
        "statement": None
        if res.statement is None
        else {i["item"]: [i["quantity"], i["amount_cents"]] for i in res.statement["items"]},
        "settlement": res.settlement,
        "levels": p.get("levels_count"),
        "score": p.get("score"),
        "crown": None
        if not p.get("crown")
        else {k: p["crown"][k] for k in ("crown", "delta", "lower", "se", "s_king", "s_challenger")},
        "run": {k: p["run"][k] for k in ("status", "reason", "pod_seconds", "cost_usd")}
        if p.get("run")
        else None,
        "judge_tokens": p.get("judge_tokens"),
        "seconds": p.get("seconds"),
        "n_human": ctx.meta["n_human"],
        "missing_human": ctx.meta["missing_human"],
        "owner_spent": {
            k: round(v["spent"] - spent0.get(k, 0.0), 3)
            for k, v in led.summary().items()
            if v["spent"] - spent0.get(k, 0.0) > 0
        },
    }
    print(json.dumps(summary, indent=1, default=str))
    out = Path("docs/reports/data") / f"09-e2e-{args.replay or 'bundle'}-{tag}.json"
    out.write_text(json.dumps(summary, indent=1, default=str))
    return 0 if res.status == "done" else 2


if __name__ == "__main__":
    raise SystemExit(main())
