"""Period rotation (D-56, D-57): one pass of period/lifecycle.py. Run by humeval-rotate.timer every 15 min.

    python scripts/rotate_period.py --state /srv/humeval/var/intake [--hf-repo org/repo] [--status]

A pass does what is due and exits: publish the next period's commitment and seed block, build the next
period once that block is final, switch CURRENT when the period ends and the server has the next one
ready (READY: H scored and the king re-run, done by the server itself), reveal the closed period into
var/public/periods/<id>/ (served at /v1/periods/<id>/...) once no evaluation of it is still running.

The first period is built by hand (build_period.py) and adopted on the first pass. Building costs about
$1 per 100 texts on OpenRouter (ledger caps apply); nothing here calls a detector or rents a pod.
--status prints the schedule and what the next pass would do, and changes nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import time
from pathlib import Path

from humanizer_eval.config import get_settings
from humanizer_eval.paths import var_dir
from humanizer_eval.period.chain import FINNEY_RPC, Chain
from humanizer_eval.period.lifecycle import Lifecycle, LifecycleConfig

CROWN_TAG = "gptzero-pangram"


def human_cache_path(pdir: Path, tag: str = CROWN_TAG) -> Path:
    return pdir / f"human_scores.{tag}.enc"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=Path, default=None, help="the intake state dir (queue, submissions)")
    ap.add_argument("--n", type=int, default=500, help="texts per period")
    ap.add_argument("--pool", default="pool-full-v2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--detectors-tag", default=CROWN_TAG, help="which human-score file the reveal reads")
    ap.add_argument("--rpc", default=FINNEY_RPC, help="finney JSON-RPC node (seed blocks)")
    ap.add_argument("--hf-repo", default=None, help="also publish commitments and reveals to <org>/<repo>")
    ap.add_argument("--hf-repo-type", default="model")
    ap.add_argument(
        "--hf-prefix", default="periods", help="folder in the HF repo (a rehearsal: rehearsal/periods)"
    )
    ap.add_argument("--period-days", type=float, default=7.0)
    ap.add_argument("--netuid", type=int, default=32)
    ap.add_argument("--notary-wallet", default=None, help="wallet that posts the remarks (period/notary.py)")
    ap.add_argument("--notary-path", default=None, help="its wallets dir (default var/keys/wallets)")
    ap.add_argument("--notary-python", default=None, help="a Python with bittensor (the validator env)")
    ap.add_argument("--notary-network", default="finney", help="must be the chain of --rpc")
    ap.add_argument("--recommit", default=None, metavar="REASON", help="abandon the unbuilt next commitment")
    ap.add_argument(
        "--status", action="store_true", help="print the schedule and what is due; change nothing"
    )
    args = ap.parse_args()

    s = get_settings()
    periods = var_dir() / "periods"
    state = args.state or (var_dir() / "intake")
    public = var_dir() / "public" / "periods"
    cfg = LifecycleConfig(period_s=args.period_days * 86400, netuid=args.netuid)
    chain = Chain(args.rpc)

    if args.status:
        lc = Lifecycle(periods, chain, builder=lambda *a: {}, cfg=cfg)
        sch = lc.schedule()
        head = chain.head()
        print(json.dumps({"head": head, **sch}, indent=1, default=str))
        nxt = sch.get("next")
        if nxt and not sch["periods"][nxt].get("built_at"):
            left = sch["periods"][nxt]["seed_block"] + cfg.finality_blocks - head
            print(f"{nxt}: build in {max(0, left)} blocks (~{max(0, left) * cfg.block_s / 3600:.1f} h)")
        cur = sch.get("current")
        if cur:
            end = sch["periods"][cur]["started_at"] + cfg.period_s
            print(f"{cur}: ends in {(end - time.time()) / 3600:.1f} h")
        return 0

    from humanizer_eval.ops.alerts import from_settings

    alerter = from_settings(state_path=var_dir() / "ops" / "alerts_state.json")

    def alert(text: str, dedupe: str | None = None) -> None:
        alerter.send(text, dedupe=dedupe)

    def builder(pid: str, secret: bytes, block_hash: str, block: int) -> dict:
        from humanizer_eval.budget import SpendLedger
        from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
        from humanizer_eval.paths import ledger_path
        from humanizer_eval.period.build import build_period, save_period
        from humanizer_eval.period.crypto import load_or_create_key
        from humanizer_eval.period.generate import GenConfig
        from humanizer_eval.period.pool_adapter import PoolAdapter

        led = SpendLedger(ledger_path(), s.caps())
        llm = OpenRouter(s.openrouter_api_key, led, pricing=ModelPricing(var_dir() / "openrouter_models.json"),
                         log_path=var_dir() / "llm_calls.jsonl")  # fmt: skip
        from humanizer_eval.llm.ratelimit import limited

        llm = limited(llm)  # OpenRouter's per-model limit: no 429s while building
        gen = GenConfig.from_dict(json.loads(Path("configs/period.json").read_text()))
        pool = PoolAdapter.load(var_dir() / args.pool)
        res = build_period(pid, secret, block_hash, pool, llm, gen, n=args.n, workers=args.workers)
        res.manifest.update(block=block, pool=args.pool)
        tmp = periods / f".{pid}.building"
        shutil.rmtree(tmp, ignore_errors=True)
        save_period(res, tmp, load_or_create_key(var_dir() / "keys" / "period.key"))
        sp = tmp / "secret.hex"
        from humanizer_eval.period.crypto import write_private

        write_private(sp, secret.hex())
        tmp.replace(periods / pid)
        return res.manifest

    def running(pid: str) -> int:
        q = state / "queue.sqlite"
        if not q.exists():
            return 0
        with sqlite3.connect(f"file:{q}?mode=ro", uri=True) as c:
            # 'paused' too: a charged run waiting for its crown is still deciding on this period's inputs
            return int(c.execute("SELECT COUNT(*) FROM submissions WHERE state IN ('queued', 'running', 'paused') "
                                 "AND period_id = ?", (pid,)).fetchone()[0])  # fmt: skip

    def revealer(pid: str) -> dict:
        from humanizer_eval.evaluation.context import load_human_scores
        from humanizer_eval.period.crypto import load_or_create_key
        from humanizer_eval.period.reveal import write_reveal

        key = load_or_create_key(var_dir() / "keys" / "period.key")
        pdir = periods / pid
        hs = load_human_scores(human_cache_path(pdir, args.detectors_tag), key)
        if hs is None:
            raise RuntimeError(f"{pid}: no human scores saved by the server ({args.detectors_tag})")
        sdir = state / "periods" / pid
        rec = json.loads((periods / "schedule.json").read_text())["periods"].get(pid)
        cur = human_cache_path(pdir, args.detectors_tag)  # H as scored before a rescore_period.py run
        older = [h for f in sorted(pdir.glob(cur.name + ".stale-*")) if (h := load_human_scores(f, key))]
        import subprocess

        from humanizer_eval.pool.store import PoolStore

        want = set(hs["ids"])
        texts: dict[str, str] = {}
        for name in dict.fromkeys([*args.pool.split(","), "pool-full-v2", "pool-full"]):  # the server's pools
            if (var_dir() / name).exists():
                texts.update({r["text_id"]: r["text"] for r in PoolStore.load(var_dir() / name).rows
                              if r["text_id"] in want and r["text_id"] not in texts})  # fmt: skip
        commit = (
            subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            or None
        )
        return write_reveal(pid, period_dir=pdir, submissions_dir=sdir / "submissions",
                            king_runs=sorted(sdir.glob("king_run.*.enc")), human_scores=hs, key=key,
                            out_dir=public / pid, schedule_record=rec, older_human_scores=older,
                            human_texts=texts, code_commit=commit)  # fmt: skip

    publish = None
    if args.hf_repo:
        if not s.hf_mirror_token:
            raise SystemExit("--hf-repo needs HF_MIRROR_TOKEN")
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi(token=s.hf_mirror_token)

        def publish(kind: str, pid: str) -> None:
            try:
                if kind == "schedule":
                    api.create_commit(repo_id=args.hf_repo, repo_type=args.hf_repo_type,
                                      operations=[CommitOperationAdd(f"{args.hf_prefix}/schedule.json",
                                                                     str(periods / "schedule.json"))],
                                      commit_message=f"SN32 humanizer periods: {pid}")  # fmt: skip
                else:
                    api.upload_folder(repo_id=args.hf_repo, repo_type=args.hf_repo_type,
                                      folder_path=str(public / pid), path_in_repo=f"{args.hf_prefix}/{pid}",
                                      commit_message=f"SN32 humanizer period {pid} revealed")  # fmt: skip
            except Exception as e:  # the API copy is the primary one; HF is retried on the next change
                alert(
                    f"publishing {kind} of {pid} to HF failed: {type(e).__name__}", dedupe=f"hf-{kind}-{pid}"
                )

    notary = None
    if args.notary_wallet:
        from humanizer_eval.period.notary import SubprocessNotary

        if not args.notary_python:
            raise SystemExit("--notary-wallet needs --notary-python (an interpreter with bittensor)")
        notary = SubprocessNotary(args.notary_python, args.notary_wallet,
                                  args.notary_path or str(var_dir() / "keys" / "wallets"), args.notary_network)  # fmt: skip
    public.mkdir(parents=True, exist_ok=True)
    lc = Lifecycle(periods, chain, builder, revealer=revealer, running=running, publish=publish, alert=alert,
                   notary=notary, cfg=cfg)  # fmt: skip
    if args.recommit:
        print(json.dumps({"abandoned": lc.recommit(args.recommit)}))
    try:
        done = lc.step()
    except Exception as e:
        alert(f"period rotation failed: {type(e).__name__}: {e}", dedupe="rotate-failed")
        raise
    print(
        json.dumps({"done": done, "current": lc.schedule().get("current"), "next": lc.schedule().get("next")})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
