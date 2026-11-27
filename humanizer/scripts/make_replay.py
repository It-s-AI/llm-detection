"""Saved outputs of a stand-in challenger for a period: a one-pass LLM rewrite of every input, encrypted
as <period dir>/<name>.enc, so that evaluate_submission.py --replay <name> can test the challenger path
(gates, detectors, crown) without a pod. Owner tests only; never a real submission.

    python scripts/make_replay.py --period p1 --name rewrite_v1 [--rewriter openai/gpt-5.4-mini]

Cost: the rewrite on OpenRouter, about $0.005 per text (ledger caps apply).
"""

from __future__ import annotations

import argparse
import json

from humanizer_eval.baselines.llm_rewrite import rewrite_all
from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, encrypt_at_rest, load_or_create_key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--rewriter", default="openai/gpt-5.4-mini")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    llm = OpenRouter(s.openrouter_api_key, led, pricing=ModelPricing(var_dir() / "openrouter_models.json"),
                     log_path=var_dir() / "llm_calls.jsonl")  # fmt: skip
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    pdir = var_dir() / "periods" / args.period
    out = pdir / f"{args.name}.enc"
    if out.exists():
        raise SystemExit(f"{out} exists")
    items = json.loads(decrypt_at_rest(key, (pdir / "dataset.enc").read_bytes()))["items"]
    texts, cost = rewrite_all(llm, args.rewriter, [it["text"] for it in items], workers=args.workers)
    out.write_bytes(encrypt_at_rest(key, json.dumps(texts).encode()))
    empty = sum(1 for t in texts if not t.strip())
    print(json.dumps({"period": args.period, "name": args.name, "n": len(texts), "empty": empty,
                      "rewrite_cost_usd": round(cost, 3)}))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
