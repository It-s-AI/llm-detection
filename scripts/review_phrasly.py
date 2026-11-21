"""Gate saved outputs of a pilot (Phrasly by default; --outputs ref_v2 for the reference humanizer) with
G1 v2 + G2 + G3 and save per-item judge detail for human review.

Facts are prepared (anchored + validated) with the same models and cache as the calibration, so items
shared with the calibration cost nothing. ~$0.02-0.04 per item.
Usage: python scripts/review_phrasly.py --judges A,B --arbiter C [--period pilot-1]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.gates.judges import GateRunner, Judge
from humanizer_eval.gates.meaning import prepare_facts
from humanizer_eval.gates.normalize import normalize
from humanizer_eval.llm.cache import CachedLLM
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, load_or_create_key

EXTRACTOR = "anthropic/claude-haiku-5.5"
VALIDATORS = ("anthropic/claude-sonnet-5.5", "openai/gpt-6.1-sol")
JUDGE_KW = {"qwen/qwen3.8-flash": {"max_tokens": 12000}}


def J(m: str) -> Judge:
    return Judge(m, **JUDGE_KW.get(m, {}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="pilot-1")
    ap.add_argument("--judges", required=True)
    ap.add_argument("--arbiter", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--outputs", default="phrasly", help="saved outputs in the period dir: phrasly | ref_v2 | ..."
    )
    ap.add_argument("--out", default=str(var_dir() / "calibration" / "phrasly_review.json"))
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    spent0 = led.spent("openrouter")
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    base = OpenRouter(
        s.openrouter_api_key,
        led,
        pricing=ModelPricing(var_dir() / "openrouter_models.json"),
        log_path=var_dir() / "llm_calls.jsonl",
    )
    llm = CachedLLM(base, var_dir() / "calibration" / "llm_cache.sqlite", key=key, cache_all=True)
    pdir = var_dir() / "periods" / args.period
    items = json.loads(decrypt_at_rest(key, (pdir / "dataset.enc").read_bytes()))["items"]
    saved = json.loads(decrypt_at_rest(key, (pdir / f"{args.outputs}.enc").read_bytes()))
    phrasly = saved if isinstance(saved, dict) else {str(i): t for i, t in enumerate(saved)}
    a, b = (J(m) for m in args.judges.split(","))
    runner = GateRunner(llm, (a, b), arbiter=J(args.arbiter))

    def one(i: int):
        src = items[i]["text"]
        facts, rep = prepare_facts(llm, J(EXTRACTOR), (J(VALIDATORS[0]), J(VALIDATORS[1])), src)
        if not facts:
            return {"item": i, "status": "error", "reasons": [rep.error]}
        out = normalize(phrasly[str(i)])[0]
        r = runner.check(src, out, facts)
        return {
            "item": i,
            "genre": items[i]["slot"]["genre"],
            "generator": items[i]["generator"]["model"],
            "input": src,
            "output": out,
            "status": r.status,
            "gates": r.gates,
            "level": r.level,
            "levels": r.levels,
            "reasons": r.reasons,
            "g1": r.detail.get("g1"),
            "style": r.detail.get("style"),
            "natural": r.detail.get("natural"),
            "facts_dropped": rep.dropped,
        }

    idx = [i for i in range(len(items)) if str(i) in phrasly]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(one, idx))
    Path(args.out).write_text(
        json.dumps({"judges": args.judges, "arbiter": args.arbiter, "rows": rows}, indent=1)
    )
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    fails = {}
    for r in rows:
        for g, v in (r.get("gates") or {}).items():
            if v is False:
                fails[g] = fails.get(g, 0) + 1
    print(json.dumps({"status": st, "failed_gate": fails, "cache": {"hits": llm.hits, "misses": llm.misses},
                      "spent": round(led.spent("openrouter") - spent0, 3)}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
