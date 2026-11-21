"""G1 sanity check with the production judges (configs/gates.json) on the G1 calibration variants (report
04b): must-pass edits stay at level 1, must-fail edits drop. Cheaper than calibrate_g1.py, which tries
every candidate judge; run it after any change to the G1 prompts.

    python scripts/check_g1_prod.py [--items 12]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.gates.judges import GateRunner
from humanizer_eval.gates.meaning import check_g1
from humanizer_eval.gates.normalize import normalize
from humanizer_eval.llm.cache import CachedLLM
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import load_or_create_key

EXPECT = {"light_edit": "pass", "paraphrase": "pass", "drop": "fail", "truncate": "fail", "contradict": "fail",
          "insert": "fail", "pad": "fail"}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=12)
    ap.add_argument("--out", default=str(var_dir() / "calibration" / "g1_prod_check.json"))
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    spent0 = led.spent("openrouter")
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    base = OpenRouter(s.openrouter_api_key, led, pricing=ModelPricing(var_dir() / "openrouter_models.json"),
                      log_path=var_dir() / "llm_calls.jsonl")  # fmt: skip
    llm = CachedLLM(base, var_dir() / "calibration" / "llm_cache.sqlite", key=key, cache_all=True)
    runner = GateRunner.from_config(llm, json.loads(Path("configs/gates.json").read_text()))
    cal = json.loads((var_dir() / "calibration" / "g1v2.json").read_text())["items"]
    ids = sorted(cal, key=int)[: args.items]
    jobs = [(i, v) for i in ids for v in EXPECT if v in cal[i]["variants"]]

    def run(job):
        i, v = job
        out = normalize(cal[i]["variants"][v]["text"])[0]
        r = check_g1(
            llm, runner.judges, runner.arbiter, cal[i]["text"], out, cal[i]["facts"], runner.config.g1
        )
        return {"item": i, "variant": v, "status": r.status, "level": r.level, "counts": r.counts}

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(run, jobs))
    per = defaultdict(list)
    for r in rows:
        per[r["variant"]].append(r)
    print(f"{'variant':11s} {'expect':6s} {'right':>6s}  levels")
    for v, exp in EXPECT.items():
        rs = [r for r in per[v] if r["status"] != "error"]
        if not rs:
            continue
        right = sum((r["level"] == 1.0) == (exp == "pass") for r in rs) / len(rs)
        print(
            f"{v:11s} {exp:6s} {right:6.2f}  {dict(sorted(Counter(r['level'] for r in rs).items(), reverse=True))}"
        )
    errors = sum(1 for r in rows if r["status"] == "error")
    Path(args.out).write_text(json.dumps({"rows": rows, "errors": errors}, indent=1))
    print(json.dumps({"errors": errors, "spent": round(led.spent("openrouter") - spent0, 3)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
