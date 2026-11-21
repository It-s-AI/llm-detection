"""Gate and score a submission's saved outputs on a pilot period (G1-G3 levels, then crown detectors).

Outputs are read from ``var/periods/<period>/<name>.enc`` (a JSON list aligned with the period items, as the
reference-humanizer scripts write them). Detector scores of H and of the inputs come from the detector cache;
only the new outputs are paid. Judge replies are cached too.
Usage: python scripts/score_outputs.py --name ref_v2 [--period pilot-1] [--paid pangram] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.detectors.registry import build_detector
from humanizer_eval.gates.judges import GateRunner, Judge
from humanizer_eval.gates.meaning import prepare_facts
from humanizer_eval.gates.normalize import apply_length_limit, normalize
from humanizer_eval.llm.cache import CachedLLM
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, load_or_create_key
from humanizer_eval.pool.store import PoolStore
from humanizer_eval.scoring.metric import HumanReference, SubmissionRun, submission_score

EXTRACTOR = "anthropic/claude-haiku-5.5"
VALIDATORS = ("anthropic/claude-sonnet-5.5", "openai/gpt-6.1-sol")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="outputs file stem in the period dir, e.g. ref_v2")
    ap.add_argument("--period", default="pilot-1")
    ap.add_argument(
        "--pool", default="pool-full-v2,pool-full", help="pool dirs in var/ to look H texts up in"
    )
    ap.add_argument("--paid", default="", help="comma-separated crown detectors to run, e.g. pangram,gptzero")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    spent0 = {k: v["spent"] for k, v in led.summary().items()}
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    base = OpenRouter(s.openrouter_api_key, led, pricing=ModelPricing(var_dir() / "openrouter_models.json"),
                      log_path=var_dir() / "llm_calls.jsonl")  # fmt: skip
    llm = CachedLLM(base, var_dir() / "calibration" / "llm_cache.sqlite", key=key, cache_all=True)
    pdir = var_dir() / "periods" / args.period
    data = json.loads(decrypt_at_rest(key, (pdir / "dataset.enc").read_bytes()))
    items = data["items"]
    raw = json.loads(decrypt_at_rest(key, (pdir / f"{args.name}.enc").read_bytes()))
    outs = [
        apply_length_limit(normalize(o or "")[0], it["text"])[0] for o, it in zip(raw, items, strict=True)
    ]
    want = {h["id"] for h in data["human_ref"]}
    pool = {}
    for name in args.pool.split(","):
        pool.update({r["text_id"]: r for r in PoolStore.load(var_dir() / name).rows if r["text_id"] in want})
    human = [pool[h["id"]] for h in data["human_ref"] if h["id"] in pool]
    print(f"H: {len(human)} of {len(data['human_ref'])} human reference texts found in {args.pool}")

    runner = GateRunner.from_config(llm, json.loads(Path("configs/gates.json").read_text()))

    def gate(i: int):
        if not outs[i].strip():
            return None
        facts, _ = prepare_facts(
            llm, Judge(EXTRACTOR), (Judge(VALIDATORS[0]), Judge(VALIDATORS[1])), items[i]["text"]
        )
        return runner.check(items[i]["text"], outs[i], facts or [])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        gres = list(ex.map(gate, range(len(items))))
    levels = np.array([0.0 if g is None or g.status == "error" else g.level for g in gres])
    strata = [f"{it['slot']['genre']}|{it['slot']['length_bin']}" for it in items]
    hstrata = [f"{h['genre']}|{h['length_bin']}" for h in human]
    report = {
        "name": args.name,
        "n": len(items),
        "n_human": len(human),
        "empty": int(sum(1 for o in outs if not o.strip())),
        "levels": {str(k): int((levels == k).sum()) for k in (1.0, 0.5, 0.0)},
        "gate_reasons": [g.reasons if g else ["empty"] for g in gres],
        "words_ratio": float(
            np.mean(
                [len(o.split()) / max(1, len(it["text"].split())) for o, it in zip(outs, items, strict=True)]
            )
        ),
        "detectors": {},
    }
    for d in ["its_ai", *[x for x in args.paid.split(",") if x]]:
        det = build_detector(d, ledger=led)
        sh = [x.score for x in det.score([h["text"] for h in human])]
        sa = [x.score for x in det.score([it["text"] for it in items])]
        so = [x.score if x.ok else np.inf for x in det.score([o or "-" for o in outs])]
        H = HumanReference({d: sh}, hstrata)
        res = {}
        for label, scores, q in (("identity", sa, np.ones(len(items))), (args.name, so, levels),
                                 (f"{args.name}_no_gates", so, np.ones(len(items)))):  # fmt: skip
            r = submission_score(SubmissionRun({d: scores}, q, strata), H)
            res[label] = {"S": r.s, "auroc": r.auroc[d], "p_human": r.p_human[d],
                          "mean_score": float(np.mean([x for x in scores if np.isfinite(x)]))}  # fmt: skip
        report["detectors"][d] = res
    report["spent"] = {k: round(v["spent"] - spent0.get(k, 0.0), 3) for k, v in led.summary().items()
                       if v["spent"] - spent0.get(k, 0.0) > 0}  # fmt: skip
    out = Path(args.out or f"docs/reports/data/07-{args.name}.json")
    out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "gate_reasons"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
