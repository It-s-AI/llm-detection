"""Pilot: build one real period from the dev pool and run the detector admission test (D-31).

Costs (caps enforced by the ledger): generation ~$0.5 on OpenRouter; detectors ~$30 for 2 x n texts.
Usage: python scripts/pilot_period.py --pool <pool_dir> --n 80 --yes
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.detectors.registry import build_detector
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.build import build_period, save_period
from humanizer_eval.period.crypto import load_or_create_key
from humanizer_eval.period.generate import GenConfig
from humanizer_eval.period.pool_adapter import PoolAdapter
from humanizer_eval.period.seed import make_secret
from humanizer_eval.scoring.metric import auroc, stratified_auroc_detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--detectors", default="its_ai,zerogpt,gptzero,pangram")
    ap.add_argument("--yes", action="store_true", help="also run paid detectors")
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    llm = OpenRouter(
        s.openrouter_api_key,
        led,
        pricing=ModelPricing(var_dir() / "openrouter_models.json"),
        log_path=var_dir() / "llm_calls.jsonl",
    )
    cfg = GenConfig.from_dict(json.loads(Path("configs/period.json").read_text()))
    pool = PoolAdapter.load(args.pool)
    secret = make_secret()
    block = hashlib.sha256(b"pilot-no-chain-yet").hexdigest()  # placeholder until on-chain commits exist
    res = build_period("pilot-1", secret, block, pool, llm, cfg, n=args.n, workers=8)
    out_dir = var_dir() / "periods" / "pilot-1"
    save_period(res, out_dir, load_or_create_key(var_dir() / "keys" / "period.key"))
    (out_dir / "secret.hex").write_text(secret.hex())  # revealed at release time
    print(
        "manifest:", json.dumps({k: v for k, v in res.manifest.items() if k != "config_sha256"}, default=str)
    )
    ai = [it["text"] for it in res.items]
    ai_strata = [f"{it['slot']['genre']}|{it['slot']['length_bin']}" for it in res.items]
    hu = [h["text"] for h in res.human_ref]
    hu_strata = [f"{h['genre']}|{h['length_bin']}" for h in res.human_ref]
    summary = {
        "manifest": res.manifest,
        "n_ai": len(ai),
        "n_human": len(hu),
        "human_match": dict(Counter(h.get("match") for h in res.human_ref)),
        "ai_words_mean": float(np.mean([len(t.split()) for t in ai])),
        "human_words_mean": float(np.mean([len(t.split()) for t in hu])),
        "detectors": {},
    }
    for name in args.detectors.split(","):
        if name != "its_ai" and not args.yes:
            continue
        det = build_detector(name, ledger=led)
        sa, sh = det.score(ai), det.score(hu)
        ok_a = [i for i, x in enumerate(sa) if x.ok]
        ok_h = [i for i, x in enumerate(sh) if x.ok]
        a_scores = np.array([sa[i].score for i in ok_a])
        h_scores = np.array([sh[i].score for i in ok_h])
        detail = stratified_auroc_detail(
            a_scores, [ai_strata[i] for i in ok_a], h_scores, [hu_strata[i] for i in ok_h]
        )
        per_genre = {}
        for g in sorted({x.split("|")[0] for x in ai_strata}):
            ia = [k for k, i in enumerate(ok_a) if ai_strata[i].startswith(g + "|")]
            ih = [k for k, i in enumerate(ok_h) if hu_strata[i].startswith(g + "|")]
            if ia and ih:
                per_genre[g] = round(float(auroc(a_scores[ia], h_scores[ih])), 4)
        cands = {}
        for key in sa[ok_a[0]].candidates if ok_a else {}:
            ca = np.array([sa[i].candidates.get(key, np.nan) for i in ok_a])
            ch = np.array([sh[i].candidates.get(key, np.nan) for i in ok_h])
            if np.isfinite(ca).all() and np.isfinite(ch).all():
                cands[key] = round(float(auroc(ca, ch)), 4)
        summary["detectors"][name] = {
            "errors_ai": len(sa) - len(ok_a),
            "errors_human": len(sh) - len(ok_h),
            "auroc_raw_ai_vs_human": round(float(auroc(a_scores, h_scores)), 4),
            "auroc_stratified": round(float(detail.value), 4),
            "excluded_strata": list(detail.excluded),
            "admitted": bool(detail.value >= 0.9),
            "per_genre_auroc": per_genre,
            "candidate_auroc": cands,
            "mean_ai": round(float(a_scores.mean()), 4),
            "mean_human": round(float(h_scores.mean()), 4),
        }
        print(name, json.dumps(summary["detectors"][name]))
    summary["ledger"] = led.summary()
    Path("docs/reports/data").mkdir(parents=True, exist_ok=True)
    Path("docs/reports/data/03-pilot.json").write_text(json.dumps(summary, indent=1, default=str))
    print("ledger:", json.dumps(led.summary(), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
