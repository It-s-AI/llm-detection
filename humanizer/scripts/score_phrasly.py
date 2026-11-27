"""Score a pilot's Phrasly outputs with every detector and the current metric (D-20, D-22).

Human reference and raw AI inputs are cached from the pilot; only Phrasly outputs are new detector calls
(~78 x $0.21 for GPTZero + ZeroGPT + Pangram). Quality levels come from scripts/review_phrasly.py.
Usage: python scripts/score_phrasly.py --pool <pool_dir> [--levels var/calibration/phrasly_review_levels.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.detectors.registry import build_detector
from humanizer_eval.gates.normalize import normalize
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, load_or_create_key
from humanizer_eval.pool.store import PoolStore
from humanizer_eval.scoring.metric import HumanReference, SubmissionRun, submission_score

DETECTORS = ("gptzero", "zerogpt", "pangram", "its_ai")
CROWN = ("gptzero", "zerogpt", "pangram")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--period", default="pilot-1")
    ap.add_argument("--levels", default=str(var_dir() / "calibration" / "phrasly_review_levels.json"))
    ap.add_argument("--out", default="docs/reports/data/07-phrasly-detectors.json")
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    pdir = var_dir() / "periods" / args.period
    data = json.loads(decrypt_at_rest(key, (pdir / "dataset.enc").read_bytes()))
    items = data["items"]
    phrasly = json.loads(decrypt_at_rest(key, (pdir / "phrasly.enc").read_bytes()))
    pool = {r["text_id"]: r for r in PoolStore.load(args.pool).rows}
    human = [pool[h["id"]] for h in data["human_ref"] if h["id"] in pool]
    rows = {r["item"]: r for r in json.loads(Path(args.levels).read_text())["rows"]}
    levels = np.array([float(rows[i]["level"]) for i in range(len(items))])
    g1 = np.array([float(rows[i]["levels"].get("G1", 0.0)) for i in range(len(items))])
    strata = [f"{it['slot']['genre']}|{it['slot']['length_bin']}" for it in items]
    hstrata = [f"{h['genre']}|{h['length_bin']}" for h in human]
    outs = [normalize(phrasly[str(i)])[0] for i in range(len(items))]

    sc_ai, sc_ph, sc_h = {}, {}, {}
    for d in DETECTORS:
        det = build_detector(d, ledger=led)
        sc_h[d] = [x.score if x.ok else np.nan for x in det.score([h["text"] for h in human])]
        sc_ai[d] = [x.score if x.ok else np.inf for x in det.score([it["text"] for it in items])]
        sc_ph[d] = [x.score if x.ok else np.inf for x in det.score(outs)]
    report = {}
    for dets, tag in ((CROWN, "crown"), (("its_ai",), "its_ai")):
        keep_h = [i for i in range(len(human)) if all(np.isfinite(sc_h[d][i]) for d in dets)]
        H = HumanReference({d: [sc_h[d][i] for i in keep_h] for d in dets}, [hstrata[i] for i in keep_h])
        runs = {
            "identity": SubmissionRun({d: sc_ai[d] for d in dets}, np.ones(len(items)), strata),
            "phrasly_levels": SubmissionRun({d: sc_ph[d] for d in dets}, levels, strata),
            "phrasly_g1_levels": SubmissionRun({d: sc_ph[d] for d in dets}, g1, strata),
            "phrasly_no_gates": SubmissionRun({d: sc_ph[d] for d in dets}, np.ones(len(items)), strata),
        }
        report[tag] = {}
        for name, run in runs.items():
            r = submission_score(run, H)
            report[tag][name] = {
                "S": r.s,
                "auroc": r.auroc,
                "p_human": r.p_human,
                "E_d": r.evasion,
                "severe": r.severe_rate,
                "minor": r.minor_rate,
            }
    report["mean_scores"] = {
        d: {
            "ai": float(np.nanmean(np.where(np.isinf(sc_ai[d]), np.nan, sc_ai[d]))),
            "phrasly": float(np.nanmean(np.where(np.isinf(sc_ph[d]), np.nan, sc_ph[d]))),
            "human": float(np.nanmean(sc_h[d])),
        }
        for d in DETECTORS
    }
    report["ledger"] = led.summary()
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "ledger"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
