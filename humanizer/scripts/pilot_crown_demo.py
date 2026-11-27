"""Crown demo on the pilot period: king = identity baseline, challenger = one-pass LLM rewrite.

Runs G0 normalization, gates (cheap judges), detectors (cached where possible) and the paired
bootstrap crown decision. Costs: rewrite + gates ~$0.5 (OpenRouter); detectors ~$16 for the
challenger outputs (king outputs and human texts are cached from the pilot).
Usage: python scripts/pilot_crown_demo.py --pool <pool_dir> --yes
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from humanizer_eval.baselines.llm_rewrite import rewrite_all
from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.detectors.registry import build_detector
from humanizer_eval.gates.judges import GateRunner, Judge
from humanizer_eval.gates.normalize import apply_length_limit, normalize
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, load_or_create_key
from humanizer_eval.pool.store import PoolStore
from humanizer_eval.scoring.crown import crown_decision
from humanizer_eval.scoring.metric import HumanReference, SubmissionRun, submission_score

CROWN = ("gptzero", "zerogpt", "pangram")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--period", default="pilot-1")
    ap.add_argument("--rewriter", default="openai/gpt-5.4-mini")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--detectors", default="gptzero,zerogpt,pangram", help="paid crown detectors to use")
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    llm = OpenRouter(
        s.openrouter_api_key,
        led,
        pricing=ModelPricing(var_dir() / "openrouter_models.json"),
        log_path=var_dir() / "llm_calls.jsonl",
    )
    pdir = var_dir() / "periods" / args.period
    data = json.loads(
        decrypt_at_rest(
            load_or_create_key(var_dir() / "keys" / "period.key"), (pdir / "dataset.enc").read_bytes()
        )
    )
    items = data["items"]
    pool = {r["text_id"]: r for r in PoolStore.load(args.pool).rows}
    human = [pool[h["id"]] for h in data["human_ref"] if h["id"] in pool]
    inputs = [it["text"] for it in items]
    strata = [f"{it['slot']['genre']}|{it['slot']['length_bin']}" for it in items]
    h_strata = [f"{h['genre']}|{h['length_bin']}" for h in human]
    # submissions
    rewritten, rw_cost = rewrite_all(llm, args.rewriter, inputs)
    subs = {"king_identity": inputs, "challenger_rewrite": rewritten}
    gates = GateRunner(llm, (Judge("anthropic/claude-haiku-5.5"), Judge("openai/gpt-6-luna")))
    report = {
        "n": len(items),
        "n_human": len(human),
        "rewriter": args.rewriter,
        "rewrite_cost": rw_cost,
        "subs": {},
    }
    runs, human_scores = {}, {}
    for name, outs in subs.items():
        prepared = [apply_length_limit(normalize(o)[0], src)[0] for o, src in zip(outs, inputs, strict=True)]

        def gate(i, prepared=prepared):
            return gates.check(inputs[i], prepared[i], items[i]["facts"])

        with ThreadPoolExecutor(max_workers=6) as ex:
            gr = list(ex.map(gate, range(len(items))))
        # quality level per output (D-22); a judge error is re-run in production, here it counts as 0
        gate_pass = np.array([g.level if g.status != "error" else 0.0 for g in gr])
        scores = {}
        crown = tuple(x for x in args.detectors.split(",") if x)
        for d in (*crown, "its_ai"):
            if d in crown and not args.yes:
                continue
            det = build_detector(d, ledger=led)
            res = det.score(prepared)
            scores[d] = np.array([x.score if x.ok else np.inf for x in res])  # failed request -> caught
            if d not in human_scores:
                hs = det.score([h["text"] for h in human])
                human_scores[d] = np.array([x.score if x.ok else np.nan for x in hs])
        runs[name] = (scores, gate_pass)
        report["subs"][name] = {
            "gate_pass_rate": float(gate_pass.mean()),
            "gate_status_counts": {k: sum(g.status == k for g in gr) for k in ("pass", "fail", "error")},
            "gate_cost": round(sum(g.cost_usd for g in gr), 4),
            "mean_scores": {d: float(np.mean(v[np.isfinite(v)])) for d, v in scores.items()},
            "words_mean": float(np.mean([len(p.split()) for p in prepared])),
        }
    dets_sets = {
        "crown": [d for d in args.detectors.split(",") if d in human_scores],
        "proxy_its_ai": ["its_ai"],
    }
    for label, dets in dets_sets.items():
        if not dets:
            continue
        ok_h = np.all([np.isfinite(human_scores[d]) for d in dets], axis=0)
        H = HumanReference({d: human_scores[d][ok_h] for d in dets}, np.array(h_strata)[ok_h])
        sr = {n: SubmissionRun({d: sc[d] for d in dets}, gp, strata) for n, (sc, gp) in runs.items()}
        out = {}
        for n, r in sr.items():
            sc = submission_score(r, H)
            out[n] = {
                "S": float(sc.s),
                "E_d": {k: float(v) for k, v in sc.evasion.items()},
                "auroc": {k: float(v) for k, v in sc.auroc.items()},
                "gate_pass_rate": float(sc.gate_pass_rate),
            }
        dec = crown_decision(
            sr["king_identity"], sr["challenger_rewrite"], H, alpha=0.05, n_boot=2000, seed=1
        )
        out["crown"] = {
            "decision": bool(dec.crown),
            "delta": float(dec.delta),
            "lower": float(dec.lower),
            "se": float(dec.se),
            "reasons": list(dec.reasons),
        }
        report[label] = out
        print(label, json.dumps(out, default=str)[:900])
    report["ledger"] = led.summary()
    Path("docs/reports/data/03-crown-demo.json").write_text(json.dumps(report, indent=1, default=str))
    print("subs:", json.dumps(report["subs"], default=str)[:900])
    print("ledger:", json.dumps(led.summary(), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
