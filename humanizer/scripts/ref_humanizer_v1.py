"""Reference humanizer v1 (feasibility, report 07): few-shot style transfer + best-of-N by a proxy detector.

For every pilot input: show a strong LLM three human texts of the same genre/length from the pool (never
the period's reference set H) and ask for a rewrite in their style that keeps every fact, the order of
ideas, register and paragraphs. Sample N candidates, pick the one It's AI (free, not a crown detector)
finds most human, then gate it (G1-G3) and score it on the crown detectors.
Usage: python scripts/ref_humanizer_v1.py --pool <pool_dir> [--paid gptzero,zerogpt] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
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
from humanizer_eval.llm.openrouter import HIDDEN, ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, encrypt_at_rest, load_or_create_key
from humanizer_eval.pool.store import PoolStore
from humanizer_eval.scoring.metric import HumanReference, SubmissionRun, submission_score

WRITERS = [("anthropic/claude-sonnet-5.5", (0.7, 0.9, 1.0)), ("openai/gpt-6.1-terra", (0.7, 0.9, 1.0))]
PROMPT = (
    "Below are three texts written by people, then a {genre} written by an AI. Rewrite the AI text so it "
    "reads like the people's texts: their sentence rhythm, word choice, how polished they are, how they "
    "move between ideas. Rules: keep every piece of information (claims, names, numbers, examples, "
    "reasons) and the order of ideas; add nothing new; keep the register, the person and tense, and the "
    "paragraph breaks; do not introduce spelling or grammar errors; similar length. Output only the "
    "rewritten text.\n\n{examples}\n\n<ai_text>\n{text}\n</ai_text>"
)
EXTRACTOR = "anthropic/claude-haiku-5.5"
VALIDATORS = ("anthropic/claude-sonnet-5.5", "openai/gpt-6.1-sol")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--period", default="pilot-1")
    ap.add_argument(
        "--paid", default="gptzero,zerogpt", help="crown detectors to run on the selected outputs"
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="docs/reports/data/07-ref-v1.json")
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    base = OpenRouter(
        s.openrouter_api_key,
        led,
        pricing=ModelPricing(var_dir() / "openrouter_models.json"),
        log_path=var_dir() / "llm_calls.jsonl",
    )
    llm = CachedLLM(base, var_dir() / "calibration" / "llm_cache.sqlite", key=key, cache_all=True)
    pdir = var_dir() / "periods" / args.period
    data = json.loads(decrypt_at_rest(key, (pdir / "dataset.enc").read_bytes()))
    items = data["items"][: args.limit]
    hidden = {h["id"] for h in data["human_ref"]}
    rows = PoolStore.load(args.pool).rows
    by_slot = defaultdict(list)
    for r in rows:
        if r["text_id"] not in hidden:
            by_slot[(r["genre"], r["length_bin"])].append(r["text"])
    pool = {r["text_id"]: r for r in rows}
    human = [pool[h["id"]] for h in data["human_ref"] if h["id"] in pool]

    def candidates(i: int) -> list[str]:
        it = items[i]
        g, lb = it["slot"]["genre"], it["slot"]["length_bin"]
        ex = random.Random(i).sample(by_slot[(g, lb)], 3)
        examples = "\n\n".join(f'<human_text n="{k + 1}">\n{t}\n</human_text>' for k, t in enumerate(ex))
        prompt = PROMPT.format(genre=g.replace("_", " "), examples=examples, text=it["text"])
        outs = []
        for model, temps in WRITERS:
            for t in temps:
                try:
                    r = llm.chat(
                        model,
                        [{"role": "user", "content": prompt}],
                        max_tokens=3000,
                        temperature=t,
                        policy=HIDDEN,
                        purpose="ref-v1:rewrite",
                        reasoning={"effort": "low"},
                    )
                    o, _ = apply_length_limit(normalize(r.text.strip())[0], it["text"])
                    if o.strip():
                        outs.append(o)
                except RuntimeError:
                    continue
        return outs

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        cands = list(ex.map(candidates, range(len(items))))
    itsai = build_detector("its_ai", ledger=led)
    flat = [(i, c) for i, cs in enumerate(cands) for c in cs]
    sc = [x.score if x.ok else 1.0 for x in itsai.score([c for _, c in flat])]
    best: dict[int, tuple[float, str]] = {}
    for (i, c), v in zip(flat, sc, strict=True):
        if i not in best or v < best[i][0]:
            best[i] = (v, c)
    chosen = [best[i][1] if i in best else items[i]["text"] for i in range(len(items))]
    (pdir / "ref_v1.enc").write_bytes(encrypt_at_rest(key, json.dumps(chosen).encode()))

    runner = GateRunner.from_config(llm, json.loads(Path("configs/gates.json").read_text()))

    def gate(i: int):
        facts, _ = prepare_facts(
            llm, Judge(EXTRACTOR), (Judge(VALIDATORS[0]), Judge(VALIDATORS[1])), items[i]["text"]
        )
        r = runner.check(items[i]["text"], chosen[i], facts or [])
        return r

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        gres = list(ex.map(gate, range(len(items))))
    levels = np.array([g.level if g.status != "error" else 0.0 for g in gres])

    strata = [f"{it['slot']['genre']}|{it['slot']['length_bin']}" for it in items]
    hstrata = [f"{h['genre']}|{h['length_bin']}" for h in human]
    dets = ["its_ai", *[d for d in args.paid.split(",") if d]]
    report = {
        "n": len(items),
        "candidates_per_text": float(np.mean([len(c) for c in cands])),
        "levels": {str(k): int((levels == k).sum()) for k in (1.0, 0.5, 0.0)},
        "gate_reasons": [g.reasons for g in gres],
        "detectors": {},
    }
    for d in dets:
        det = build_detector(d, ledger=led)
        sh = [x.score for x in det.score([h["text"] for h in human])]
        sa = [x.score for x in det.score([it["text"] for it in items])]
        so = [x.score if x.ok else np.inf for x in det.score(chosen)]
        H = HumanReference({d: sh}, hstrata)
        res = {}
        for name, scores, q in (
            ("identity", sa, np.ones(len(items))),
            ("ref_v1", so, levels),
            ("ref_v1_no_gates", so, np.ones(len(items))),
        ):
            r = submission_score(SubmissionRun({d: scores}, q, strata), H)
            res[name] = {
                "S": r.s,
                "auroc": r.auroc[d],
                "p_human": r.p_human[d],
                "mean_score": float(np.mean([x for x in scores if np.isfinite(x)])),
            }
        report["detectors"][d] = res
    report["ledger"] = {k: v for k, v in led.summary().items() if k in ("openrouter", *dets)}
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "gate_reasons"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
