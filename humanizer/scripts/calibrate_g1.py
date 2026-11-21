"""G1 v2 calibration: which judge pair + arbiter separates must-pass from must-fail rewrites?

For a stratified sample of pilot inputs we build anchored, validated facts and these outputs:

  must pass : identity, typography, light_edit (LLM synonym edits, every sentence kept)
  must fail : drop (sentences removed), truncate (tail cut), contradict (one sentence made false),
              insert (one new specific claim), pad (5 content-free sentences)
  unlabelled: paraphrase (strong-LLM rewrite), phrasly (commercial humanizer) -> reviewed by people

Every candidate judge labels every open fact/sentence once; arbiter candidates label the union of
items any cross-vendor pair disputes. All pair x arbiter configs are then evaluated offline with the
production aggregation code (``meaning.run_two_family`` + ``decide_g1``). Replies are cached on disk
(encrypted), so re-running is free. Output: var/calibration/g1v2.json (+ printed table).
Usage: python scripts/calibrate_g1.py [--per-genre 3] [--limit N]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.gates.judges import Judge, _obj
from humanizer_eval.gates.meaning import (
    _CLASS,
    Fact,
    G1Config,
    decide_g1,
    plan_g1,
    precision_verdicts,
    prepare_facts,
    recall_verdicts,
    run_two_family,
    split_sentences,
)
from humanizer_eval.gates.normalize import normalize
from humanizer_eval.llm.cache import CachedLLM
from humanizer_eval.llm.openrouter import HIDDEN, ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, load_or_create_key

CANDIDATES = [
    "anthropic/claude-haiku-5.5",
    "openai/gpt-6-luna",
    "qwen/qwen3.8-flash",
    "google/gemini-3.8-flash",
    "openai/gpt-6.1-terra",
]
ARBITERS = ["anthropic/claude-sonnet-5.5", "openai/gpt-6.1-sol"]
EXTRACTOR = "anthropic/claude-haiku-5.5"
VALIDATORS = ("anthropic/claude-sonnet-5.5", "openai/gpt-6.1-sol")
VARIANT_MODEL = "x-ai/grok-4.7"  # not a judge candidate family
#: per-model judge settings found in the smoke run (qwen's reasoning overran 4096 tokens -> cut JSON)
JUDGE_KW = {"qwen/qwen3.8-flash": {"max_tokens": 12000}}


def J(model: str) -> Judge:
    return Judge(model, **JUDGE_KW.get(model, {}))


MUST_PASS = ("identity", "typography", "light_edit")
MUST_FAIL = ("drop", "truncate", "contradict", "insert", "pad")
SOFT = ("paraphrase", "phrasly")

LIGHT = (
    "Copy-edit the text lightly: replace roughly one word in six with a natural synonym or rephrase a few "
    "short phrases. Keep every sentence and every piece of information: all names, numbers, claims and "
    "the meaning stay exactly the same. Do not add, remove, merge or split sentences. Output only the "
    "edited text.\n\n<text>\n{text}\n</text>"
)
PARA = (
    "Rewrite the text in your own words so that it reads naturally, as if a person wrote it. Keep ALL "
    "information: every claim, number, name, example and reason, and the author's stance; add nothing "
    "new. Keep the language, genre, format, person and roughly the same length. Output only the "
    "rewrite.\n\n<text>\n{text}\n</text>"
)
CONTRA = (
    "The sentences of the text are numbered. Pick ONE sentence that states a concrete fact and rewrite it "
    "so that it clearly CONTRADICTS the original while still sounding natural: change a number, date or "
    "name, negate the claim, or swap who did what to whom. Return the sentence index, the rewritten "
    "sentence and the kind of change.\n\n<text>\n{numbered}\n</text>"
)
INSERT = (
    "The sentences of the text are numbered. Write ONE new sentence that would fit naturally right after "
    "one of them and adds a specific, plausible piece of information that is NOT in the text (a number, a "
    "named person, place or organisation, a date, or a study). Return the index it follows and the "
    "sentence.\n\n<text>\n{numbered}\n</text>"
)
PAD = (
    "Write 5 short sentences in the same voice and style as the text below that add NO information: "
    "generic remarks, transitions, the writer's attitude, rhetorical questions. They must not state any "
    "fact, number, name, example or reason.\n\n<text>\n{text}\n</text>"
)
CONTRA_SCHEMA = {
    "title": "edit",
    **_obj({"index": {"type": "integer"}, "new_sentence": {"type": "string"}, "kind": {"type": "string"}}),
}
INSERT_SCHEMA = {"title": "insert", **_obj({"after": {"type": "integer"}, "sentence": {"type": "string"}})}
PAD_SCHEMA = {"title": "pad", **_obj({"sentences": {"type": "array", "items": {"type": "string"}}})}


def vendor(m: str) -> str:
    return m.split("/")[0]


def numbered(sents):
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(sents))


def gen_text(llm, prompt, temperature=0.7):
    r = llm.chat(
        VARIANT_MODEL,
        [{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=temperature,
        policy=HIDDEN,
        purpose="calib:variant",
    )
    return r.text.strip()


def gen_json(llm, prompt, schema):
    r = llm.chat(
        VARIANT_MODEL,
        [{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.7,
        json_schema=schema,
        policy=HIDDEN,
        purpose="calib:variant",
    )
    return r.json if isinstance(r.json, dict) else None


def removed_facts(facts: list[Fact], removed: set[int]) -> int:
    return sum(1 for f in facts if f.sents and set(f.sents) <= removed)


def build_variants(llm, text: str, facts: list[Fact], phrasly: str | None, rng: random.Random):
    sents = split_sentences(text)
    allowed = G1Config().omitted_allowed(len(facts))
    v: dict[str, dict] = {"identity": {"text": text}}
    typo = text.replace("'", "’").replace(" - ", " — ").replace(". ", ".  ")
    v["typography"] = {"text": typo}
    v["light_edit"] = {"text": gen_text(llm, LIGHT.format(text=text), 0.3)}
    v["paraphrase"] = {"text": gen_text(llm, PARA.format(text=text))}
    if phrasly:
        v["phrasly"] = {"text": phrasly}
    # drop: random content sentences until clearly more facts are gone than allowed
    order = [i for i in range(len(sents)) if any(i in f.sents for f in facts)]
    rng.shuffle(order)
    removed: set[int] = set()
    for i in order:
        removed.add(i)
        if removed_facts(facts, removed) >= allowed + 2:
            out = text
            for k in removed:
                out = out.replace(sents[k], "", 1)
            v["drop"] = {"text": re.sub(r"[ \t]{2,}", " ", out).strip(), "removed": sorted(removed)}
            break
    # truncate: cut the tail at the latest point that clearly loses too many facts
    for cut in range(len(sents) - 1, 0, -1):
        if removed_facts(facts, set(range(cut, len(sents)))) >= allowed + 2:
            pos = text.find(sents[cut])
            if pos > 0:
                v["truncate"] = {"text": text[:pos].strip(), "cut_at": cut}
            break
    e = gen_json(llm, CONTRA.format(numbered=numbered(sents)), CONTRA_SCHEMA)
    if e and 0 <= int(e.get("index", -1)) < len(sents) and e.get("new_sentence", "").strip():
        old, new = sents[int(e["index"])], e["new_sentence"].strip()
        if new != old:
            v["contradict"] = {
                "text": text.replace(old, new, 1),
                "old": old,
                "new": new,
                "kind": e.get("kind"),
            }
    e = gen_json(llm, INSERT.format(numbered=numbered(sents)), INSERT_SCHEMA)
    if e and 0 <= int(e.get("after", -1)) < len(sents) and e.get("sentence", "").strip():
        anchor = sents[int(e["after"])]
        v["insert"] = {
            "text": text.replace(anchor, anchor + " " + e["sentence"].strip(), 1),
            "new": e["sentence"],
        }
    e = gen_json(llm, PAD.format(text=text), PAD_SCHEMA)
    if e and len(e.get("sentences", [])) >= 3:
        pads = [s.strip() for s in e["sentences"] if s.strip()][:5]
        out, step = text, max(1, len(sents) // (len(pads) + 1))
        for j, p in enumerate(pads):
            anchor = sents[min(len(sents) - 1, (j + 1) * step - 1)]
            out = out.replace(anchor, anchor + " " + p, 1)
        v["pad"] = {"text": out, "new": pads}
    for d in v.values():
        d["text"] = normalize(d["text"])[0]  # G0, as in production
    return v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="pilot-1")
    ap.add_argument("--per-genre", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="first N sampled items only (smoke run)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=str(var_dir() / "calibration" / "g1v2.json"))
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
    phr_path = pdir / "phrasly.enc"
    phrasly = json.loads(decrypt_at_rest(key, phr_path.read_bytes())) if phr_path.exists() else {}

    rng = random.Random(20261109)
    by_genre = defaultdict(list)
    for i, it in enumerate(items):
        by_genre[it["slot"]["genre"]].append(i)
    chosen = sorted(
        i for g in sorted(by_genre) for i in rng.sample(by_genre[g], min(args.per_genre, len(by_genre[g])))
    )
    chosen = chosen[: args.limit] if args.limit else chosen
    print(f"items: {len(chosen)}", flush=True)

    # 1. facts + variants
    def prep(i):
        text = items[i]["text"]
        facts, rep = prepare_facts(llm, J(EXTRACTOR), tuple(J(m) for m in VALIDATORS), text)
        if not facts:
            return i, None
        return i, {
            "facts": facts,
            "facts_report": {"extracted": rep.extracted, "kept": rep.kept, "dropped": rep.dropped,
                             "uncovered_sentences": rep.uncovered_sentences, "n_sentences": len(split_sentences(text))},
            "variants": build_variants(llm, text, facts, phrasly.get(str(i)), random.Random(i)),
        }  # fmt: skip

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(8, args.workers)) as ex:
        prepared = {i: d for i, d in ex.map(prep, chosen) if d}
    print(f"facts+variants: {len(prepared)} items, {time.time() - t0:.0f}s", flush=True)

    tasks = []  # (item, variant, plan, output)
    for i, d in prepared.items():
        for name, var in d["variants"].items():
            tasks.append((i, name, plan_g1(items[i]["text"], var["text"], d["facts"]), var["text"]))

    # 2. every candidate labels every open id (recall + precision)
    store: dict[tuple, dict | None] = {}  # (task_idx, model, kind) -> {id: (label, note)} | None
    cost: dict[tuple, float] = {}

    def label(job):
        t, model, kind, ids = job
        i, _, plan, out = tasks[t]
        j = J(model)
        if kind == "recall":
            v, c = recall_verdicts(llm, j, plan.facts, plan.src_sents, out, ids)
        else:
            v, c = precision_verdicts(llm, j, items[i]["text"], plan.out_sents, ids)
        return job, v, c

    def run(jobs):
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for (t, model, kind, _ids), v, c in ex.map(label, jobs):
                store[(t, model, kind)] = None if v is None else {**(store.get((t, model, kind)) or {}), **v}
                cost[(t, model, kind)] = cost.get((t, model, kind), 0.0) + c

    jobs = []
    for t, (_, _, plan, _) in enumerate(tasks):
        for m in CANDIDATES:
            if plan.open_facts:
                jobs.append((t, m, "recall", plan.open_facts))
            if plan.open_sents:
                jobs.append((t, m, "precision", plan.open_sents))
    t0 = time.time()
    run(jobs)
    print(f"candidates: {len(jobs)} calls, {time.time() - t0:.0f}s", flush=True)

    # 3. arbiters on the union of cross-vendor disputes
    pairs = [(a, b) for a, b in itertools.combinations(CANDIDATES, 2) if vendor(a) != vendor(b)]
    arb_jobs = []
    for t, (_, _, plan, _) in enumerate(tasks):
        for kind, ids in (("recall", plan.open_facts), ("precision", plan.open_sents)):
            disputed = set()
            for a, b in pairs:
                va, vb = store.get((t, a, kind)), store.get((t, b, kind))
                if va is None or vb is None:
                    continue
                for k in ids:
                    x, y = va.get(k), vb.get(k)
                    if not x or not y or _CLASS[x[0]] != _CLASS[y[0]]:
                        disputed.add(k)
            if disputed:
                arb_jobs += [(t, m, kind, sorted(disputed)) for m in ARBITERS]
    t0 = time.time()
    run(arb_jobs)
    print(f"arbiters: {len(arb_jobs)} calls, {time.time() - t0:.0f}s", flush=True)

    # 4. offline evaluation of every config with the production aggregation
    def evaluate(a, b, arb):
        rows = []
        for t, (i, name, plan, _) in enumerate(tasks):

            def ask(kind, t=t):
                def f(judge, ids):
                    v = store.get((t, judge.model, kind))
                    return (None, 0.0) if v is None else ({k: v[k] for k in ids if k in v}, 0.0)

                return f

            rv, pv, _, errs = run_two_family(
                plan, ask("recall"), ask("precision"), (Judge(a), Judge(b)), Judge(arb) if arb else None
            )
            if rv is None or pv is None:
                rows.append({"item": i, "variant": name, "status": "error", "reasons": errs})
                continue
            r = decide_g1(plan, rv, pv)
            c = sum(cost.get((t, m, k), 0.0) for m in (a, b) for k in ("recall", "precision"))
            if arb:
                n_arb = r.counts["arbitrated"]
                union = sum(len(set(store.get((t, arb, k)) or {})) for k in ("recall", "precision"))
                c += sum(cost.get((t, arb, k), 0.0) for k in ("recall", "precision")) * (
                    n_arb / union if union else 0
                )
            rows.append({"item": i, "variant": name, "status": r.status, "reasons": r.reasons, "counts": r.counts,
                         "facts": r.facts, "sentences": r.sentences, "cost": c})  # fmt: skip
        return rows

    def summarise(rows):
        by = defaultdict(list)
        for r in rows:
            by[r["variant"]].append(r)
        rate = {}
        for name, rs in by.items():
            ok = [r for r in rs if r["status"] != "error"]
            want = "fail" if name in MUST_FAIL else "pass"
            rate[name] = {"n": len(rs), "errors": len(rs) - len(ok),
                          "correct" if name not in SOFT else "pass": (sum(r["status"] == want for r in ok) / len(ok)) if ok else None}  # fmt: skip
        real = [r for r in rows if r["variant"] in SOFT and r["status"] != "error"]
        avg_cost = sum(r["cost"] for r in real) / len(real) if real else 0.0
        hard = [
            rate[v]["correct"] for v in MUST_PASS + MUST_FAIL if v in rate and rate[v]["correct"] is not None
        ]
        return {"rates": rate, "cost_per_output": avg_cost, "worst": min(hard) if hard else None,
                "mean": sum(hard) / len(hard) if hard else None}  # fmt: skip

    configs = {}
    for a, b in pairs:
        for arb in [*ARBITERS, None]:
            rows = evaluate(a, b, arb)
            configs[f"{a} + {b} | {arb or 'no arbiter'}"] = {"summary": summarise(rows), "rows": rows}
    ranked = sorted(
        configs.items(),
        key=lambda kv: (
            -(kv[1]["summary"]["worst"] or 0),
            -(kv[1]["summary"]["mean"] or 0),
            kv[1]["summary"]["cost_per_output"],
        ),
    )

    hdr = ["config", *MUST_PASS, *MUST_FAIL, *SOFT, "$/output"]
    print(" | ".join(hdr))
    for name, c in ranked:
        rt = c["summary"]["rates"]
        cells = [name]
        for v in MUST_PASS + MUST_FAIL + SOFT:
            x = rt.get(v, {})
            val = x.get("correct", x.get("pass"))
            cells.append(
                "-" if val is None else f"{val:.2f}" + (f"(e{x['errors']})" if x.get("errors") else "")
            )
        cells.append(f"{c['summary']['cost_per_output']:.4f}")
        print(" | ".join(cells))

    out = {
        "items": {
            str(i): {"genre": items[i]["slot"]["genre"], "text": items[i]["text"], "facts": [f.to_dict() for f in d["facts"]],
                     "facts_report": d["facts_report"], "variants": d["variants"]}
            for i, d in prepared.items()
        },
        "candidates": CANDIDATES, "arbiters": ARBITERS, "extractor": EXTRACTOR, "validators": VALIDATORS,
        "raw": [{"task": t, "item": tasks[t][0], "variant": tasks[t][1], "model": m, "kind": k, "verdicts": v,
                 "cost": cost.get((t, m, k))} for (t, m, k), v in store.items()],
        "configs": {k: v for k, v in ranked},
        "cache": {"hits": llm.hits, "misses": llm.misses},
        "openrouter_spent_this_run": led.spent("openrouter") - spent0,
    }  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(out, indent=1, default=lambda o: list(o) if isinstance(o, tuple) else str(o))
    )
    fr = [d["facts_report"] for d in prepared.values()]
    print(json.dumps({"facts_extracted": sum(x["extracted"] for x in fr), "facts_kept": sum(x["kept"] for x in fr),
                      "uncovered_sentences": sum(len(x["uncovered_sentences"]) for x in fr),
                      "sentences": sum(x["n_sentences"] for x in fr),
                      "cache": out["cache"], "spent": round(out["openrouter_spent_this_run"], 3),
                      "openrouter": led.summary()["openrouter"]}, indent=1))  # fmt: skip
    return 0 if math.isfinite(out["openrouter_spent_this_run"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
