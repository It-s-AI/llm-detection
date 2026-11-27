"""G2/G3 calibration of quality levels (D-22): do the judges + arbiter give the expected 1 / 0.5 / 0?

Reuses the 24 inputs (and their light_edit / paraphrase variants) of the G1 calibration and adds variants
with a known level for one gate:

  G2 (style)        0.5: person_tense                         0: language, genre
                    1 (not checked, owner decision): register, format
  G3 (naturalness)  0.5: typos, odd_words, splice (comma splices)   0: broken, shuffle, artifacts
                    1 (meaning changes are G1's job): contradict, insert (from the G1 calibration)
  both 1 (clean)    : light_edit, paraphrase (soft)

Unchanged / typography-only outputs never reach the judges (GateRunner shortcut), so they are not run.
Each variant goes through the production consensus (two judges, arbiter on disputes); the raw judge replies
are kept so the old rules ("any judge flags", "both judges, no arbiter") can be compared offline.
Output: var/calibration/g23.json + printed tables. Replies are cached (re-running is free).
Usage: python scripts/calibrate_g23.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.gates.judges import STYLE_KEYS, GateConfig, GateRunner
from humanizer_eval.gates.meaning import split_sentences
from humanizer_eval.gates.normalize import normalize
from humanizer_eval.llm.cache import CachedLLM
from humanizer_eval.llm.openrouter import HIDDEN, ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import load_or_create_key

# G3 v2 flags (this script calibrated G3 v2, report 04c; G3 v3 lists defects with severity instead)
DEFECT_KEYS = (
    "introduced_errors",
    "broken_sentences",
    "run_ons",
    "odd_word_choice",
    "incoherent",
    "artifacts",
)
G3_SEVERE_V2 = ("broken_sentences", "incoherent", "artifacts")
G3_MINOR_V2 = ("introduced_errors", "run_ons", "odd_word_choice")

VARIANT_MODEL = "x-ai/grok-4.7"
EXPECT = {  # variant -> (expected G2 level, expected G3 level); None = not a test of that gate
    "light_edit": (1.0, 1.0),
    "paraphrase": (1.0, 1.0),
    "register": (1.0, None),
    "person_tense": (0.5, None),
    "format": (1.0, None),
    "language": (0.0, None),
    "genre": (0.0, None),
    "typos": (None, 0.5),
    "odd_words": (None, 0.5),
    "splice": (None, 0.5),
    "contradict": (None, 1.0),
    "insert": (None, 1.0),
    "broken": (None, 0.0),
    "shuffle": (None, 0.0),
    "artifacts": (None, 0.0),
}
GENRE_SWAP = {
    "essay": "a news report",
    "article_blog": "a short poem",
    "academic": "a short story",
    "news": "a personal diary entry",
    "review": "a news report",
    "forum_qa": "a formal academic abstract",
    "creative": "a news report",
    "email_letter": "a news report",
}
PROMPTS = {
    "register": (
        "Rewrite the text in the opposite register. If it is formal, make it clearly casual and "
        "conversational (contractions, everyday words, a chatty tone); if it is informal, make it clearly "
        "formal. Keep every fact, the paragraphs and roughly the length. Output only the rewrite."
    ),
    "person_tense": (
        "Rewrite the text changing its grammatical person (first person <-> third person) or, if that does "
        "not apply, its tense (past <-> present). Keep every fact, the register and the paragraphs. Output "
        "only the rewrite."
    ),
    "language": "Translate the text into Spanish. Output only the translation.",
    "odd_words": (
        "Copy the text exactly, but replace about ten ordinary words with unusual thesaurus-style synonyms "
        "that sound unnatural in context (for example 'use' -> 'harness', 'big' -> 'gargantuan', 'said' -> "
        "'opined'). Change nothing else. Output only the text."
    ),
}


def gen(llm, prompt: str, text: str) -> str:
    r = llm.chat(
        VARIANT_MODEL,
        [{"role": "user", "content": f"{prompt}\n\n<text>\n{text}\n</text>"}],
        max_tokens=3000,
        temperature=0.3,
        policy=HIDDEN,
        purpose="calib23:variant",
    )
    return r.text.strip()


def typos(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    cand = [i for i, w in enumerate(words) if re.fullmatch(r"[A-Za-z]{5,}[.,;:!?]?", w)]
    for i in rng.sample(cand, min(len(cand), max(4, len(words) // 25))):
        w = words[i]
        j = rng.randrange(1, len(w.rstrip(".,;:!?")) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def broken(text: str, rng: random.Random) -> str:
    sents = split_sentences(text)
    for k in rng.sample(range(len(sents)), min(3, len(sents))):
        ws = sents[k].split()
        if len(ws) > 6:
            text = text.replace(sents[k], " ".join(ws[: len(ws) // 2]), 1)
    return text


def splice(text: str, rng: random.Random) -> str:
    """Join three pairs of neighbouring sentences with a comma (comma splices)."""
    sents = split_sentences(text)
    pairs = [k for k in range(len(sents) - 1) if sents[k].endswith(".") and sents[k + 1][:1].isupper()]
    for k in sorted(rng.sample(pairs, min(3, len(pairs))), reverse=True):
        a, b = sents[k], sents[k + 1]
        b2 = b if b.split(" ", 1)[0] in ("I", "I'm", "I've") else b[0].lower() + b[1:]
        joined = text.replace(f"{a} {b}", f"{a[:-1]}, {b2}", 1)
        text = joined if joined != text else text.replace(f"{a}\n{b}", f"{a[:-1]}, {b2}", 1)
    return text


def shuffle(text: str, rng: random.Random) -> str:
    sents = split_sentences(text)
    order = list(range(len(sents)))
    while len(order) > 2 and order == sorted(order):
        rng.shuffle(order)
    return " ".join(sents[i] for i in order)


def artifacts(text: str) -> str:
    paras = text.split("\n\n")
    paras.insert(1, "As an AI language model, I cannot verify these claims. [citation needed] <br>")
    return "\n\n".join(paras) + "\n\n---\n*Note: This text was rewritten to sound more natural.*"


def fmt(text: str) -> str:
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paras) > 1:
        return " ".join(p.replace("\n", " ") for p in paras)
    return "\n".join(f"- {s}" for s in split_sentences(text))


def g2_level(changed, cfg: GateConfig) -> float:
    return 0.0 if set(changed) & set(cfg.g2_severe) else 0.5 if changed else 1.0


def g3_level(flags, worse, cfg: GateConfig) -> float:
    f = set(flags)
    return 0.0 if f & set(G3_SEVERE_V2) else 0.5 if (f & set(G3_MINOR_V2)) or worse else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(var_dir() / "calibration" / "g23.json"))
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
    cfg_gates = json.loads(Path("configs/gates.json").read_text())
    runner = GateRunner.from_config(llm, cfg_gates)
    cfg = runner.config
    cal = json.loads((var_dir() / "calibration" / "g1v2.json").read_text())["items"]
    ids = sorted(cal, key=int)[: args.limit]

    def variants(i: str):
        it = cal[i]
        src, rng = it["text"], random.Random(int(i))
        v = {
            "light_edit": it["variants"]["light_edit"]["text"],
            "paraphrase": it["variants"]["paraphrase"]["text"],
            "contradict": it["variants"]["contradict"]["text"],
            "insert": it["variants"]["insert"]["text"],
        }
        for name, prompt in PROMPTS.items():
            v[name] = gen(llm, prompt, src)
        v["genre"] = gen(llm, f"Rewrite the same content as {GENRE_SWAP.get(it['genre'], 'a news report')}. "
                         "Keep the facts. Output only the rewrite.", src)  # fmt: skip
        v["format"] = fmt(src)
        v["typos"] = typos(src, rng)
        v["broken"] = broken(src, rng)
        v["splice"] = splice(src, random.Random(int(i) + 1))
        v["shuffle"] = shuffle(src, rng)
        v["artifacts"] = artifacts(src)
        return i, {k: normalize(x)[0] for k, x in v.items()}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        built = dict(ex.map(variants, ids))
    jobs = [(i, name) for i in ids for name in EXPECT]

    def judge(job):
        i, name = job
        src, out = cal[i]["text"], built[i][name]
        st, _c1, e1 = runner._consensus_style(src, out)
        nat, _c2, e2 = runner._consensus_natural(src, out)
        return job, st, nat, e1 or e2

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(judge, jobs))

    rows = []
    for (i, name), st, nat, err in results:
        row = {"item": i, "genre": cal[i]["genre"], "variant": name, "error": err}
        if not err:
            sj = st["judges"]
            any_changed = [k for k in STYLE_KEYS if any(o.get(k) is not True for o in sj)]
            both_changed = [k for k in STYLE_KEYS if all(o.get(k) is not True for o in sj)]
            nj = nat["judges"]
            fl = [{k for k in DEFECT_KEYS if o["order1"].get(k)} for o in nj]
            wv = [o["order1"].get("second_text_readability") == "worse"
                  and o["order2"].get("second_text_readability") == "better" for o in nj]  # fmt: skip
            row["G2"] = {
                "arbiter": g2_level(st["changed"], cfg),
                "any": g2_level(any_changed, cfg),
                "both": g2_level(both_changed, cfg),
                "changed": st["changed"],
            }
            row["G3"] = {
                "arbiter": g3_level(nat["flags"], nat["worse"], cfg),
                "any": g3_level(fl[0] | fl[1], any(wv), cfg),
                "both": g3_level(fl[0] & fl[1], all(wv), cfg),
                "flags": nat["flags"],
                "worse": nat["worse"],
            }
        rows.append(row)

    summary = {}
    for gate, gi in (("G2", 0), ("G3", 1)):
        summary[gate] = {}
        for rule in ("arbiter", "both", "any"):
            per = defaultdict(Counter)
            for r in rows:
                exp = EXPECT[r["variant"]][gi]
                if exp is None or r["error"]:
                    continue
                per[r["variant"]][r[gate][rule]] += 1
            table = {}
            for v, cnt in per.items():
                exp = EXPECT[v][gi]
                n = sum(cnt.values())
                table[v] = {
                    "expected": exp,
                    "n": n,
                    "exact": cnt[exp] / n,
                    "flagged": sum(c for lv, c in cnt.items() if lv < 1.0) / n,
                    "dist": {str(k): c for k, c in sorted(cnt.items())},
                }
            summary[gate][rule] = table
    errors = sum(1 for r in rows if r["error"])
    out = {"summary": summary, "rows": rows, "variants": built, "errors": errors,
           "spent": led.spent("openrouter") - spent0, "cache": {"hits": llm.hits, "misses": llm.misses}}  # fmt: skip
    Path(args.out).write_text(json.dumps(out, indent=1))
    for gate in ("G2", "G3"):
        print(f"== {gate}  (exact level / any flag, per rule)")
        for v in EXPECT:
            cells = []
            for rule in ("arbiter", "both", "any"):
                t = summary[gate][rule].get(v)
                if t:
                    cells.append(f"{rule}: {t['exact']:.2f}/{t['flagged']:.2f} {t['dist']}")
            if cells:
                print(f"  {v:13s} exp={summary[gate]['arbiter'][v]['expected']}  " + " | ".join(cells))
    print(json.dumps({"errors": errors, "spent": round(out["spent"], 3), "cache": out["cache"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
