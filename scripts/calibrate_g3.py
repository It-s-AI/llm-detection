"""G3 v3 calibration (D-43): do the judges + arbiter give the expected level 1 / 0.75 / 0.5 / 0?

Uses the known-answer variants of the G2/G3 calibration (report 04c, var/calibration/g23.json: 24 inputs,
variants built from each input) plus two deterministic ones for the new 0.75 level. Expected G3 level:

  1           light_edit, paraphrase, register, format, person_tense, contradict, insert (no new form defect;
              meaning changes are G1's job)
  0.75        typos_few (two swapped letters), splice_one (one comma splice)
  0.5         typos (max(4, words/25) swapped letters), odd_words (~10 thesaurus words); splice (three comma
              splices) may be 0.75 or 0.5
  0           broken (three sentences cut in half), shuffle (sentence order shuffled), artifacts (meta-text)

Each variant goes through the production consensus (configs/gates.json). The table shows, per variant, the
share at the expected level, too strict (below it) and too lenient (above it), and how often the two
judges agreed. Replies are cached, so re-running is free; a prompt change re-asks the judges.
Usage: python scripts/calibrate_g3.py [--limit N]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.gates.judges import GateRunner
from humanizer_eval.gates.normalize import normalize
from humanizer_eval.llm.cache import CachedLLM
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import load_or_create_key

EXPECT = {  # variant -> acceptable G3 levels
    "light_edit": {1.0},
    "paraphrase": {1.0},
    "register": {1.0},
    "format": {1.0},
    "person_tense": {1.0},
    "contradict": {1.0},
    "insert": {1.0},
    "typos_few": {0.75},
    "splice_one": {0.75},
    "typos": {0.5},
    "odd_words": {0.5},
    "splice": {0.75, 0.5},
    "broken": {0.0},
    "shuffle": {0.0},
    "artifacts": {0.0},
}


def _g23():
    spec = importlib.util.spec_from_file_location(
        "calibrate_g23", Path(__file__).with_name("calibrate_g23.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def typos_n(text: str, rng: random.Random, n: int) -> str:
    import re

    words = text.split(" ")
    cand = [i for i, w in enumerate(words) if re.fullmatch(r"[A-Za-z]{5,}[.,;:!?]?", w)]
    for i in rng.sample(cand, min(len(cand), n)):
        w = words[i]
        j = rng.randrange(1, len(w.rstrip(".,;:!?")) - 2)
        words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
    return " ".join(words)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(var_dir() / "calibration" / "g3v3.json"))
    args = ap.parse_args()
    g23 = _g23()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    spent0 = led.spent("openrouter")
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    base = OpenRouter(s.openrouter_api_key, led, pricing=ModelPricing(var_dir() / "openrouter_models.json"),
                      log_path=var_dir() / "llm_calls.jsonl")  # fmt: skip
    llm = CachedLLM(base, var_dir() / "calibration" / "llm_cache.sqlite", key=key, cache_all=True)
    runner = GateRunner.from_config(llm, json.loads(Path("configs/gates.json").read_text()))
    cal = json.loads((var_dir() / "calibration" / "g1v2.json").read_text())["items"]
    saved = json.loads((var_dir() / "calibration" / "g23.json").read_text())["variants"]
    ids = sorted(saved, key=int)[: args.limit]

    built = {}
    for i in ids:
        src = cal[i]["text"]
        v = {k: saved[i][k] for k in EXPECT if k in saved[i]}
        v["typos_few"] = normalize(typos_n(src, random.Random(1000 + int(i)), 2))[0]
        v["splice_one"] = normalize(g23.splice_n(src, random.Random(2000 + int(i)), 1))[0] if hasattr(g23, "splice_n") \
            else normalize(_splice_one(g23, src, random.Random(2000 + int(i))))[0]  # fmt: skip
        built[i] = v
    jobs = [(i, name) for i in ids for name in EXPECT if name in built[i]]

    def judge(job):
        i, name = job
        src, out = cal[i]["text"], built[i][name]
        nat, _c, err = runner._consensus_natural(src, out)
        return job, nat, err

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(judge, jobs))

    rows = []
    for (i, name), nat, err in results:
        row = {"item": i, "genre": cal[i]["genre"], "variant": name, "error": err}
        if not err:
            row.update(level=nat["level"], judges=[v["level"] for v in nat["judges"]],
                       arbiter=None if not nat["arbiter"] else nat["arbiter"]["level"],
                       defects=[v["defects"] for v in nat["judges"]],
                       arbiter_defects=None if not nat["arbiter"] else nat["arbiter"]["defects"])  # fmt: skip
        rows.append(row)

    table = {}
    per = defaultdict(list)
    for r in rows:
        if not r["error"]:
            per[r["variant"]].append(r)
    for v in EXPECT:
        rs = per.get(v, [])
        if not rs:
            continue
        exp = EXPECT[v]
        n = len(rs)
        table[v] = {
            "expected": sorted(exp, reverse=True),
            "n": n,
            "ok": sum(r["level"] in exp for r in rs) / n,
            "too_strict": sum(r["level"] < min(exp) for r in rs) / n,
            "too_lenient": sum(r["level"] > max(exp) for r in rs) / n,
            "judges_agree": sum(r["judges"][0] == r["judges"][1] for r in rs) / n,
            "dist": {str(k): c for k, c in sorted(Counter(r["level"] for r in rs).items(), reverse=True)},
        }
    errors = sum(1 for r in rows if r["error"])
    out = {"table": table, "rows": rows, "variants": built, "errors": errors,
           "spent": led.spent("openrouter") - spent0, "cache": {"hits": llm.hits, "misses": llm.misses}}  # fmt: skip
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{'variant':12s} {'expected':10s} {'ok':>5s} {'strict':>7s} {'lenient':>8s} {'agree':>6s}  levels")
    for v, t in table.items():
        exp = "/".join(str(x) for x in t["expected"])
        print(f"{v:12s} {exp:10s} {t['ok']:5.2f} {t['too_strict']:7.2f} {t['too_lenient']:8.2f} "
              f"{t['judges_agree']:6.2f}  {t['dist']}")  # fmt: skip
    print(json.dumps({"errors": errors, "spent": round(out["spent"], 3), "cache": out["cache"]}))
    return 0


def _splice_one(g23, text: str, rng: random.Random) -> str:
    """One comma splice: g23.splice joins three pairs; join one pair the same way."""
    sents = g23.split_sentences(text)
    pairs = [k for k in range(len(sents) - 1) if sents[k].endswith(".") and sents[k + 1][:1].isupper()]
    if not pairs:
        return text
    k = rng.choice(pairs)
    a, b = sents[k], sents[k + 1]
    b2 = b if b.split(" ", 1)[0] in ("I", "I'm", "I've") else b[0].lower() + b[1:]
    joined = text.replace(f"{a} {b}", f"{a[:-1]}, {b2}", 1)
    return joined if joined != text else text.replace(f"{a}\n{b}", f"{a[:-1]}, {b2}", 1)


if __name__ == "__main__":
    raise SystemExit(main())
