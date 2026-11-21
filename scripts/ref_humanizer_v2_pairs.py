"""Training pairs for the reference humanizer v2 (inverse paraphrase): AI rewrite -> original human text.

Human texts come from the pool in the production genre mix (configs/period.json), never from a pilot's
reference set H. Each one is rewritten "as an AI assistant would write it" by the period's own generator
mix (same models, weights and per-model settings as the period inputs), so the model learns to map the
inputs it will see back to human writing. Pairs whose rewrite is far off in length are dropped.

Output: var/ref_v2/pairs.jsonl ({"id", "genre", "ai", "human", "generator"}), resumable.
Usage: python scripts/ref_humanizer_v2_pairs.py --pool <pool_dir> --n 2400 [--max-usd 8]
"""

from __future__ import annotations

import argparse
import json
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.llm.openrouter import HIDDEN, ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, load_or_create_key
from humanizer_eval.period.generate import GenConfig, clean_generation, generator_settings, pick_generator
from humanizer_eval.pool.store import PoolStore

PROMPT = (
    "Below is a {genre} written by a person. Write the version an AI assistant would produce if it had been "
    "asked to write this text itself: the same content (every claim, name, number, example and reason, in "
    "the same order) but in your own fluent wording and sentence structure. Add no new information and "
    "drop nothing. About {words} words. Output only the text.\n\n<text>\n{text}\n</text>"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--max-usd", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261111)
    ap.add_argument("--exclude-period", action="append", default=["pilot-1"])
    ap.add_argument("--out", default=str(var_dir() / "ref_v2" / "pairs.jsonl"))
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    spent0 = led.spent("openrouter")
    llm = OpenRouter(s.openrouter_api_key, led, pricing=ModelPricing(var_dir() / "openrouter_models.json"),
                     log_path=var_dir() / "llm_calls.jsonl")  # fmt: skip
    cfg = GenConfig.from_dict(json.loads(Path("configs/period.json").read_text()))
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    hidden: set[str] = set()
    for p in args.exclude_period:
        f = var_dir() / "periods" / p / "dataset.enc"
        if f.exists():
            hidden |= {h["id"] for h in json.loads(decrypt_at_rest(key, f.read_bytes()))["human_ref"]}

    rows = [r for r in PoolStore.load(args.pool).rows if r["text_id"] not in hidden]
    by_genre = defaultdict(list)
    for r in rows:
        by_genre[r["genre"]].append(r)
    props = {g: w for g, w in cfg.genre_proportions.items() if by_genre.get(g)}
    tot = sum(props.values())
    want = {g: max(30, round(args.n * w / tot)) for g, w in props.items()}
    rng = random.Random(args.seed)
    picked = [r for g, k in want.items() for r in rng.sample(by_genre[g], min(k, len(by_genre[g])))]
    rng.shuffle(picked)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["id"] for x in out.read_text().splitlines()} if out.exists() else set()
    lock = threading.Lock()
    stats = Counter()

    def one(r: dict) -> None:
        if r["text_id"] in done or led.spent("openrouter") - spent0 > args.max_usd:
            return
        prng = random.Random(f"{args.seed}:{r['text_id']}")
        words = len(r["text"].split())
        model = pick_generator(prng, cfg)
        max_tokens, reasoning = generator_settings(cfg, model)
        prompt = PROMPT.format(genre=r["genre"].replace("_", " "), words=words, text=r["text"])
        try:
            res = llm.chat(model, [{"role": "user", "content": prompt}], max_tokens=max_tokens,
                           temperature=prng.choice([0.7, 1.0]), policy=HIDDEN, purpose="ref-v2:pairs",
                           reasoning=reasoning)  # fmt: skip
        except Exception as e:  # one provider failing must not stop the batch
            with lock:
                stats[f"error:{type(e).__name__}"] += 1
            return
        ai = clean_generation(res.text)
        n = len(ai.split())
        with lock:
            if not 0.7 * words <= n <= 1.4 * words:
                stats["length_off"] += 1
                return
            stats["ok"] += 1
            stats[f"gen:{model}"] += 1
            with out.open("a") as f:
                f.write(json.dumps({"id": r["text_id"], "genre": r["genre"], "ai": ai, "human": r["text"],
                                    "generator": res.model}) + "\n")  # fmt: skip

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(one, picked))
    n_pairs = len(out.read_text().splitlines()) if out.exists() else 0
    summary = {"pairs": n_pairs, "wanted": want, "stats": dict(stats),
               "spent": round(led.spent("openrouter") - spent0, 3)}  # fmt: skip
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
