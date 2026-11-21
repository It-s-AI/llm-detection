"""D-51 leak test: are evaluation inputs closer to the human pool than production-like AI texts?

Production-like texts are generated for the same genres/lengths but with topics proposed WITHOUT the
pool's cluster labels. For every text we take its maximum cosine similarity (TF-IDF fitted on the pool)
to any pool text of the same genre. If the pool leaks into evaluation inputs, eval similarities are
systematically higher: AUROC(eval vs production-like) noticeably above 0.5.
Usage: python scripts/leak_test.py --pool <pool_dir> [--n-prod 40]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, load_or_create_key
from humanizer_eval.period.generate import GenConfig, generate_input, propose_topics
from humanizer_eval.pool.store import PoolStore
from humanizer_eval.scoring.metric import auroc


def max_sim(vec, texts, pool_by_genre, genres):
    out = []
    for t, g in zip(texts, genres, strict=True):
        P = pool_by_genre[g]
        v = vec.transform([t])
        out.append(float((P @ v.T).max()))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--period", default="pilot-1")
    ap.add_argument("--n-prod", type=int, default=40)
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    llm = OpenRouter(
        s.openrouter_api_key,
        led,
        pricing=ModelPricing(var_dir() / "openrouter_models.json"),
        log_path=var_dir() / "llm_calls.jsonl",
    )
    data = json.loads(
        decrypt_at_rest(
            load_or_create_key(var_dir() / "keys" / "period.key"),
            (var_dir() / "periods" / args.period / "dataset.enc").read_bytes(),
        )
    )
    items = data["items"]
    rows = PoolStore.load(args.pool).rows
    vec = TfidfVectorizer(min_df=2, max_features=200_000, sublinear_tf=True, stop_words="english")
    vec.fit([r["text"] for r in rows])
    pool_by_genre = {}
    for g in {r["genre"] for r in rows}:
        pool_by_genre[g] = vec.transform([r["text"] for r in rows if r["genre"] == g])
    # production-like: same genre / length-bin mix, topics WITHOUT cluster labels
    cfg = GenConfig.from_dict(
        {
            **json.loads(Path("configs/period.json").read_text()),
            "generators": [{"model": "openai/gpt-5.4-mini", "weight": 1}],
        }
    )
    rng = random.Random(7)
    sample = rng.sample(items, min(args.n_prod, len(items)))
    prod, prod_genres = [], []
    genre_counts = Counter(it["slot"]["genre"] for it in sample)
    topics = {}
    for g, k in genre_counts.items():
        topics[g], _ = propose_topics(
            llm,
            cfg.topic_model,
            g,
            "any everyday subject people write about",
            k + 2,
            seed=rng.randint(0, 10**9),
        )
    used = Counter()
    for it in sample:
        g = it["slot"]["genre"]
        t = topics[g][used[g] % max(1, len(topics[g]))] if topics[g] else {"topic": "daily life"}
        used[g] += 1
        r = generate_input(llm, random.Random(rng.random()), cfg, it["slot"], t)
        if r.ok:
            prod.append(r.text)
            prod_genres.append(g)
    eval_sims = max_sim(
        vec, [it["text"] for it in items], pool_by_genre, [it["slot"]["genre"] for it in items]
    )
    prod_sims = max_sim(vec, prod, pool_by_genre, prod_genres)
    a = float(auroc(eval_sims, prod_sims))
    out = {
        "n_eval": len(eval_sims),
        "n_prod": len(prod_sims),
        "eval_maxsim_mean": float(eval_sims.mean()),
        "eval_maxsim_median": float(np.median(eval_sims)),
        "prod_maxsim_mean": float(prod_sims.mean()),
        "prod_maxsim_median": float(np.median(prod_sims)),
        "auroc_eval_closer_than_prod": a,
        "verdict": "no evidence of leak" if a < 0.65 else "eval inputs are closer to the pool -> investigate",
        "openrouter": led.summary().get("openrouter"),
    }
    print(json.dumps(out, indent=1))
    Path("docs/reports/data/03-leak-test.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
