#!/usr/bin/env python
"""Build the human pool (dev scale by default: a few thousand texts per genre) and an HTML sample viewer.

    export UV_CACHE_DIR=/workspace/sn32_owner/.uv-cache HF_HOME=/workspace/sn32_owner/.hf-cache
    export HUMEVAL_ENV_FILE=/workspace/sn32_owner/.env      # provides HF_API_KEY (never printed)
    .venv/bin/python scripts/build_pool_dev.py   # -> var/pool/{pool,metadata}.parquet, samples.html

Full scale (D-51 cluster minimum 300; registry strides are tuned for ~3000/genre, so divide them by the
scale factor -- otherwise strided sources run out of rows):

    .venv/bin/python scripts/build_pool_dev.py --out var/pool-full-v2 --per-genre 30000 \\
        --min-cluster-size 300 --stride-divisor 10 --workers 48

Re-cluster / re-render without the network: add ``--reuse-candidates``. Candidates are cached in a wide
window (``--cache-min-words``/``--cache-max-words``, default 100-500); the pool window (``--min-words``/
``--max-words``, default 150-350) is applied at assembly, so changing it inside the cache window needs only
``--reuse-candidates``. ``--genre-target academic=26000`` overrides the per-genre target of one genre.
Free: Hugging Face streaming + local TF-IDF/SVD embeddings; no paid API is called.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from humanizer_eval.paths import var_dir

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: <var>/pool)")
    ap.add_argument("--per-genre", type=int, default=3000, help="target texts per genre")
    ap.add_argument("--min-cluster-size", type=int, default=50, help="dev minimum; 300 at full scale (D-51)")
    ap.add_argument("--cluster-size-factor", type=int, default=3, help="k = n // (min_size * factor)")
    ap.add_argument(
        "--stride-divisor", type=float, default=1.0, help="divide registry strides (use per_genre / 3000)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default="", help="comma-separated source ids (default: all enabled)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--samples-per-genre", type=int, default=20)
    ap.add_argument("--reuse-candidates", action="store_true", help="reuse cached per-source candidates")
    ap.add_argument("--min-words", type=int, default=150, help="pool length window, lower bound (D-55)")
    ap.add_argument("--max-words", type=int, default=350, help="pool length window, upper bound (D-55)")
    ap.add_argument("--cache-min-words", type=int, default=100, help="candidate cache window, lower bound")
    ap.add_argument("--cache-max-words", type=int, default=500, help="candidate cache window, upper bound")
    ap.add_argument(
        "--genre-target", action="append", default=[], metavar="GENRE=N", help="per-genre target override"
    )
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)  # progress lines reach a redirected log at once

    out = args.out or (var_dir() / "pool")
    # keep the HF cache off the small root disk unless the caller chose a location
    os.environ.setdefault("HF_HOME", str((out.parent / "hf-cache").resolve()))

    from humanizer_eval.pool.build import BuildConfig, build_pool, format_stats

    cfg = BuildConfig(
        out_dir=out,
        per_genre=args.per_genre,
        min_cluster_size=args.min_cluster_size,
        cluster_size_factor=args.cluster_size_factor,
        stride_divisor=args.stride_divisor,
        seed=args.seed,
        source_ids=tuple(s for s in args.sources.split(",") if s) or None,
        workers=args.workers,
        reuse_candidates=args.reuse_candidates,
        samples_per_genre=args.samples_per_genre,
        min_words=args.min_words,
        max_words=args.max_words,
        cache_min_words=args.cache_min_words,
        cache_max_words=args.cache_max_words,
        per_genre_overrides={g: int(n) for g, n in (kv.split("=", 1) for kv in args.genre_target)},
    )
    stats = build_pool(cfg)
    print()
    print(format_stats(stats))
    print(f"\nwrote {out}/pool.parquet, metadata.parquet, clusters.json, stats.json, samples.html")
    print(f"elapsed {stats['elapsed_s']}s")
    sys.stdout.flush()
    (out / "build_summary.json").write_text(
        json.dumps({k: stats[k] for k in ("total", "elapsed_s", "cluster_check")}, indent=2)
    )
    return 0 if stats["cluster_check"]["ok"] else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # datasets' streaming threads can crash the interpreter during finalization (PyGILState_Release);
    # all outputs are written and flushed at this point, so skip finalization.
    os._exit(code)
