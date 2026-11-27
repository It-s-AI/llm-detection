"""Human text pool: guaranteed-human (pre-2022) reference texts plus public metadata.

Modules:
    sources   -- declarative registry of corpora (and the ones we rejected, with reasons)
    clean     -- normalization, quality/PII filters, excerpting, deduplication
    metadata  -- text ids, length bins, register heuristic, genre proportions
    cluster   -- embedder protocol, per-genre clustering, generic cluster labels + checks
    store     -- parquet storage and deterministic slot/reference sampling (D-51, D-54)
    build     -- end-to-end dev build used by ``scripts/build_pool_dev.py``
    viewer    -- HTML sample viewer for eyeballing the pool

Heavy third-party imports (datasets, scikit-learn, langid, datasketch) are done lazily so that
``store`` can be used without the ``pool`` extra's heavier pieces.
"""
