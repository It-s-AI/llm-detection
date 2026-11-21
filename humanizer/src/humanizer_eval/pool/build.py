"""End-to-end pool build: stream sources -> clean/excerpt -> dedupe -> metadata -> clusters -> store.

One worker process per source streams rows from the Hugging Face Hub (``datasets`` streaming mode,
nothing is downloaded in full) until it has ``quota`` accepted excerpts. Accepted candidates and the
per-source rejection counts are cached as JSONL so re-clustering does not need the network.

Two length windows: candidates are collected and cached in a wide window (``cache_min_words`` ..
``cache_max_words``, default 100-500; a longer document contributes its longest paragraph-aligned prefix
that fits), and the pool window (``min_words`` .. ``max_words``, default 150-350, D-55) is applied at
assembly by ``clean.fit_window``. Changing the pool window inside the cache window therefore only needs
``--reuse-candidates``, not a new pass over the Hub.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import platform
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from humanizer_eval.pool.clean import (
    DEFAULT_FILTERS,
    Deduplicator,
    FilterConfig,
    fit_window,
    normalize_text,
    process_document,
    word_count,
)
from humanizer_eval.pool.cluster import (
    capitalization_ratios,
    check_clusters,
    cluster_genre,
    proper_from_ratios,
)
from humanizer_eval.pool.metadata import (
    DEFAULT_MAX_WORDS,
    DEFAULT_MIN_WORDS,
    LENGTH_BIN_LABELS,
    REGISTERS,
    LengthBins,
    TextRecord,
    make_length_bins,
)
from humanizer_eval.pool.sources import CUTOFF, GENRES, SOURCES, Source, enabled_sources, get_source
from humanizer_eval.pool.store import write_pool


@dataclass
class BuildConfig:
    out_dir: Path
    per_genre: int = 3000
    oversample: float = 1.3  # collect extra to absorb cross-source dedupe
    min_cluster_size: int = 50
    cluster_size_factor: int = 3
    seed: int = 0
    max_rows_per_candidate: int = 60  # give up on a source after quota * this many rows
    # registry strides are tuned for the dev scale (~3000/genre); a larger build divides them so a
    # source is not exhausted early (stride 50 -> 5 at divisor 10), never below 1
    stride_divisor: float = 1.0
    source_ids: tuple[str, ...] | None = None
    workers: int = 16
    reuse_candidates: bool = False
    samples_per_genre: int = 20
    # pool length window (D-55), applied at assembly; must lie inside the cache window
    min_words: int = DEFAULT_MIN_WORDS
    max_words: int = DEFAULT_MAX_WORDS
    # candidates are cached in this wider window, so the pool window can change without re-streaming
    cache_min_words: int = 100
    cache_max_words: int = 500
    # per-genre target overrides (e.g. a genre whose sources cannot reach ``per_genre``)
    per_genre_overrides: dict[str, int] = field(default_factory=dict)
    # quality/PII/language/dedupe thresholds; its min/max/target words are replaced by the windows above
    filters: FilterConfig = field(default_factory=lambda: DEFAULT_FILTERS)

    def __post_init__(self) -> None:
        if not self.cache_min_words <= self.min_words < self.max_words <= self.cache_max_words:
            raise ValueError(
                f"pool window {self.min_words}-{self.max_words} must lie inside the cache window "
                f"{self.cache_min_words}-{self.cache_max_words}"
            )

    def pool_filters(self) -> FilterConfig:
        """Filters of the pool window: excerpt target N(target_mean, sd) clipped to the window."""
        return replace(self.filters, min_words=self.min_words, max_words=self.max_words)

    def cache_filters(self) -> FilterConfig:
        """Filters of the cache window: a document longer than the window yields its longest prefix of
        whole paragraphs (sentences inside an over-long paragraph) that fits -- the target is the
        window's maximum -- so ``fit_window`` can later cut it to any pool window inside."""
        return replace(
            self.filters,
            min_words=self.cache_min_words,
            max_words=self.cache_max_words,
            target_mean_words=float(self.cache_max_words),
            target_sd_words=0.0,
        )

    def length_bins(self) -> LengthBins:
        return make_length_bins(self.min_words, self.max_words)

    def genre_target(self, genre: str) -> int:
        return self.per_genre_overrides.get(genre, self.per_genre)


@dataclass
class Candidate:
    text: str
    source_id: str
    genre: str
    source_doc_id: str
    date: str
    date_evidence: str


@dataclass
class SourceReport:
    source_id: str
    genre: str
    quota: int
    rows_scanned: int = 0
    documents: int = 0
    candidates: int = 0
    usable: int = 0  # candidates of >= ``count_min_words`` words (they count towards the quota)
    kept: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
    exhausted: bool = False
    error: str = ""
    window: list[int] = field(default_factory=list)  # cache window [min_words, max_words] of the candidates


# --------------------------------------------------------------------------- streaming
def iter_rows(source: Source, token: str | None) -> Iterator[dict]:
    import datasets

    def load(files):
        if source.builder:  # raw files of a repo whose loading script datasets>=4 refuses to run
            names = list(files) if isinstance(files, tuple | list) else [files]
            return datasets.load_dataset(
                source.builder,
                data_files=[f"hf://datasets/{source.hf_id}/{f}" for f in names],
                split="train",
                streaming=True,
                token=token,
                **dict(source.builder_kwargs),
            )
        kw = {"data_files": list(files) if isinstance(files, tuple | list) else files} if files else {}
        return datasets.load_dataset(
            source.hf_id, source.config, split=source.split, streaming=True, token=token, **kw
        )

    if source.data_files and source.interleave_files:
        iters = [iter(load(f)) for f in source.data_files]
        while iters:
            alive = []
            for it in iters:
                try:
                    yield next(it)
                    alive.append(it)
                except StopIteration:
                    pass
            iters = alive
    else:
        yield from load(source.data_files)


def source_quotas(
    sources: list[Source], per_genre: int, overrides: dict[str, int] | None = None
) -> dict[str, int]:
    """Per-source target counts after dedupe: the genre target (``per_genre`` or its override) split by
    weight within the genre."""
    by_genre: dict[str, list[Source]] = defaultdict(list)
    for s in sources:
        by_genre[s.genre].append(s)
    out: dict[str, int] = {}
    for genre, ss in by_genre.items():
        target = (overrides or {}).get(genre, per_genre)
        total = sum(s.weight for s in ss)
        for s in ss:
            out[s.id] = math.ceil(target * s.weight / total)
    return out


def effective_stride(stride: int, divisor: float = 1.0) -> int:
    """Registry stride scaled for the build size: ``round(stride / divisor)``, at least 1."""
    return max(1, round(stride / divisor)) if divisor > 0 else stride


def collect_source(
    source_id: str,
    quota: int,
    filters: FilterConfig,
    max_rows: int,
    stride_divisor: float = 1.0,
    data_files: tuple[str, ...] | None = None,
    count_min_words: int = 0,
) -> tuple[SourceReport, list[Candidate]]:
    """Stream one source (or one slice of its ``data_files``) until ``quota`` excerpts of at least
    ``count_min_words`` words pass all per-text filters (runs in a worker). Shorter candidates inside the
    ``filters`` window are kept in the cache too, but do not count towards the quota."""
    from humanizer_eval.config import get_settings

    src = get_source(source_id)
    if data_files is not None:
        src = replace(src, data_files=tuple(data_files))
    stride = effective_stride(src.stride, stride_divisor)
    rep = SourceReport(
        source_id=src.id, genre=src.genre, quota=quota, window=[filters.min_words, filters.max_words]
    )
    rejected: Counter[str] = Counter()
    cands: list[Candidate] = []
    t0 = time.time()
    try:
        exhausted = True
        for idx, row in enumerate(iter_rows(src, get_settings().hf_api_key or None)):
            if rep.rows_scanned >= max_rows or rep.usable >= quota:
                exhausted = False
                break
            if stride > 1 and idx % stride:
                continue
            rep.rows_scanned += 1
            if (row := src.parse_row(row)) is None:
                rejected["unparseable_row"] += 1
                continue
            if src.row_filter is not None and (why := src.row_filter(row)):
                rejected[why] += 1
                continue
            ok, evidence = src.check_date(row)
            if not ok:
                rejected[evidence] += 1
                continue
            docs = src.row_documents(row)
            if not docs:
                rejected["empty"] += 1
                continue
            dt = src.row_date(row) if evidence == "row" else None
            doc_id = src.row_id(row, idx)
            for d_i, doc in enumerate(docs):
                rep.documents += 1
                res = process_document(doc, f"{src.id}:{doc_id}:{d_i}", filters, src.preclean)
                rejected.update(res.rejected)
                for e_i, ex in enumerate(res.excerpts):
                    rep.usable += word_count(ex) >= count_min_words
                    cands.append(
                        Candidate(
                            text=ex,
                            source_id=src.id,
                            genre=src.genre,
                            source_doc_id=f"{doc_id}#{d_i}.{e_i}" if len(docs) > 1 or e_i else doc_id,
                            date=dt.date().isoformat() if dt else "",
                            date_evidence=evidence,
                        )
                    )
                if res.excerpts:
                    break  # one document per row: posts of one thread stay out of the same pool
        rep.exhausted = exhausted
    except Exception as e:  # network/schema errors are reported, not fatal for the build
        rep.error = f"{type(e).__name__}: {e}"[:500]
    rep.rejected = dict(rejected.most_common())
    rep.candidates = len(cands)
    rep.elapsed_s = round(time.time() - t0, 1)
    return rep, cands


def source_jobs(src: Source, quota: int, max_rows: int) -> list[tuple[tuple[str, ...] | None, int, int]]:
    """Worker jobs ``(data_files slice or None, quota, max_rows)`` for one source: a single job, or
    ``src.parallel_files`` contiguous slices of its data files with the quota and row budget split evenly."""
    files = src.data_files or ()
    n = min(src.parallel_files, len(files))
    if n <= 1:
        return [(None, quota, max_rows)]
    size = math.ceil(len(files) / n)
    chunks = [tuple(files[i : i + size]) for i in range(0, len(files), size)]
    return [(c, math.ceil(quota / len(chunks)), math.ceil(max_rows / len(chunks))) for c in chunks]


def merge_reports(
    parts: list[tuple[SourceReport, list[Candidate]]], quota: int
) -> tuple[SourceReport, list[Candidate]]:
    """Combine the slice reports of one source (in slice order, so candidates keep file order)."""
    first = parts[0][0]
    rep = SourceReport(source_id=first.source_id, genre=first.genre, quota=quota, window=first.window)
    rejected: Counter[str] = Counter()
    cands: list[Candidate] = []
    for r, cs in parts:
        rep.rows_scanned += r.rows_scanned
        rep.documents += r.documents
        rep.usable += r.usable
        rejected.update(r.rejected)
        cands.extend(cs)
    rep.rejected = dict(rejected.most_common())
    rep.candidates = len(cands)
    rep.elapsed_s = max(r.elapsed_s for r, _ in parts)
    rep.exhausted = any(r.exhausted for r, _ in parts)  # a slice ran dry: no more under this split
    rep.error = "; ".join(r.error for r, _ in parts if r.error)
    return rep, cands


_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def single_threaded_children() -> Iterator[None]:
    """Spawned workers inherit ``os.environ``: one BLAS thread each. Without this every worker starts a
    pool of ~nproc BLAS threads for langid's tiny matrix products and 20 workers oversubscribe the CPU
    (measured: 5x slower per document). The parent's limits (used later for clustering) are untouched."""
    saved = {k: os.environ.get(k) for k in _THREAD_ENV}
    os.environ.update(dict.fromkeys(_THREAD_ENV, "1"))
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cache_paths(out_dir: Path, source_id: str) -> tuple[Path, Path]:
    d = out_dir / "candidates"
    return d / f"{source_id}.jsonl", d / f"{source_id}.report.json"


def _save_cache(out_dir: Path, rep: SourceReport, cands: list[Candidate]) -> None:
    cpath, rpath = _cache_paths(out_dir, rep.source_id)
    cpath.parent.mkdir(parents=True, exist_ok=True)
    with cpath.open("w") as f:
        for c in cands:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    rpath.write_text(json.dumps(asdict(rep), indent=2))


def take_usable(cands: list[Candidate], quota: int, min_words: int) -> list[Candidate]:
    """The shortest prefix of ``cands`` holding ``quota`` candidates of >= ``min_words`` words (shorter
    candidates inside the prefix stay: a later, lower pool window can use them)."""
    n = 0
    for i, c in enumerate(cands):
        if n >= quota:
            return cands[:i]
        n += word_count(c.text) >= min_words
    return cands


def _load_cache(
    out_dir: Path,
    source_id: str,
    quota: int,
    window: tuple[int, int],
    count_min_words: int = 0,
) -> tuple[SourceReport, list[Candidate]] | None:
    """Cached candidates of a source, or None if missing, collected in another cache ``window``, failed,
    or too few for ``quota`` (unless the source was exhausted)."""
    cpath, rpath = _cache_paths(out_dir, source_id)
    if not (cpath.exists() and rpath.exists()):
        return None
    rep = SourceReport(**json.loads(rpath.read_text()))
    if rep.error or list(rep.window) != list(window):
        return None
    # split on \n only: str.splitlines() would also split on U+2028 inside JSON strings
    cands = [Candidate(**json.loads(line)) for line in cpath.read_text().split("\n") if line.strip()]
    for c in cands:  # caches written by older code may predate a normalization fix
        c.text = normalize_text(c.text)
    usable = sum(word_count(c.text) >= count_min_words for c in cands)
    if usable < quota and not rep.exhausted:
        return None
    return rep, take_usable(cands, max(quota, 0), count_min_words)


def collect_all(cfg: BuildConfig, log=print) -> tuple[dict[str, SourceReport], dict[str, list[Candidate]]]:
    sources = enabled_sources(list(cfg.source_ids) if cfg.source_ids else None)
    by_id = {s.id: s for s in sources}
    quotas = {
        sid: math.ceil(q * cfg.oversample * by_id[sid].oversample)
        for sid, q in source_quotas(sources, cfg.per_genre, cfg.per_genre_overrides).items()
    }
    cache_filters = cfg.cache_filters()
    window = (cache_filters.min_words, cache_filters.max_words)
    reports: dict[str, SourceReport] = {}
    cands: dict[str, list[Candidate]] = {}
    todo = []
    for s in sources:
        cached = (
            _load_cache(cfg.out_dir, s.id, quotas[s.id], window, cfg.min_words)
            if cfg.reuse_candidates
            else None
        )
        if cached:
            reports[s.id], cands[s.id] = cached
            log(f"[cache] {s.id}: {len(cached[1])} candidates")
        else:
            todo.append(s)
    if todo:
        ctx = mp.get_context("spawn")
        jobs = {
            s.id: source_jobs(
                s, quotas[s.id], quotas[s.id] * (s.max_rows_per_candidate or cfg.max_rows_per_candidate)
            )
            for s in todo
        }
        n_jobs = sum(len(j) for j in jobs.values())
        with (
            single_threaded_children(),
            ProcessPoolExecutor(max_workers=min(cfg.workers, n_jobs), mp_context=ctx) as ex,
        ):
            futs = {
                ex.submit(
                    collect_source, s.id, q, cache_filters, rows, cfg.stride_divisor, files, cfg.min_words
                ): (s, i)
                for s in todo
                for i, (files, q, rows) in enumerate(jobs[s.id])
            }
            parts: dict[str, dict[int, tuple[SourceReport, list[Candidate]]]] = defaultdict(dict)
            for fut in as_completed(futs):
                s, i = futs[fut]
                parts[s.id][i] = fut.result()
                if len(parts[s.id]) < len(jobs[s.id]):
                    continue
                ordered = [parts[s.id][k] for k in range(len(jobs[s.id]))]
                rep, cs = ordered[0] if len(ordered) == 1 else merge_reports(ordered, quotas[s.id])
                reports[s.id], cands[s.id] = rep, cs
                if not rep.error:
                    _save_cache(cfg.out_dir, rep, cs)
                status = f"ERROR {rep.error}" if rep.error else f"{rep.usable}/{rep.quota}"
                log(f"[done] {s.id:18s} {status}  rows={rep.rows_scanned} in {rep.elapsed_s}s")
    return reports, cands


# --------------------------------------------------------------------------- assembly
def assemble(
    cfg: BuildConfig, reports: dict[str, SourceReport], cands: dict[str, list[Candidate]]
) -> list[TextRecord]:
    """Fit candidates to the pool window, global dedupe (registry order, first occurrence wins), then
    per-genre selection by source weight."""
    sources = [s for s in enabled_sources(list(cfg.source_ids) if cfg.source_ids else None) if s.id in cands]
    pool_filters = cfg.pool_filters()
    dedup = Deduplicator(pool_filters)
    kept: dict[str, list[Candidate]] = {}
    for s in sources:
        rej: Counter[str] = Counter(reports[s.id].rejected)
        kept[s.id] = []
        for c in cands[s.id]:
            text, reason = fit_window(c.text, f"{c.source_id}:{c.source_doc_id}:window", pool_filters)
            if text is not None:
                reason = dedup.check_and_add(text) or ""
            if reason:
                rej[reason] += 1
            else:
                kept[s.id].append(replace(c, text=text))
        reports[s.id].rejected = dict(rej.most_common())
    targets = source_quotas(sources, cfg.per_genre, cfg.per_genre_overrides)
    chosen: list[Candidate] = []
    for genre in GENRES:
        gs = [s for s in sources if s.genre == genre]
        picked = {s.id: kept[s.id][: targets[s.id]] for s in gs}
        short = cfg.genre_target(genre) - sum(len(v) for v in picked.values())
        for s in gs:  # top up from sources that have spare candidates
            if short <= 0:
                break
            extra = kept[s.id][len(picked[s.id]) : len(picked[s.id]) + short]
            picked[s.id] += extra
            short -= len(extra)
        for s in gs:
            reports[s.id].kept = len(picked[s.id])
            chosen.extend(picked[s.id])
    bins = cfg.length_bins()
    return [
        TextRecord.from_text(
            c.text,
            genre=c.genre,
            source_id=c.source_id,
            source_doc_id=c.source_doc_id,
            date=c.date,
            date_evidence=c.date_evidence,
            bins=bins,
        )
        for c in chosen
    ]


def cluster_records(cfg: BuildConfig, records: list[TextRecord], log=print) -> tuple[list[dict], list[str]]:
    by_genre: dict[str, list[TextRecord]] = defaultdict(list)
    for r in records:
        by_genre[r.genre].append(r)
    clusters: list[dict] = []
    # capitalization evidence from the whole pool: a name that is rare in one genre is common elsewhere
    caps = capitalization_ratios(r.text for r in records)
    proper = proper_from_ratios(caps)
    for genre in GENRES:
        recs = by_genre.get(genre)
        if not recs:
            continue
        texts = [r.text for r in recs]
        res = cluster_genre(
            genre,
            texts,
            min_size=cfg.min_cluster_size,
            size_factor=cfg.cluster_size_factor,
            seed=cfg.seed,
            caps=caps,
        )
        info = {c.id: c for c in res.clusters}
        for r, cid in zip(recs, res.assignments, strict=True):
            r.cluster_id = cid
            r.cluster_label = info[cid].label
        clusters.extend(c.to_dict() for c in res.clusters)
        log(f"[cluster] {genre:13s} n={len(recs):5d} k={len(res.clusters):3d} ({res.embedder})")
    proper_by_genre = {g: proper for g in by_genre}
    problems = check_clusters((r.to_row() for r in records), cfg.min_cluster_size, proper_by_genre)
    return clusters, problems


# --------------------------------------------------------------------------- statistics
def compute_stats(
    cfg: BuildConfig,
    records: list[TextRecord],
    reports: dict[str, SourceReport],
    clusters: list[dict],
    problems,
) -> dict:
    labels = [b[2] for b in cfg.length_bins()]
    subtype = {s.id: s.subtype or "-" for s in SOURCES}
    genres: dict[str, dict] = {}
    for genre in GENRES:
        recs = [r for r in records if r.genre == genre]
        if not recs:
            genres[genre] = {"count": 0}
            continue
        wc = [r.word_count for r in recs]
        sizes = sorted((c["size"] for c in clusters if c["genre"] == genre), reverse=True)
        genres[genre] = {
            "count": len(recs),
            "target": cfg.genre_target(genre),
            "by_source": dict(Counter(r.source_id for r in recs).most_common()),
            "by_subtype": dict(Counter(subtype.get(r.source_id, "-") for r in recs).most_common()),
            "by_length_bin": {b: sum(r.length_bin == b for r in recs) for b in labels},
            "by_register": {g: sum(r.register == g for r in recs) for g in REGISTERS},
            "words": {
                "mean": round(statistics.mean(wc), 1),
                "median": statistics.median(wc),
                "min": min(wc),
                "max": max(wc),
            },
            "clusters": {
                "k": len(sizes),
                "min": sizes[-1],
                "median": statistics.median(sizes),
                "max": sizes[0],
            },
            "date_evidence": dict(Counter(r.date_evidence for r in recs)),
        }
    return {
        "total": len(records),
        "per_genre_target": cfg.per_genre,
        "window": [cfg.min_words, cfg.max_words],
        "cache_window": [cfg.cache_min_words, cfg.cache_max_words],
        "length_bins": labels,
        "min_cluster_size": cfg.min_cluster_size,
        "genres": genres,
        "sources": {sid: asdict(rep) for sid, rep in sorted(reports.items())},
        "cluster_check": {"ok": not problems, "problems": problems},
    }


def format_stats(stats: dict) -> str:
    labels = stats.get("length_bins") or list(LENGTH_BIN_LABELS)
    lines = [f"Pool: {stats['total']} texts (target {stats['per_genre_target']}/genre)", ""]
    hdr = f"{'genre':18s} {'n':>5s} " + " ".join(f"{b:>7s}" for b in labels)
    hdr += f" {'formal':>7s} {'informal':>8s} {'words':>6s} {'k':>3s} {'cl.min':>6s} {'cl.max':>6s}"
    lines.append(hdr)
    for g, s in stats["genres"].items():
        if not s["count"]:
            lines.append(f"{g:18s} {0:5d}  (no source)")
            continue
        row = f"{g:18s} {s['count']:5d} " + " ".join(f"{s['by_length_bin'][b]:7d}" for b in labels)
        row += (
            f" {s['by_register']['formal']:7d} {s['by_register']['informal']:8d} {s['words']['mean']:6.0f}"
            f" {s['clusters']['k']:3d} {s['clusters']['min']:6d} {s['clusters']['max']:6d}"
        )
        lines.append(row)
    lines += ["", "within-genre mix (source subtype):"]
    lines += [
        f"  {g:18s} " + ", ".join(f"{k} {v}" for k, v in s["by_subtype"].items())
        for g, s in stats["genres"].items()
        if s["count"]
    ]
    lines += ["", f"{'source':18s} {'genre':18s} {'rows':>7s} {'cand':>6s} {'kept':>5s}  top rejections"]
    for sid, r in stats["sources"].items():
        top = ", ".join(f"{k}={v}" for k, v in list(r["rejected"].items())[:6])
        err = f"  ERROR: {r['error']}" if r["error"] else ""
        lines.append(
            f"{sid:18s} {r['genre']:18s} {r['rows_scanned']:7d} {r['candidates']:6d} {r['kept']:5d}"
            f"  {top}{err}"
        )
    chk = stats["cluster_check"]
    lines += ["", "cluster check: " + ("OK" if chk["ok"] else f"{len(chk['problems'])} problems")]
    lines += [f"  {p}" for p in chk["problems"][:20]]
    return "\n".join(lines)


def _versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for pkg in ("datasets", "scikit-learn", "langid", "datasketch", "numpy", "pyarrow"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "missing"
    return out


def build_pool(cfg: BuildConfig, log=print) -> dict:
    """Run the whole build; returns the stats dict (also written to ``stats.json``)."""
    from humanizer_eval.pool.viewer import render_samples_html

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    reports, cands = collect_all(cfg, log)
    records = assemble(cfg, reports, cands)
    clusters, problems = cluster_records(cfg, records, log)
    stats = compute_stats(cfg, records, reports, clusters, problems)
    sources = enabled_sources(list(cfg.source_ids) if cfg.source_ids else None)
    manifest = {
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cutoff": CUTOFF.date().isoformat(),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "sources": [
            {
                "id": s.id,
                "genre": s.genre,
                "hf_id": s.hf_id,
                "config": s.config,
                "split": s.split,
                "data_files": list(s.data_files or ()),
                "builder": s.builder,
                "stride": effective_stride(s.stride, cfg.stride_divisor),
                "license": s.license,
                "date_evidence": s.date_evidence,
                "date_field": s.date_field,
                "weight": s.weight,
                "subtype": s.subtype,
            }
            for s in sources
        ],
        "embedder": "tfidf-svd",
        "versions": _versions(),
    }
    rows = [r.to_row() for r in records]
    write_pool(cfg.out_dir, rows, clusters, manifest)
    (cfg.out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    html = render_samples_html(rows, clusters, stats, per_genre=cfg.samples_per_genre, seed=cfg.seed)
    (cfg.out_dir / "samples.html").write_text(html, encoding="utf-8")
    stats["elapsed_s"] = round(time.time() - t0, 1)
    return stats
