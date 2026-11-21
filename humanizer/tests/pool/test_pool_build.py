import dataclasses
from datetime import UTC, datetime

from pool_helpers import distinct_doc, doc_with_paragraphs

from humanizer_eval.pool import build
from humanizer_eval.pool.build import (
    BuildConfig,
    Candidate,
    SourceReport,
    assemble,
    collect_source,
    source_quotas,
)
from humanizer_eval.pool.clean import DEFAULT_FILTERS
from humanizer_eval.pool.sources import get_source
from humanizer_eval.pool.viewer import render_samples_html


def test_source_quotas_split_genre_by_weight():
    srcs = [get_source("yelp"), get_source("imdb"), get_source("enron")]
    q = source_quotas(srcs, 100)
    assert q == {"yelp": 65, "imdb": 35, "enron": 100}


def test_collect_source_filters_dates_and_text(monkeypatch):
    good = doc_with_paragraphs([80, 90])
    rows = [
        {"from": "a.b@enron.com", "message_id": "a", "date": datetime(2001, 3, 1, tzinfo=UTC), "body": good},
        {
            "from": "a.b@enron.com",
            "message_id": "b",
            "date": datetime(2022, 3, 1, tzinfo=UTC),
            "body": good,
        },  # too late
        {"from": "a.b@enron.com", "message_id": "c", "date": None, "body": good},  # no evidence
        {
            "from": "a.b@enron.com",
            "message_id": "d",
            "date": datetime(2000, 1, 1, tzinfo=UTC),
            "body": "short",
        },
        {
            "from": "a.b@enron.com",
            "message_id": "e",
            "date": datetime(2000, 1, 2, tzinfo=UTC),
            "body": good + " jane@example.com",
        },
        {
            "from": "a.b@enron.com",
            "message_id": "f",
            "date": datetime(2000, 1, 3, tzinfo=UTC),
            "body": doc_with_paragraphs([70, 95]),
        },
        {"from": "news@ft.com", "message_id": "g", "date": datetime(2000, 1, 4, tzinfo=UTC), "body": good},
    ]
    monkeypatch.setattr(build, "iter_rows", lambda src, token: iter(rows))
    monkeypatch.setattr(build, "get_source", lambda sid: dataclasses.replace(get_source(sid), stride=1))
    rep, cands = collect_source("enron", quota=10, filters=DEFAULT_FILTERS, max_rows=100)
    assert rep.error == "" and rep.exhausted
    assert [c.source_doc_id for c in cands] == ["a", "f"]
    assert cands[0].date == "2001-03-01" and cands[0].date_evidence == "row"
    assert rep.rejected["date_too_late"] == 1 and rep.rejected["date_missing"] == 1
    assert rep.rejected["too_short"] == 1 and rep.rejected["pii_email"] == 1
    assert rep.rejected["external_sender"] == 1  # newsletters/ads are not written by Enron staff


def test_collect_source_respects_quota_and_stride(monkeypatch):
    rows = [{"full_text": doc_with_paragraphs([80, 90 + i])} for i in range(100)]
    monkeypatch.setattr(build, "iter_rows", lambda src, token: iter(rows))
    rep, cands = collect_source("persuade2", quota=5, filters=DEFAULT_FILTERS, max_rows=1000)
    assert len(cands) == 5 and not rep.exhausted
    stride = get_source("persuade2").stride
    assert [c.source_doc_id for c in cands] == [f"row{i * stride}" for i in range(5)]


def test_stride_divisor_scales_registry_strides_for_large_builds(monkeypatch):
    assert build.effective_stride(50, 10) == 5
    assert build.effective_stride(3, 10) == 1 and build.effective_stride(7, 1) == 7
    rows = [{"full_text": doc_with_paragraphs([80, 90 + i])} for i in range(100)]
    monkeypatch.setattr(build, "iter_rows", lambda src, token: iter(rows))
    _, cands = collect_source("persuade2", quota=5, filters=DEFAULT_FILTERS, max_rows=1000, stride_divisor=7)
    assert [c.source_doc_id for c in cands] == [f"row{i}" for i in range(5)]  # stride 7 / 7 -> 1


class _InlineExecutor:  # run "workers" in-process
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args):
        import concurrent.futures

        fut = concurrent.futures.Future()
        fut.set_result(fn(*args))
        return fut


def test_per_source_row_budget_and_parallel_slices(monkeypatch, tmp_path):
    calls = []

    def fake_collect(sid, quota, filters, max_rows, stride_divisor, data_files, count_min_words):
        calls.append((sid, quota, max_rows, stride_divisor, data_files))
        assert (filters.min_words, filters.max_words, count_min_words) == (100, 500, 150)  # cache window
        n = len(data_files or ())
        rep = SourceReport(source_id=sid, genre=get_source(sid).genre, quota=quota, rows_scanned=n)
        rep.rejected = {"too_short": 1}
        return rep, [_cand(f"text {f}", sid, rep.genre, f) for f in (data_files or ("only",))]

    monkeypatch.setattr(build, "collect_source", fake_collect)
    monkeypatch.setattr(build, "ProcessPoolExecutor", _InlineExecutor)
    cfg = BuildConfig(out_dir=tmp_path, per_genre=100, source_ids=("enron", "c4_letters"), stride_divisor=2)
    reports, cands = build.collect_all(cfg, log=lambda *_: None)
    letters = get_source("c4_letters")
    slices = [c for c in calls if c[0] == "c4_letters"]
    assert len(slices) == letters.parallel_files == 8
    assert [f for c in slices for f in c[4]] == list(letters.data_files)  # contiguous, in order, complete
    q = reports["c4_letters"].quota
    assert all(c[1] == -(-q // 8) and c[2] == -(-q * letters.max_rows_per_candidate // 8) for c in slices)
    assert all(c[3] == 2 for c in calls)
    assert reports["c4_letters"].rows_scanned == len(letters.data_files)
    assert reports["c4_letters"].rejected == {"too_short": 8}
    assert [c.source_doc_id for c in cands["c4_letters"]] == list(letters.data_files)
    (enron,) = [c for c in calls if c[0] == "enron"]
    assert enron[4] is None and enron[2] == enron[1] * cfg.max_rows_per_candidate


def test_source_jobs_single_and_split():
    enron = get_source("enron")
    assert build.source_jobs(enron, 10, 600) == [(None, 10, 600)]
    one = dataclasses.replace(get_source("c4_letters"), data_files=("a",))
    assert build.source_jobs(one, 10, 600) == [(None, 10, 600)]  # nothing to split
    three = dataclasses.replace(get_source("c4_letters"), data_files=("a", "b", "c"), parallel_files=2)
    assert build.source_jobs(three, 10, 601) == [(("a", "b"), 5, 301), (("c",), 5, 301)]


def test_workers_are_spawned_with_one_blas_thread(monkeypatch):
    import os

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    with build.single_threaded_children():
        assert os.environ["OMP_NUM_THREADS"] == "1" and os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "8" and "OPENBLAS_NUM_THREADS" not in os.environ


def test_iter_rows_reads_raw_files_with_a_builder(monkeypatch):
    import datasets

    calls = []

    def fake_load(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        return iter([{"premises": []}])

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    rows = list(build.iter_rows(get_source("argsme"), token=None))
    assert rows == [{"premises": []}]
    path, args, kwargs = calls[0]
    assert path == "json" and not args
    assert kwargs["data_files"] == ["hf://datasets/webis/args_me/args-me.jsonl"]
    assert kwargs["streaming"] is True


def test_collect_source_reports_errors(monkeypatch):
    def boom(src, token):
        raise ConnectionError("offline")

    monkeypatch.setattr(build, "iter_rows", boom)
    rep, cands = collect_source("imdb", quota=5, filters=DEFAULT_FILTERS, max_rows=10)
    assert cands == [] and "offline" in rep.error


def _cand(text, sid, genre, i):
    return Candidate(
        text=text, source_id=sid, genre=genre, source_doc_id=str(i), date="", date_evidence="dataset"
    )


def test_assemble_dedupes_across_sources_and_caps_per_genre(tmp_path):
    texts = [distinct_doc(i) for i in range(8)]
    cands = {
        "yelp": [_cand(t, "yelp", "review", i) for i, t in enumerate(texts[:6])],
        "imdb": [_cand(texts[0], "imdb", "review", 0)]
        + [_cand(t, "imdb", "review", i) for i, t in enumerate(texts[6:])],
    }
    reports = {sid: SourceReport(source_id=sid, genre="review", quota=10) for sid in cands}
    cfg = BuildConfig(out_dir=tmp_path, per_genre=7, source_ids=("yelp", "imdb"))
    records = assemble(cfg, reports, cands)
    assert len(records) == 7
    assert len({r.text_id for r in records}) == 7
    assert reports["imdb"].rejected.get("exact_dup") == 1  # same text as a yelp candidate
    assert sum(r.source_id == "imdb" for r in records) == 2  # its non-duplicate candidates


def test_viewer_escapes_text_and_shows_labels():
    row = {
        "text_id": "abc",
        "text": "Hello <script>alert(1)</script>\n\nSecond paragraph.",
        "genre": "review",
        "source_id": "yelp",
        "source_doc_id": "1",
        "word_count": 180,
        "length_bin": "150-199",
        "register": "informal",
        "register_score": 2.0,
        "date": "2015-01-01",
        "date_evidence": "row",
        "cluster_id": "review.c00",
        "cluster_label": "food, service, staff",
    }
    clusters = [
        {"id": "review.c00", "genre": "review", "label": "food, service, staff", "size": 1, "top_terms": []}
    ]
    stats = {"total": 1, "genres": {}, "cluster_check": {"ok": True, "problems": []}, "min_cluster_size": 1}
    html = render_samples_html([row], clusters, stats, per_genre=5)
    assert "<script>alert" not in html and "&lt;script&gt;" in html
    assert "food, service, staff" in html and "Second paragraph." in html


def test_candidate_cache_roundtrip_keeps_unicode_separators(tmp_path):
    text = doc_with_paragraphs([80, 90])
    cands = [_cand(text, "imdb", "review", 0), _cand(text.replace(". ", ".\u2028", 1), "imdb", "review", 1)]
    rep = SourceReport(source_id="imdb", genre="review", quota=2, candidates=2, window=[100, 500])
    build._save_cache(tmp_path, rep, cands)
    loaded = build._load_cache(tmp_path, "imdb", quota=2, window=(100, 500))
    assert loaded is not None
    _, back = loaded
    assert len(back) == 2 and back[0].text == text and "\u2028" not in back[1].text
    assert build._load_cache(tmp_path, "imdb", quota=2, window=(150, 350)) is None  # other cache window


# --------------------------------------------------------------------------- cache window / pool window
_LOOSE = dataclasses.replace(DEFAULT_FILTERS, min_type_token_ratio=0.0)  # distinct_doc has a tiny vocabulary


def test_build_config_windows_and_validation(tmp_path):
    import pytest

    cfg = BuildConfig(out_dir=tmp_path)
    assert (cfg.min_words, cfg.max_words, cfg.cache_min_words, cfg.cache_max_words) == (150, 350, 100, 500)
    pool, cache = cfg.pool_filters(), cfg.cache_filters()
    assert (pool.min_words, pool.max_words, pool.target_mean_words) == (150, 350, 230.0)
    assert (cache.min_words, cache.max_words) == (100, 500)
    assert (cache.target_mean_words, cache.target_sd_words) == (500.0, 0.0)  # longest prefix that fits
    assert [b[2] for b in BuildConfig(out_dir=tmp_path, max_words=300).length_bins()][-1] == "250-300"
    with pytest.raises(ValueError):
        BuildConfig(out_dir=tmp_path, min_words=90)  # outside the cache window
    with pytest.raises(ValueError):
        BuildConfig(out_dir=tmp_path, max_words=600)


def test_collect_source_caches_short_candidates_but_counts_only_usable_ones(monkeypatch, tmp_path):
    rows = [{"full_text": doc_with_paragraphs([60, 50 + 2 * i])} for i in range(60)]  # ~110 .. ~230 words
    monkeypatch.setattr(build, "iter_rows", lambda src, token: iter(rows))
    monkeypatch.setattr(build, "get_source", lambda sid: dataclasses.replace(get_source(sid), stride=1))
    cache = BuildConfig(out_dir=tmp_path).cache_filters()
    rep, cands = collect_source("persuade2", quota=5, filters=cache, max_rows=1000, count_min_words=150)
    wcs = [len(c.text.split()) for c in cands]
    assert rep.usable == 5 == sum(w >= 150 for w in wcs) and min(wcs) < 150 and rep.window == [100, 500]
    usable = [c for c in cands if len(c.text.split()) >= 150]
    assert build.take_usable(cands, 2, 150)[-1] is usable[1]


def test_assemble_fits_candidates_to_the_pool_window(tmp_path):
    long = "\n\n".join(distinct_doc(i, 120) for i in range(4))  # ~480 words of distinct paragraphs
    texts = {"short": distinct_doc(10, 120), "mid": distinct_doc(11, 240), "long": long}
    cands = {"yelp": [_cand(t, "yelp", "review", k) for k, t in texts.items()]}
    reports = {"yelp": SourceReport(source_id="yelp", genre="review", quota=3)}
    cfg = BuildConfig(out_dir=tmp_path, per_genre=10, source_ids=("yelp",), filters=_LOOSE)
    records = {r.source_doc_id: r for r in assemble(cfg, reports, cands)}
    assert set(records) == {"mid", "long"} and reports["yelp"].rejected == {"window_short": 1}
    assert records["mid"].text == texts["mid"]
    assert 150 <= records["long"].word_count <= 350 and long.startswith(records["long"].text)
    narrow = BuildConfig(out_dir=tmp_path, per_genre=10, source_ids=("yelp",), max_words=200, filters=_LOOSE)
    reports = {"yelp": SourceReport(source_id="yelp", genre="review", quota=3)}
    recs = assemble(narrow, reports, cands)
    assert recs and all(150 <= r.word_count <= 200 and r.length_bin == "150-200" for r in recs)


def test_per_genre_overrides_change_quotas_and_genre_targets(tmp_path):
    srcs = [get_source("yelp"), get_source("imdb"), get_source("enron")]
    assert source_quotas(srcs, 100, {"review": 20}) == {"yelp": 13, "imdb": 7, "enron": 100}
    cfg = BuildConfig(out_dir=tmp_path, per_genre=100, per_genre_overrides={"review": 20})
    assert cfg.genre_target("review") == 20 and cfg.genre_target("news") == 100
