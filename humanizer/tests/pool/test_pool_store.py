import itertools
from collections import Counter

import pytest

from humanizer_eval.pool.metadata import PUBLIC_FIELDS, text_id
from humanizer_eval.pool.store import (
    SLOT_FIELDS,
    PoolStore,
    ReferenceText,
    Slot,
    allocate,
    content_sha256,
    sample_reference,
    sample_slots,
    write_pool,
)


def make_pool(per_stratum=6, genres=("essay", "news", "review")):
    rows = []
    for g, c, b, r in itertools.product(genres, range(3), ("150-199", "200-249"), ("formal", "informal")):
        for i in range(per_stratum + c):  # uneven strata sizes
            text = f"{g} cluster {c} bin {b} {r} text number {i}"
            rows.append(
                {
                    "text_id": text_id(text),
                    "text": text,
                    "genre": g,
                    "source_id": f"src_{g}",
                    "source_doc_id": str(i),
                    "word_count": 180 if b == "150-199" else 220,
                    "length_bin": b,
                    "register": r,
                    "register_score": 1.0 if r == "informal" else -1.0,
                    "date": "",
                    "date_evidence": "dataset",
                    "cluster_id": f"{g}.c{c:02d}",
                    "cluster_label": f"topic, words, {'abc'[c]}",
                }
            )
    return rows


PROPS = {"essay": 0.5, "news": 0.3, "review": 0.2}


# --------------------------------------------------------------------------- allocation
def test_allocate_largest_remainder():
    assert allocate(10, {"a": 0.5, "b": 0.3, "c": 0.2}, "abc") == {"a": 5, "b": 3, "c": 2}
    out = allocate(7, {"a": 1, "b": 1, "c": 1}, "abc")
    assert sum(out.values()) == 7 and out == {"a": 3, "b": 2, "c": 2}  # tie -> genre name order
    assert allocate(0, {"a": 1}, "a") == {"a": 0}


def test_allocate_missing_genre():
    with pytest.raises(ValueError, match="absent"):
        allocate(10, {"a": 0.5, "z": 0.5}, "a")
    assert allocate(10, {"a": 0.5, "z": 0.5}, "a", missing="renormalize") == {"a": 10}


# --------------------------------------------------------------------------- slots (D-51)
def test_sample_slots_never_returns_text_or_ids():
    pool = make_pool()
    meta = [{f: r[f] for f in PUBLIC_FIELDS} for r in pool]
    slots = sample_slots(meta, 60, PROPS, seed=7)
    texts = {r["text"] for r in pool}
    ids = {r["text_id"] for r in pool}
    for s in slots:
        assert isinstance(s, Slot) and s._fields == SLOT_FIELDS
        assert not hasattr(s, "text") and not hasattr(s, "text_id")
        assert not (set(s) & texts) and not (set(s) & ids)
    # even when given full rows (with text), only the stratum tuple comes back
    assert sample_slots(pool, 60, PROPS, seed=7) == slots


def test_sample_slots_is_deterministic_and_seed_dependent():
    meta = make_pool()
    a = sample_slots(meta, 60, PROPS, seed="period-1")
    assert a == sample_slots(list(reversed(meta)), 60, PROPS, seed="period-1")  # order-independent
    assert a != sample_slots(meta, 60, PROPS, seed="period-2")
    assert sample_slots(meta, 60, PROPS, seed=5) == sample_slots(meta, 60, PROPS, seed="5")
    assert sample_slots(meta, 60, PROPS, seed=b"\x00\x01") == sample_slots(meta, 60, PROPS, seed=b"\x00\x01")


def test_sample_slots_genre_counts_and_strata_follow_pool():
    meta = make_pool()
    slots = sample_slots(meta, 60, PROPS, seed=1)
    assert Counter(s.genre for s in slots) == {"essay": 30, "news": 18, "review": 12}
    # asking for the whole genre returns exactly the pool's strata histogram
    n_news = sum(r["genre"] == "news" for r in meta)
    full = sample_slots(meta, n_news, {"news": 1.0}, seed=1)
    pool_hist = Counter(Slot(*(r[f] for f in SLOT_FIELDS)) for r in meta if r["genre"] == "news")
    assert Counter(full) == pool_hist


def test_sample_slots_errors():
    meta = make_pool()
    with pytest.raises(ValueError, match="absent"):
        sample_slots(meta, 10, {"essay": 0.5, "email_letter": 0.5})
    with pytest.raises(ValueError, match="slots requested"):
        sample_slots(meta, 10_000, {"essay": 1.0})


# --------------------------------------------------------------------------- reference (D-54)
def test_sample_reference_matches_stratum_counts_exactly():
    pool = make_pool()
    slots = sample_slots(pool, 60, PROPS, seed=3)
    refs = sample_reference(pool, slots, seed=3)
    assert all(isinstance(r, ReferenceText) for r in refs)
    assert Counter(r.slot for r in refs) == Counter(slots)
    assert all(r.match == "exact" and r.actual == r.slot for r in refs)
    assert len({r.text_id for r in refs}) == len(refs)  # no text twice
    by_id = {r["text_id"]: r for r in pool}
    assert all(by_id[r.text_id]["text"] == r.text for r in refs)


def test_sample_reference_is_deterministic_and_independent_of_slot_draw():
    pool = make_pool()
    slots = sample_slots(pool, 60, PROPS, seed=3)
    a = sample_reference(pool, slots, seed=3)
    assert a == sample_reference(list(reversed(pool)), Counter(slots), seed=3)
    assert [r.text_id for r in a] != [r.text_id for r in sample_reference(pool, slots, seed=4)]


def test_sample_reference_excludes_ids_and_relaxes_when_needed():
    pool = make_pool(per_stratum=2)
    slot = Slot("news", "news.c00", "topic, words, a", "150-199", "formal")
    exact = [r for r in pool if Slot(*(r[f] for f in SLOT_FIELDS)) == slot]
    assert len(exact) == 2
    refs = sample_reference(pool, {slot: 2}, seed=0, exclude_ids=[exact[0]["text_id"]])
    assert exact[0]["text_id"] not in {r.text_id for r in refs}
    assert [r.match for r in refs] == ["exact", "any_register"]
    assert refs[1].actual.length_bin == slot.length_bin and refs[1].actual.cluster_id == slot.cluster_id
    many = sample_reference(pool, {slot: 9}, seed=0)  # beyond the cluster -> genre x length (D-20 group)
    assert Counter(r.match for r in many)["genre_length"] > 0
    assert all(r.actual.genre == "news" and r.actual.length_bin == "150-199" for r in many)
    with pytest.raises(ValueError):
        sample_reference(pool, {slot: 3}, seed=0, relax=False)
    with pytest.raises(ValueError):
        sample_reference(pool, {slot: 10_000}, seed=0)


def test_exact_matches_are_served_before_any_relaxation():
    pool = make_pool(per_stratum=1)
    a = Slot("essay", "essay.c00", "topic, words, a", "150-199", "formal")
    b = Slot("essay", "essay.c00", "topic, words, a", "150-199", "informal")
    # a needs more than its stratum holds; b must still get its own exact text
    refs = sample_reference(pool, {a: 2, b: 1}, seed=0)
    assert [r.match for r in refs if r.slot == b] == ["exact"]


# --------------------------------------------------------------------------- storage
def test_write_and_load_roundtrip(tmp_path):
    import pyarrow.parquet as pq

    pool = make_pool()
    clusters = [{"id": "essay.c00", "genre": "essay", "label": "topic, words, a", "size": 6}]
    paths = write_pool(tmp_path / "p", pool, clusters, {"note": "test"})
    assert "text" not in pq.read_schema(paths["metadata"]).names
    store = PoolStore.load(tmp_path / "p")
    assert len(store) == len(pool) and store.clusters == clusters
    assert store.manifest["n_texts"] == len(pool) and store.manifest["note"] == "test"
    assert store.manifest["content_sha256"] == content_sha256(reversed(pool))
    assert all("text" not in m for m in store.metadata())
    assert store.genres() == ["essay", "news", "review"]
    assert sum(store.strata_counts().values()) == len(pool)
    slots = store.sample_slots(30, PROPS, seed=9)
    assert slots == sample_slots(pool, 30, PROPS, seed=9)
    assert Counter(r.slot for r in store.sample_reference(slots, seed=9)) == Counter(slots)
