import random

import numpy as np
import pytest

from humanizer_eval.pool.cluster import (
    Embedder,
    TfidfSvdEmbedder,
    capitalization_ratios,
    check_clusters,
    check_label,
    cluster_genre,
    make_labels,
    merge_small_clusters,
    proper_terms,
)

TOPICS = {
    "cooking": "recipe oven flour butter sugar dough bake kitchen pan sauce garlic onion taste".split(),
    "finance": "loan bank interest mortgage credit savings budget investment tax income payment debt".split(),
    "garden": "soil seeds plants water compost tomatoes weeds roots garden flowers harvest shade".split(),
}
FILLER = "the a and of to in that it was for on with as at this but they from have very really".split()
NAMES = ["Johnson", "Martinez", "Chicago", "Toyota"]


def make_corpus(n_per_topic=120, seed=1):
    rng = random.Random(seed)
    texts, truth = [], []
    for topic, vocab in TOPICS.items():
        for _ in range(n_per_topic):
            words = []
            for _ in range(60):
                words.append(rng.choice(vocab) if rng.random() < 0.45 else rng.choice(FILLER))
            # proper nouns and numbers appear mid-sentence, capitalized -- they must never become labels
            words.insert(5, rng.choice(NAMES))
            words.insert(9, str(rng.randint(1990, 2020)))
            words.insert(20, NAMES[0])
            texts.append("Then " + " ".join(words) + ".")
            truth.append(topic)
    return texts, truth


def test_embedder_protocol_and_normalization():
    texts, _ = make_corpus(30)
    emb = TfidfSvdEmbedder(n_components=16)
    assert isinstance(emb, Embedder)
    x = emb.fit_transform(texts)
    assert x.shape == (len(texts), 16)
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)
    assert emb.transform(texts[:3]).shape == (3, 16)


def test_clusters_respect_min_size_and_recover_topics():
    texts, truth = make_corpus(120)
    res = cluster_genre("review", texts, min_size=50, size_factor=2, seed=0)
    sizes = [c.size for c in res.clusters]
    assert len(sizes) >= 2 and min(sizes) >= 50 and sum(sizes) == len(texts)
    # purity: each cluster is dominated by one synthetic topic
    for c in res.clusters:
        members = [t for t, a in zip(truth, res.assignments, strict=True) if a == c.id]
        top = max(members.count(t) for t in TOPICS)
        assert top / len(members) > 0.8
    proper = proper_terms(texts)
    assert {"johnson", "martinez", "chicago"} <= proper
    for c in res.clusters:
        assert check_label(c.label, proper) == []
        assert not any(name.lower() in c.label for name in NAMES)


def test_cluster_ids_are_stable_and_sorted_by_size():
    texts, _ = make_corpus(80)
    a = cluster_genre("essay", texts, min_size=40, size_factor=2, seed=3)
    b = cluster_genre("essay", texts, min_size=40, size_factor=2, seed=3)
    assert a.assignments == b.assignments and [c.label for c in a.clusters] == [c.label for c in b.clusters]
    assert [c.size for c in a.clusters] == sorted((c.size for c in a.clusters), reverse=True)
    assert all(c.id.startswith("essay.c") for c in a.clusters)


def test_genre_smaller_than_min_size_is_one_cluster():
    texts, _ = make_corpus(10)
    res = cluster_genre("news", texts, min_size=300)
    assert len(res.clusters) == 1 and res.clusters[0].size == len(texts)


def test_custom_embedder_can_be_plugged_in():
    class OneHotTopic:
        name = "fake"

        def fit_transform(self, texts):
            counts = [[t.count(w) for w in ("oven", "bank", "soil")] for t in texts]
            return np.array(counts, dtype=float) + 1e-3

        def transform(self, texts):
            return self.fit_transform(texts)

    texts, _ = make_corpus(60)
    res = cluster_genre("forum_qa", texts, min_size=30, size_factor=1, embedder=OneHotTopic())
    assert res.embedder == "fake" and min(c.size for c in res.clusters) >= 30


def test_merge_small_clusters():
    x = np.vstack([np.tile([1.0, 0.0], (50, 1)), np.tile([0.0, 1.0], (50, 1)), np.tile([0.9, 0.1], (5, 1))])
    labels = np.array([0] * 50 + [1] * 50 + [2] * 5)
    merged = merge_small_clusters(x, labels, min_size=20)
    assert set(merged.tolist()) == {0, 1}
    assert (merged[-5:] == 0).all()  # joined the nearest centroid


def test_capitalization_counts_signature_lines():
    texts = ["Thanks for the update.\n\nPhillip", "Jeff,\n\nPlease call me.\n\nPhillip Allen"] * 3
    caps = capitalization_ratios(texts)
    assert caps["phillip"] == 1.0 and caps["jeff"] == 1.0
    assert caps["update"] == 0.0


@pytest.mark.parametrize(
    "label,problem",
    [
        ("", "empty"),
        ("election, Obama, vote", "uppercase"),
        ("covid, 2020, masks", "digit"),
        ("food, the, menu", "stopword:the"),
        ("food, café, menu", "non_generic_term:café"),
        ("a, b, c, d, e, f", "too_many_terms"),
    ],
)
def test_check_label_rejects_specifics(label, problem):
    assert any(p.startswith(problem) for p in check_label(label))


def test_check_label_accepts_generic_and_uses_proper_vocabulary():
    assert check_label("food, service, staff") == []
    assert check_label("general") == []
    assert check_label("johnson, loans, bank", proper={"johnson"}) == ["proper_noun:johnson"]


def test_make_labels_are_unique_within_genre():
    labels = make_labels({0: ["food", "menu", "staff", "price"], 1: ["food", "menu", "staff", "wine"]})
    assert labels[0] != labels[1]


def test_check_clusters_reports_problems():
    ok = [
        {"genre": "review", "cluster_id": "review.c00", "cluster_label": "food, menu", "text_id": str(i)}
        for i in range(60)
    ] + [
        {"genre": "review", "cluster_id": "review.c01", "cluster_label": "hotel, room", "text_id": f"b{i}"}
        for i in range(60)
    ]
    assert check_clusters(ok, min_size=50) == []
    extra = {"genre": "review", "cluster_id": "review.c02", "cluster_label": "car, road", "text_id": "z"}
    small = [*ok, extra]
    assert any(p.startswith("size:review.c02") for p in check_clusters(small, 50))
    dup = [dict(r, cluster_label="food, menu") for r in ok]
    assert any(p.startswith("duplicate_label") for p in check_clusters(dup, 50))
    mixed = [*ok, dict(ok[0], cluster_label="other, label")]
    assert any(p.startswith("inconsistent_label") for p in check_clusters(mixed, 50))
    bad = [dict(r, cluster_label="Paris, 2019") for r in ok[:60]]
    assert any(p.startswith("label:review.c00") for p in check_clusters(bad, 50))
    assert any(p.startswith("unassigned") for p in check_clusters([dict(ok[0], cluster_id="")], 1))
