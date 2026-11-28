from humanizer_eval.evaluation.context import _subset_of_larger, save_human_scores

KEY = b"k" * 32


def test_a_changed_crown_set_reuses_the_shared_detectors_scores(tmp_path):
    ids = ["h1", "h2"]
    old = {"gptzero": [0.1, 0.2], "pangram": [0.0, 0.3], "zerogpt": [0.5, 0.6]}
    base = {d: {"versions": [f"v-{d}"]} for d in old}
    save_human_scores(tmp_path / "human_scores.gptzero-pangram-zerogpt.enc", KEY, ids, old, base)
    got = _subset_of_larger(tmp_path / "human_scores.gptzero-originality-pangram.enc", KEY,
                            {"gptzero", "originality", "pangram"})  # fmt: skip
    assert got["ids"] == ids and set(got["scores"]) == {"gptzero", "pangram"}  # originality: scored anew
    assert got["scores"]["pangram"] == [0.0, 0.3] and set(got["baseline"]) == {"gptzero", "pangram"}
    assert _subset_of_larger(tmp_path / "human_scores.x.enc", KEY, {"originality"}) is None
