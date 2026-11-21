import json

import pytest

from humanizer_eval.detectors.base import DetectorScore
from humanizer_eval.evaluation.context import load_period
from humanizer_eval.period.crypto import encrypt_at_rest

KEY = b"k" * 32


class Flaky:
    """Fails each listed text ``fails`` times (a vendor's "try again"), then scores it."""

    name = "flaky"

    def __init__(self, bad: set[str], fails: int):
        self.left = {t: fails for t in bad}
        self.calls = []

    def score(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            if self.left.get(t, 0) > 0:
                self.left[t] -= 1
                out.append(DetectorScore("flaky", "x", 0.0, error="code=500 try again"))
            else:
                out.append(DetectorScore("flaky", "x", 0.1))
        return out


def _period(tmp_path):
    items = [{"id": f"i{i}", "text": f"ai text {i}", "facts": None, "slot": {"genre": "essay", "length_bin": "m"}}
             for i in range(3)]  # fmt: skip
    human = [{"id": f"h{i}", "genre": "essay", "length_bin": "m"} for i in range(4)]
    (tmp_path / "manifest.json").write_text(json.dumps({"period_id": "p-t", "block_hash": "00" * 32}))
    (tmp_path / "secret.hex").write_text("11" * 32)
    blob = json.dumps({"items": items, "human_ref": human}).encode()
    (tmp_path / "dataset.enc").write_bytes(encrypt_at_rest(KEY, blob))
    return {f"h{i}": f"human text {i}" for i in range(4)}


def test_transient_errors_on_human_texts_are_retried(tmp_path):
    texts = _period(tmp_path)
    det = Flaky({"human text 1", "human text 3"}, fails=2)
    ctx = load_period(tmp_path, KEY, human_texts=texts, detectors={"flaky": det}, retry_wait_s=0)
    assert ctx.human_scores["flaky"] == [0.1] * 4
    assert det.calls[1:] == [["human text 1", "human text 3"]] * 2  # only the failed texts again


def test_a_text_that_keeps_failing_stops_the_load(tmp_path):
    texts = _period(tmp_path)
    det = Flaky({"human text 2"}, fails=10)
    with pytest.raises(RuntimeError, match="1 human texts could not be scored"):
        load_period(tmp_path, KEY, human_texts=texts, detectors={"flaky": det}, retry_wait_s=0)
    assert len(det.calls) == 4  # the first pass and three retries


def test_anchored_facts_from_the_builder_are_used_as_they_are(tmp_path):
    texts = _period(tmp_path)
    items = [{"id": "i0", "text": "ai text 0", "slot": {"genre": "essay", "length_bin": "m"},
              "facts": [{"text": "A fact.", "sents": [0]}]}]  # fmt: skip
    human = [{"id": f"h{i}", "genre": "essay", "length_bin": "m"} for i in range(4)]
    (tmp_path / "dataset.enc").write_bytes(
        encrypt_at_rest(KEY, json.dumps({"items": items, "human_ref": human}).encode())
    )

    class NoLLM:
        def chat(self, *a, **kw):
            raise AssertionError("facts were recomputed although the period has anchored facts")

    ctx = load_period(tmp_path, KEY, human_texts=texts, detectors={}, facts_llm=NoLLM())
    assert ctx.items[0]["facts"] == [{"text": "A fact.", "sents": [0]}]


def test_human_scores_are_kept_for_the_next_load(tmp_path):
    texts = _period(tmp_path)
    det = Flaky(set(), fails=0)
    cache = tmp_path / "human_scores.enc"
    load_period(tmp_path, KEY, human_texts=texts, detectors={"flaky": det}, human_cache=cache)
    assert len(det.calls) == 1 and cache.exists()
    assert b"human" not in cache.read_bytes()  # encrypted: the ids stay hidden until the reveal
    ctx = load_period(tmp_path, KEY, human_texts=texts, detectors={"flaky": det}, human_cache=cache)
    assert len(det.calls) == 1 and ctx.human_scores["flaky"] == [0.1] * 4
    # a new detector is scored and added; the old one still comes from the file
    other = Flaky(set(), fails=0)
    other.name = "other"
    ctx = load_period(tmp_path, KEY, human_texts=texts, detectors={"flaky": det, "other": other},
                      human_cache=cache)  # fmt: skip
    assert len(det.calls) == 1 and len(other.calls) == 1 and set(ctx.human_scores) == {"flaky", "other"}


class Versioned:
    """Scores 0.1 and reports model ``version`` (a list: one per call, to simulate a change mid-way)."""

    def __init__(self, name, versions):
        self.name, self.versions, self.calls = name, list(versions), 0

    def score(self, texts):
        v = self.versions[min(self.calls, len(self.versions) - 1)]
        self.calls += 1
        out = []
        for i in range(len(texts)):
            ver = v[i % len(v)] if isinstance(v, tuple) else v
            out.append(DetectorScore(self.name, "x", 0.1, version=ver))
        return out


def test_the_detector_models_of_h_are_kept_with_its_scores(tmp_path):
    texts = _period(tmp_path)
    cache = tmp_path / "human_scores.enc"
    ctx = load_period(
        tmp_path, KEY, human_texts=texts, detectors={"gz": Versioned("gz", ["m-1"])}, human_cache=cache
    )
    assert ctx.meta["detector_baseline"] == {"gz": {"versions": ["m-1"]}}
    again = Versioned("gz", ["m-2"])  # not asked: the scores and their baseline come from the file
    ctx = load_period(tmp_path, KEY, human_texts=texts, detectors={"gz": again}, human_cache=cache)
    assert again.calls == 0 and ctx.meta["detector_baseline"]["gz"]["versions"] == ["m-1"]


def test_h_scored_across_a_model_change_is_refused(tmp_path):
    texts = _period(tmp_path)
    from humanizer_eval.evaluation import conditions

    ctx = load_period(tmp_path, KEY, human_texts=texts, detectors={"gz": Versioned("gz", [("m-1", "m-2")])})
    assert (
        "across a model change" in conditions.check(ctx.meta["detector_baseline"], None)[0]
    )  # held, not compared


class Graded:
    """Deterministic scores in (0, 1) by text; ``shift`` simulates a vendor that changed its model."""

    def __init__(self, name, shift=0.0):
        self.name, self.shift, self.calls = name, shift, []

    def score(self, texts):
        self.calls.append(list(texts))
        return [
            DetectorScore(self.name, "x", min(1.0, 0.2 + 0.1 * int(t.split()[-1]) + self.shift))
            for t in texts
        ]


def test_canaries_are_mid_range_h_texts_checked_without_the_cache(tmp_path):
    texts = _period(tmp_path)  # human text 0..3 -> scores 0.2, 0.3, 0.4, 0.5
    ctx = load_period(tmp_path, KEY, human_texts=texts, detectors={"zg": Graded("zg")},
                      watch={"zg": Graded("zg")})  # fmt: skip
    can = ctx.meta["detector_baseline"]["zg"]["canary"]
    assert can["texts"] == ["human text 3", "human text 2"] and can["scores"] == pytest.approx([0.5, 0.4])
    # H's scores came from a cache filled by an older model: the fresh canaries do not match -> refused
    from humanizer_eval.evaluation import conditions

    ctx = load_period(tmp_path, KEY, human_texts=texts, detectors={"zg": Graded("zg")},
                      watch={"zg": Graded("zg", shift=0.01)})  # fmt: skip
    assert (
        "differ from the vendor's current model" in conditions.check(ctx.meta["detector_baseline"], None)[0]
    )
    # numeric noise far below a model change is not a change
    ctx = load_period(tmp_path, KEY, human_texts=texts, detectors={"zg": Graded("zg")},
                      watch={"zg": Graded("zg", shift=4e-5)})  # fmt: skip
    assert conditions.check(ctx.meta["detector_baseline"], None) == []
