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
