import pytest

from humanizer_eval.pool.metadata import (
    DEFAULT_GENRE_PROPORTIONS,
    LENGTH_BIN_LABELS,
    PUBLIC_FIELDS,
    TextRecord,
    length_bin,
    register,
    register_score,
    text_id,
)
from humanizer_eval.pool.sources import GENRES


@pytest.mark.parametrize(
    "n,label",
    [
        (150, "150-199"),
        (199, "150-199"),
        (200, "200-249"),
        (249, "200-249"),
        (250, "250-299"),
        (300, "300-350"),
    ],
)
def test_length_bins(n, label):
    assert length_bin(n) == label
    assert length_bin(350) == "300-350"


@pytest.mark.parametrize("n", [0, 149, 351, 1000])
def test_length_bin_out_of_range(n):
    with pytest.raises(ValueError):
        length_bin(n)


def test_bins_cover_range_without_gaps():
    assert {length_bin(n) for n in range(150, 351)} == set(LENGTH_BIN_LABELS)


def test_text_id_is_stable_content_hash():
    a = text_id("Hello world.")
    assert a == text_id("Hello world.") and len(a) == 20 and int(a, 16) >= 0
    assert a != text_id("Hello world!")


def test_register_heuristic_on_clear_cases():
    informal = (
        "lol I can't believe it, honestly you guys!! I'm gonna try that place again tomorrow, "
        "it's pretty awesome and the staff were super cool :) yeah, definitely going back."
    )
    formal = (
        "The present study examines the relationship between institutional regulation and market "
        "concentration. Furthermore, the analysis demonstrates that the implementation of oversight "
        "mechanisms significantly reduces volatility; therefore, policy recommendations are discussed."
    )
    assert register(informal) == "informal"
    assert register(formal) == "formal"
    assert register_score(informal) > 0 > register_score(formal)


def test_text_record_fields_and_public_view():
    text = " ".join(["word"] * 210)
    rec = TextRecord.from_text(
        text, genre="essay", source_id="persuade2", source_doc_id="X1", date="", date_evidence="dataset"
    )
    assert rec.word_count == 210 and rec.length_bin == "200-249"
    assert rec.text_id == text_id(text)
    assert rec.register in {"formal", "informal"}
    assert "text" not in PUBLIC_FIELDS and "text_id" in PUBLIC_FIELDS
    assert set(rec.to_row()) == set(PUBLIC_FIELDS) | {"text"}


def test_length_bins_follow_the_window():
    from humanizer_eval.pool.metadata import LENGTH_BINS, make_length_bins

    assert make_length_bins(150, 350) == LENGTH_BINS
    assert [b[2] for b in make_length_bins(150, 300)] == ["150-199", "200-249", "250-300"]
    wide = make_length_bins(100, 500)
    assert wide[0][2] == "100-149" and wide[-1][2] == "450-500" and len(wide) == 8
    assert {length_bin(n, wide) for n in range(100, 501)} == {b[2] for b in wide}
    with pytest.raises(ValueError):
        length_bin(99, wide)
    with pytest.raises(ValueError):
        make_length_bins(300, 200)
    text = " ".join(["word"] * 420)
    rec = TextRecord.from_text(
        text, genre="essay", source_id="x", source_doc_id="1", date_evidence="dataset", bins=wide
    )
    assert rec.length_bin == "400-449"


def test_default_genre_proportions_cover_all_genres_but_letters():
    # Q-11 (2026-11-09): email_letter stays in the pool but is out of the mix; application too (owner
    # decision 2026-11-10: no realistic AI inputs for it); weights are relative
    assert set(DEFAULT_GENRE_PROPORTIONS) == set(GENRES) - {"email_letter", "application"}
    # 2026-11 production distribution (shares in percent / 100; the "other" bucket is ignored)
    assert DEFAULT_GENRE_PROPORTIONS["academic"] == 0.39 and DEFAULT_GENRE_PROPORTIONS["news"] == 0.007
    assert sum(DEFAULT_GENRE_PROPORTIONS.values()) == pytest.approx(0.813)


def test_period_config_genre_proportions_match_the_pool_defaults():
    import json
    from pathlib import Path

    cfg = json.loads((Path(__file__).parents[2] / "configs" / "period.json").read_text())["genre_proportions"]
    assert set(cfg) == set(DEFAULT_GENRE_PROPORTIONS)
    for g, w in DEFAULT_GENRE_PROPORTIONS.items():
        assert cfg[g] == pytest.approx(100 * w), g
