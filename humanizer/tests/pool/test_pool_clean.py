import statistics

import pytest
from pool_helpers import SENTENCES, doc_with_paragraphs, prose

from humanizer_eval.pool import clean
from humanizer_eval.pool.clean import (
    Deduplicator,
    FilterConfig,
    check_quality,
    excerpt,
    find_pii,
    is_english,
    normalize_text,
    paragraphs,
    process_document,
    scrub_pii,
    strip_boilerplate,
)


# --------------------------------------------------------------------------- normalization / boilerplate
def test_normalize_strips_invisible_and_collapses_whitespace():
    raw = "Café​  menu today\r\n\n\n\n  second\tline  \x07"
    assert normalize_text(raw) == "Café menu today\n\nsecond line"
    assert normalize_text("one\u2028two\u2029three") == "one\ntwo\n\nthree"


def test_boilerplate_lines_and_bylines_are_removed_but_prose_kept():
    text = "\n".join(
        [
            "By Jane Doe and John Smith",
            "Photo by Someone on Unsplash",
            "By then, it was over.",
            "Click here to subscribe to our newsletter",
            "Call me at 713-853-5555",
            "Get daily updates directly to your inbox + Subscribe Thank you for subscribing!",
            "× Expand Photo credit: Phil Roeder",
            "The real paragraph stays here.",
        ]
    )
    out = strip_boilerplate(text)
    assert "Jane Doe" not in out and "Unsplash" not in out and "Click here" not in out
    assert "713-853" not in out  # short contact line dropped
    assert "subscribing" not in out and "Photo credit" not in out  # web widgets
    assert "By then, it was over." in out and "The real paragraph stays here." in out


# --------------------------------------------------------------------------- language
def test_language_filter():
    assert is_english(prose(120))
    french = (
        "Le comité s'est réuni par un après-midi pluvieux pour discuter du budget de la bibliothèque. "
        "Plusieurs membres ont soutenu que la salle de lecture avait besoin d'un meilleur éclairage."
    )
    german = (
        "Der Ausschuss traf sich an einem regnerischen Nachmittag, um über das neue Budget der "
        "Bibliothek zu sprechen. Mehrere Mitglieder meinten, der Lesesaal brauche besseres Licht."
    )
    assert not is_english(french)
    assert not is_english(german)


# --------------------------------------------------------------------------- length & quality
def test_length_bounds():
    assert check_quality(prose(100)) == "too_short"
    assert check_quality(prose(400)) == "too_long"
    assert check_quality(prose(220)) is None


@pytest.mark.parametrize(
    "snippet,reason",
    [
        ("<div class='x'>hello</div>", "markup"),
        ("the formula $x^2 + y_1$ shows", "markup"),
        ("see [[Main Page]] for more", "markup"),
        ("my essay about Generic_Name and her school", "placeholder"),
        ("Dear @CAPS1, I am writing", "placeholder"),
        ("[IMAGE] full story here", "placeholder"),
        ("people don�t want it", "encoding"),
    ],
)
def test_rejects_markup_and_placeholders(snippet, reason):
    assert check_quality(prose(110) + " " + snippet + " " + prose(60, 3)) == reason


def test_code_is_rejected_but_science_and_money_are_not():
    code = prose(150) + "\n\nfor (i = 0; i < n; i++) {\nx = foo(i);\n}\n\n" + prose(40)
    assert check_quality(code) == "code"
    identifiers = prose(170) + " call get_user_name and set_value, then parseInput and readFile."
    assert check_quality(identifiers) == "code"
    science = prose(170) + " The NaCl solution raised mRNA and bFGF levels; mTOR and NaCl again."
    assert check_quality(science) is None
    money = prose(170) + " Prices rose from $4.70 to $4.80 and the fee was $6,000 in total."
    assert check_quality(money) is None


def test_caps_list_and_repetition():
    assert check_quality(prose(200).upper()) == "caps"
    listy = "\n".join(f"- item number {i} is here" for i in range(40))
    assert check_quality(listy) == "list_like"
    para = prose(80)
    assert check_quality("\n\n".join([para, para, prose(40, 2)])) == "repetitive"


# --------------------------------------------------------------------------- PII
@pytest.mark.parametrize(
    "text,kind",
    [
        ("write to john.doe@example.com today", "pii_email"),
        ("ask Tana Jones/HOU/ECT@ECT for it", "pii_email"),
        ("visit https://example.org/page now", "pii_url"),
        ("see www.example.net for details", "pii_url"),
        ("call (713)463-8626 tonight", "pii_phone"),
        ("call 713-853-5555 tonight", "pii_phone"),
        ("dial +44 20 7946 0958 please", "pii_phone"),
        ("ssn 123-45-6789 on file", "pii_ssn"),
        ("thanks @johnsmith for this", "pii_handle"),
        ("as u/some_user said", "pii_handle"),
    ],
)
def test_pii_detection(text, kind):
    assert kind in find_pii(text)


def test_no_pii_in_plain_prose():
    assert find_pii(prose(200) + " It cost $4.80 in 1999, about 15-20 minutes away.") == []


def test_scrub_pii_reject_and_mask():
    text = "Contact me at jane@example.com or 713-853-5555."
    assert scrub_pii(text, "reject") == (None, ["pii_email", "pii_phone"])
    masked, kinds = scrub_pii(text, "mask")
    assert masked == "Contact me at [EMAIL] or [PHONE]."
    assert kinds == ["pii_email", "pii_phone"]


# --------------------------------------------------------------------------- dedupe
def test_dedupe_exact_and_near_duplicates():
    d = Deduplicator()
    base = prose(230)
    assert d.check_and_add(base) is None
    assert d.check_and_add("  " + base.upper().replace(",", "") + " ") == "exact_dup"
    words = base.split()
    words[100] = "zebra"
    assert d.check_and_add(" ".join(words)) == "near_dup"
    other = " ".join(f"word{i} token{i * 7 % 13}" for i in range(120))
    assert d.check_and_add(other) is None
    assert len(d) == 2


# --------------------------------------------------------------------------- excerpts
def test_short_and_in_range_documents():
    assert excerpt(prose(100), "k") == []
    doc = doc_with_paragraphs([60, 70, 80])
    assert excerpt(doc, "k") == ["\n\n".join(paragraphs(doc))]


def test_excerpts_are_contiguous_paragraph_aligned_and_in_bounds():
    sizes = [40, 75, 55, 90, 30, 65, 80, 45, 70, 60, 85, 50, 35, 95, 40]
    doc = doc_with_paragraphs(sizes)
    paras = paragraphs(doc)
    pieces = excerpt(doc, "doc-1")
    assert len(pieces) >= 3
    pos = 0
    for piece in pieces:
        n = len(piece.split())
        assert 150 <= n <= 350
        ps = piece.split("\n\n")
        start = paras.index(ps[0], pos)
        assert paras[start : start + len(ps)] == ps  # contiguous whole paragraphs
        pos = start + len(ps)
    assert pieces[0].startswith(paras[0])  # the first fragment opens the document


def test_overlong_paragraph_is_split_at_sentence_boundaries():
    doc = prose(900)  # one paragraph, ~900 words
    pieces = excerpt(doc, "long")
    assert pieces
    for piece in pieces:
        assert 150 <= len(piece.split()) <= 350
        assert piece.rstrip()[-1] in ".!?"
        assert piece.split(". ")[0] + "." in SENTENCES or piece.split(". ")[0] in SENTENCES


def test_run_on_unit_longer_than_max_is_skipped():
    run_on = " ".join(["word"] * 400)  # no sentence boundary, 400 words
    doc = run_on + "\n\n" + doc_with_paragraphs([80, 90, 70])
    pieces = excerpt(doc, "runon")
    assert pieces and all("word word word" not in p for p in pieces)


def test_excerpt_determinism_and_target_mean():
    doc = doc_with_paragraphs([30, 45, 25, 50, 40, 35, 55, 20, 45, 30] * 4)
    assert excerpt(doc, "same") == excerpt(doc, "same")
    lengths = [len(p.split()) for k in range(150) for p in excerpt(doc, f"key{k}", limit=1)]
    assert 200 <= statistics.mean(lengths) <= 260  # target mean 230 (D-55)
    assert len(set(lengths)) >= 3  # lengths vary with the per-document target (at paragraph ends)


def test_limit_and_custom_bounds():
    cfg = FilterConfig(min_words=50, max_words=100, target_mean_words=75, target_sd_words=10)
    doc = doc_with_paragraphs([20] * 30)
    pieces = excerpt(doc, "k", cfg, limit=2)
    assert len(pieces) == 2 and all(50 <= len(p.split()) <= 100 for p in pieces)


# --------------------------------------------------------------------------- full document pipeline
def test_process_document_accepts_clean_text_and_reports_rejections():
    doc = doc_with_paragraphs([70, 80, 60, 90])
    res = process_document(doc, "a")
    assert len(res.excerpts) == 1 and res.rejected == []
    assert process_document("Hi Generic_Name, " + doc, "b").rejected == ["placeholder"]
    assert process_document("too short", "c").rejected == ["too_short"]
    assert process_document("", "d").rejected == ["empty"]
    spanish = " ".join(
        ["El comité se reunió una tarde lluviosa para hablar del presupuesto de la biblioteca."] * 20
    )
    assert process_document(spanish, "e").rejected[0] in {"not_english", "repetitive"}


def test_process_document_rejects_pii_by_default():
    doc = doc_with_paragraphs([70, 80]) + " Write to me at jane@example.com about it."
    assert process_document(doc, "x").rejected == ["pii_email"]
    masked = process_document(doc, "x", FilterConfig(pii_policy="mask"))
    assert "[EMAIL]" in masked.excerpts[0]


def test_preclean_hook_is_applied():
    doc = doc_with_paragraphs([80, 90])
    res = process_document("JUNK" + doc, "p", preclean=lambda t: t.replace("JUNK", ""))
    assert res.excerpts and not res.excerpts[0].startswith("JUNK")


def test_language_detector_is_cached():
    assert clean._lang_identifier() is clean._lang_identifier()


# --------------------------------------------------------------------------- cache window -> pool window
def test_cache_window_keeps_longest_prefix_then_fit_window_cuts_like_a_direct_build():
    cache = FilterConfig(min_words=100, max_words=500, target_mean_words=500, target_sd_words=0)
    doc = doc_with_paragraphs([30, 45, 25, 50, 40, 35, 55, 20, 45, 30] * 4)
    paras = paragraphs(doc)
    cached = excerpt(doc, "c", cache)[0]  # what the build caches for this document
    assert 450 <= len(cached.split()) <= 500 and cached.startswith(paras[0])  # longest prefix <= 500
    lengths = []
    for k in range(100):
        text, reason = clean.fit_window(cached, f"key{k}")
        assert reason == "" and 150 <= len(text.split()) <= 350 and cached.startswith(text)
        assert text == excerpt(doc, f"key{k}", limit=1)[0]  # identical to excerpting the document directly
        lengths.append(len(text.split()))
    assert 200 <= statistics.mean(lengths) <= 260


def test_fit_window_keeps_in_range_drops_short_and_rechecks_cut_pieces():
    mid = doc_with_paragraphs([100, 120])
    assert clean.fit_window(mid, "k") == (mid, "")
    assert clean.fit_window(prose(120), "k") == (None, "window_short")
    # the first piece of a long candidate carries an e-mail address: the next piece is used instead
    first = "\n\n".join(prose(90, k) for k in (0, 1)).replace("library", "jane@example.com", 1)
    long = first + "\n\n" + "\n\n".join(prose(90, k) for k in (5, 6, 7))
    text, reason = clean.fit_window(long, "k")
    assert reason == "" and "@" not in text and 150 <= len(text.split()) <= 350
    narrow = FilterConfig(min_words=150, max_words=300, target_mean_words=220)
    text, _ = clean.fit_window("\n\n".join(prose(110, k) for k in (0, 3, 6)), "k", narrow)
    assert 150 <= len(text.split()) <= 300
