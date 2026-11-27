from humanizer_eval.gates.normalize import apply_length_limit, normalize


def test_removes_invisible_and_bidi():
    s, rep = normalize("he\u200bllo\u202e wor\ufeffld\u00ad!")
    assert s == "hello world!"
    assert rep.removed_invisible >= 4


def test_homoglyphs_fixed_only_in_mixed_latin_words():
    cyrillic_words = "\u0441\u043b\u043e\u0432\u043e \u0442\u0435\u043a\u0441\u0442"  # two all-Cyrillic words
    # Cyrillic small ie (U+0435) and a (U+0430) hidden inside Latin words
    s, rep = normalize(f"Pl\u0435ase visit p\u0430ypal now. {cyrillic_words}")
    assert "Please" in s and "paypal" in s
    assert cyrillic_words in s  # genuine Cyrillic text untouched
    assert rep.homoglyph_words == 2


def test_spaces_and_newlines():
    s, _ = normalize("a\u00a0\u2009b   c\r\n\r\n\r\n\r\nd")
    assert s == "a b c\n\nd"


def test_fullwidth_and_ligatures_via_nfkc():
    s, _ = normalize("ｆｕｌｌ ﬁne")
    assert s == "full fine"


def test_arabic_tatweel_and_lookalikes():
    s, _ = normalize("كـــتاب ی ک ١٢٣", lang="ar")
    assert s == "كتاب ي ك 123"


def test_length_limit():
    src = " ".join(["w"] * 100)
    out = ("Sentence one is here. " * 40).strip()
    clipped, cut = apply_length_limit(out, src, 1.3)
    assert cut and len(clipped.split()) <= 130 and clipped.endswith(".")
    same, cut2 = apply_length_limit("short text.", src)
    assert not cut2 and same == "short text."


def test_one_word_per_line_and_huge_words_are_bounded():
    from humanizer_eval.gates.normalize import apply_length_limit

    src = "The council approved the budget. It funds buses.\n\nResidents welcomed it."
    joined, cut = apply_length_limit("\n".join(src.split()), src)
    assert cut and "\n" not in joined.replace("\n\n", "")
    big, cut = apply_length_limit("x" * 50_000, src)
    assert cut and len(big) <= max(200, int(len(src) * 1.3 * 1.25))


def test_variation_selectors_and_braille_blank_are_removed():
    from humanizer_eval.gates.normalize import normalize

    assert normalize("T\ufe0fh\u2800e c\U000e0100ity")[0] == "The city"
