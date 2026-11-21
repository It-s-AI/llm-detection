"""G0: text normalization as a *transformation* (D-40) + length limit.

Everything downstream (detectors, judges, the product) sees the normalized text,
so invisible-character / homoglyph tricks simply stop having any effect.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# Character classes are built from explicit code points so that no formatter or
# editor can silently turn escapes into invisible literal characters in this file.
def _cls(*items: int | range) -> str:
    cps: list[int] = []
    for it in items:
        cps.extend(it if isinstance(it, range) else [it])
    return "[" + "".join(re.escape(chr(c)) for c in cps) + "]"


# zero-width, joiners, word joiner, invisible operators, BOM, soft hyphen, bidi controls
_INVISIBLE = re.compile(
    _cls(
        0x00AD,
        0x034F,
        0x061C,
        0x115F,
        0x1160,
        0x17B4,
        0x17B5,
        0x180E,
        range(0x200B, 0x2010),
        range(0x202A, 0x202F),
        range(0x2060, 0x2065),
        range(0x2066, 0x2070),
        0x3164,
        0xFEFF,
        0xFFA0,
    )
)
_SPACES = re.compile(_cls(0x20, 0x09, 0x00A0, 0x1680, range(0x2000, 0x200B), 0x202F, 0x205F, 0x3000) + "+")
_MANY_NEWLINES = re.compile(r"\n{3,}")

# Cyrillic / Greek letters that render like Latin ones (applied only to Latin-script words).
_HOMOGLYPHS = str.maketrans(
    {
        "\u0430": "a",  # Cyrillic Small A
        "\u0435": "e",  # Cyrillic Small Ie
        "\u043e": "o",  # Cyrillic Small O
        "\u0440": "p",  # Cyrillic Small Er
        "\u0441": "c",  # Cyrillic Small Es
        "\u0443": "y",  # Cyrillic Small U
        "\u0445": "x",  # Cyrillic Small Ha
        "\u0456": "i",  # Cyrillic Small Byelorussian-Ukrainian I
        "\u0458": "j",  # Cyrillic Small Je
        "\u0455": "s",  # Cyrillic Small Dze
        "\u0501": "d",  # Cyrillic Small Komi De
        "\u0261": "g",  # Latin Small Script G
        "\u04bb": "h",  # Cyrillic Small Shha
        "\u04cf": "l",  # Cyrillic Small Palochka
        "\u051b": "q",  # Cyrillic Small Qa
        "\u051d": "w",  # Cyrillic Small We
        "\u0410": "A",  # Cyrillic Capital A
        "\u0412": "B",  # Cyrillic Capital Ve
        "\u0415": "E",  # Cyrillic Capital Ie
        "\u041a": "K",  # Cyrillic Capital Ka
        "\u041c": "M",  # Cyrillic Capital Em
        "\u041d": "H",  # Cyrillic Capital En
        "\u041e": "O",  # Cyrillic Capital O
        "\u0420": "P",  # Cyrillic Capital Er
        "\u0421": "C",  # Cyrillic Capital Es
        "\u0422": "T",  # Cyrillic Capital Te
        "\u0425": "X",  # Cyrillic Capital Ha
        "\u0423": "Y",  # Cyrillic Capital U
        "\u0406": "I",  # Cyrillic Capital Byelorussian-Ukrainian I
        "\u0408": "J",  # Cyrillic Capital Je
        "\u0405": "S",  # Cyrillic Capital Dze
        "\u03bf": "o",  # Greek Small Omicron
        "\u03b1": "a",  # Greek Small Alpha
        "\u03bd": "v",  # Greek Small Nu
        "\u03c1": "p",  # Greek Small Rho
        "\u03c4": "t",  # Greek Small Tau
        "\u03b9": "i",  # Greek Small Iota
        "\u03ba": "k",  # Greek Small Kappa
        "\u03c5": "u",  # Greek Small Upsilon
        "\u0391": "A",  # Greek Capital Alpha
        "\u0392": "B",  # Greek Capital Beta
        "\u0395": "E",  # Greek Capital Epsilon
        "\u0396": "Z",  # Greek Capital Zeta
        "\u0397": "H",  # Greek Capital Eta
        "\u0399": "I",  # Greek Capital Iota
        "\u039a": "K",  # Greek Capital Kappa
        "\u039c": "M",  # Greek Capital Mu
        "\u039d": "N",  # Greek Capital Nu
        "\u039f": "O",  # Greek Capital Omicron
        "\u03a1": "P",  # Greek Capital Rho
        "\u03a4": "T",  # Greek Capital Tau
        "\u03a5": "Y",  # Greek Capital Upsilon
        "\u03a7": "X",  # Greek Capital Chi
    }
)
_LATIN = re.compile(r"[A-Za-z]")
_NONLATIN_LETTER = re.compile(_cls(range(0x0370, 0x0400), range(0x0400, 0x0500), range(0x0500, 0x0530)))

# Arabic: tatweel, harakat/diacritics, Persian/Urdu look-alikes -> Arabic letters.
_AR_TATWEEL = chr(0x0640)
_AR_DIACRITICS = re.compile(_cls(range(0x0610, 0x061B), range(0x064B, 0x0660), 0x0670, range(0x06D6, 0x06EE)))
_AR_LOOKALIKES = str.maketrans({"ی": "ي", "ک": "ك", "ۀ": "ه", "ە": "ه", "ھ": "ه", "ي": "ي"})
_AR_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


@dataclass
class NormalizationReport:
    changed_chars: int = 0
    removed_invisible: int = 0
    homoglyph_words: int = 0
    notes: list[str] = field(default_factory=list)


def _fix_homoglyph_word(word: str) -> str:
    if _LATIN.search(word) and _NONLATIN_LETTER.search(word):
        return word.translate(_HOMOGLYPHS)
    return word


def normalize(text: str, lang: str = "en") -> tuple[str, NormalizationReport]:
    rep = NormalizationReport()
    original = text
    s = unicodedata.normalize("NFKC", text)
    s, rep.removed_invisible = _INVISIBLE.subn("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # remaining format characters (category Cf) other than newlines/tabs
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    if lang == "ar":
        s = s.replace(_AR_TATWEEL, "")
        s = _AR_DIACRITICS.sub("", s)
        s = s.translate(_AR_LOOKALIKES).translate(_AR_INDIC_DIGITS)
    else:
        words = re.split(r"(\s+)", s)
        fixed = [_fix_homoglyph_word(w) for w in words]
        rep.homoglyph_words = sum(1 for a, b in zip(words, fixed, strict=True) if a != b)
        s = "".join(fixed)
    lines = [_SPACES.sub(" ", ln).strip() for ln in s.split("\n")]
    s = _MANY_NEWLINES.sub("\n\n", "\n".join(lines)).strip()
    rep.changed_chars = sum(1 for a, b in zip(original, s, strict=False) if a != b) + abs(
        len(original) - len(s)
    )
    return s, rep


_SENT_END = re.compile(r"(?<=[.!?…])\s+")


def apply_length_limit(output: str, input_text: str, max_ratio: float = 1.3) -> tuple[str, bool]:
    """Truncate `output` to at most `max_ratio` x the input's word count, preferring a sentence end.
    Returns (text, truncated)."""
    limit = max(1, int(len(input_text.split()) * max_ratio))
    words = output.split()
    if len(words) <= limit:
        return output, False
    clipped = " ".join(words[:limit])
    parts = _SENT_END.split(clipped)
    if len(parts) > 1 and len(" ".join(parts[:-1]).split()) >= 0.8 * limit:
        clipped = " ".join(parts[:-1])
    return clipped, True
