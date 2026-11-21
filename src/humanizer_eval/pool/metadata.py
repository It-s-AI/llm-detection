"""Per-text public metadata: stable id, length bin, register; genre proportions (spec Q-2).

Register heuristic (dev, deterministic, documented in docs/design/02-pool.md):

    informal = 1.5*contractions + 0.5*first/second-person pronouns + 2*'!' + 0.5*'?' + 1*'...'
             + 3*colloquial words + 3*lowercase 'i' + 3*emoticons            (all per 100 words)
    formal   = 1*nominalizations (-tion, -ment, -ity, -ness, -ance, -ence, -ism, -ology)
             + 2*academic connectors (however, therefore, moreover, ...) + 0.3*words of >= 9 letters
    score    = informal - formal;   register = "informal" if score > REGISTER_THRESHOLD else "formal"

It is a cheap proxy (e.g. polite business email lands near the threshold); the raw ``register_score`` is
stored so the threshold can be changed without a rebuild.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

LengthBins = tuple[tuple[int, int, str], ...]

#: default pool length window (D-55, Q-9); a build can choose another one (``BuildConfig.min_words``)
DEFAULT_MIN_WORDS = 150
DEFAULT_MAX_WORDS = 350
LENGTH_BIN_WIDTH = 50


def make_length_bins(min_words: int, max_words: int, width: int = LENGTH_BIN_WIDTH) -> LengthBins:
    """Bins of ``width`` words from ``min_words``; the last bin is closed at ``max_words``
    (150-350 -> 150-199, 200-249, 250-299, 300-350)."""
    if not 0 < min_words < max_words or width <= 0:
        raise ValueError(f"bad length window {min_words}-{max_words} (bin width {width})")
    starts = list(range(min_words, max_words, width))
    if len(starts) > 1 and max_words - starts[-1] < width // 2:
        starts.pop()  # a stub shorter than half a bin joins the previous bin
    ends = [s - 1 for s in starts[1:]] + [max_words]
    return tuple((lo, hi, f"{lo}-{hi}") for lo, hi in zip(starts, ends, strict=True))


LENGTH_BINS: LengthBins = make_length_bins(DEFAULT_MIN_WORDS, DEFAULT_MAX_WORDS)
LENGTH_BIN_LABELS: tuple[str, ...] = tuple(b[2] for b in LENGTH_BINS)

#: Q-2: shares of the production distribution measured by the owner (2026-11, 2000 English texts submitted to
#: the humanizer; the "other" bucket ignored). email_letter (Q-11) and application (owner decision 2026-11-10)
#: stay in the pool but are out of the mix.
#: Weights are relative (they sum to 0.813); ``configs/period.json`` holds the same numbers in percent.
DEFAULT_GENRE_PROPORTIONS: dict[str, float] = {
    "academic": 0.39,
    "essay": 0.15,
    "business_marketing": 0.07,
    "forum_qa": 0.06,
    "homework_answer": 0.06,
    "article_blog": 0.045,
    "creative": 0.022,
    "review": 0.009,
    "news": 0.007,
}

REGISTERS: tuple[str, ...] = ("formal", "informal")
REGISTER_THRESHOLD = 0.0


def word_count(text: str) -> int:
    return len(text.split())


def length_bin(n_words: int, bins: LengthBins = LENGTH_BINS) -> str:
    for lo, hi, label in bins:
        if lo <= n_words <= hi:
            return label
    raise ValueError(f"{n_words} words is outside the pool length range {bins[0][0]}-{bins[-1][1]}")


def text_id(text: str) -> str:
    """Stable content id: first 20 hex chars of sha256 over the UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


# --------------------------------------------------------------------------- register
_CONTRACTION = re.compile(
    r"\b\w+(?:n't|'m|'re|'ve|'ll|'d)\b|\b(?:it|that|there|what|let|here|he|she|who)'s\b", re.IGNORECASE
)
_FIRST_SECOND = re.compile(r"\b(?:i|me|my|mine|myself|you|your|yours|yourself|yourselves)\b", re.IGNORECASE)
_LOWER_I = re.compile(r"(?:^|\s)i(?:\s|'|$)")
_EMOTICON = re.compile(r"(?::|;|=)-?[()DPp]|\^_\^|<3|\bxD\b")
_COLLOQUIAL = frozenset(
    """
    lol lmao omg btw imo imho tbh gonna wanna gotta kinda sorta yeah yep yup nope nah ok okay hey
    dude guys stuff awesome cool totally haha hahaha wow ugh meh cuz coz ya y'all
    """.split()
)
_CONNECTORS = frozenset(
    """
    however therefore moreover furthermore thus hence whereas consequently nevertheless nonetheless
    additionally accordingly respectively notably whereby regarding
    """.split()
)
_NOMINAL = re.compile(r"\b[a-z]{3,}(?:tion|sion|ment|ity|ness|ance|ence|ism|ology)s?\b")


def register_score(text: str) -> float:
    t = text.replace("’", "'")
    words = re.findall(r"[A-Za-z']+", t)
    n = max(len(words), 1)
    lower = [w.lower() for w in words]
    per100 = 100.0 / n
    informal = (
        1.5 * len(_CONTRACTION.findall(t))
        + 0.5 * len(_FIRST_SECOND.findall(t))
        + 2.0 * t.count("!")
        + 0.5 * t.count("?")
        + 1.0 * t.count("...")
        + 3.0 * sum(w in _COLLOQUIAL for w in lower)
        + 3.0 * len(_LOWER_I.findall(t))
        + 3.0 * len(_EMOTICON.findall(t))
    ) * per100
    formal = (
        1.0 * len(_NOMINAL.findall(t.lower()))
        + 2.0 * sum(w in _CONNECTORS for w in lower)
        + 0.3 * sum(len(w) >= 9 for w in words)
    ) * per100
    return round(informal - formal, 4)


def register(text: str, threshold: float = REGISTER_THRESHOLD) -> str:
    return "informal" if register_score(text) > threshold else "formal"


# --------------------------------------------------------------------------- record
@dataclass
class TextRecord:
    text_id: str
    text: str
    genre: str
    source_id: str
    source_doc_id: str
    word_count: int
    length_bin: str
    register: str
    register_score: float
    date: str  # ISO date if the source has row-level dates, else ""
    date_evidence: str  # "row" | "dataset"
    cluster_id: str = ""
    cluster_label: str = ""

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        genre: str,
        source_id: str,
        source_doc_id: str,
        date: str = "",
        date_evidence: str,
        bins: LengthBins = LENGTH_BINS,
    ) -> TextRecord:
        n = word_count(text)
        score = register_score(text)
        return cls(
            text_id=text_id(text),
            text=text,
            genre=genre,
            source_id=source_id,
            source_doc_id=source_doc_id,
            word_count=n,
            length_bin=length_bin(n, bins),
            register="informal" if score > REGISTER_THRESHOLD else "formal",
            register_score=score,
            date=date,
            date_evidence=date_evidence,
        )

    def to_row(self) -> dict:
        return asdict(self)


RECORD_FIELDS: tuple[str, ...] = tuple(TextRecord.__dataclass_fields__)
PUBLIC_FIELDS: tuple[str, ...] = tuple(f for f in RECORD_FIELDS if f != "text")
