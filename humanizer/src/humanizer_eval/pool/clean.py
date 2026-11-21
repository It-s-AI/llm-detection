"""Cleaning, quality filters, PII handling, excerpting and deduplication for the human pool.

Design choices (see docs/design/02-pool.md):

* Filters *reject* rather than repair whenever a repair would leave a trace that AI-generated inputs
  never have (e.g. ``[EMAIL]`` masks): the reference set H must differ from AI text only in authorship.
  ``pii_policy="mask"`` exists for experiments but is not the default.
* Excerpts are contiguous and paragraph-aligned (D-55). Paragraphs longer than ``max_words`` are split
  into sentences, so a fragment may start/end mid-paragraph only inside such an over-long paragraph.
* Every random choice is seeded from a stable key (source id + row index), so a rebuild is reproducible.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FilterConfig:
    min_words: int = 150
    max_words: int = 350
    target_mean_words: float = 230.0
    target_sd_words: float = 55.0
    max_excerpts_per_doc: int = 1
    lang: str = "en"
    min_lang_prob: float = 0.90
    pii_policy: Literal["reject", "mask"] = "reject"
    min_alpha_ratio: float = 0.75
    max_caps_ratio: float = 0.25
    min_type_token_ratio: float = 0.30
    near_dup_threshold: float = 0.80
    minhash_perm: int = 128
    shingle_size: int = 5


DEFAULT_FILTERS = FilterConfig()


# --------------------------------------------------------------------------- normalization
_ZERO_WIDTH = re.compile("[​‌‍⁠﻿­]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def normalize_text(text: str) -> str:
    """NFKC, drop invisible/control characters, unify newlines and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = text.replace("\u2028", "\n").replace("\u2029", "\n\n")  # Unicode line/paragraph separators
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    text = re.sub(r"[   -   　]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- PII
_EMAIL = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*|[\w.-]+(?:/[\w-]+)+@[\w-]+"
)  # incl. Lotus Notes "Name@ENRON"
_HANDLE = re.compile(r"(?<![\w@])@[A-Za-z_]\w{1,30}\b|(?<![\w/])u/[A-Za-z0-9_-]{3,}")
_URL = re.compile(
    r"(?:https?://|www\.)\S+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|org|net|edu|gov|io|co\.uk|info|biz)\b(?:/\S*)?",
    re.IGNORECASE,
)
_PHONE = re.compile(
    r"(?<![\w.])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}(?![\w])"
    r"|(?<![\w.$-])\d{3}-\d{4}(?![\w-])"  # 7-digit local number
    r"|(?<![\w.])\+\d{1,3}[\s.-]?\d{2,4}(?:[\s.-]?\d{2,4}){2,4}(?![\w])"
    r"|\b(?:ext|x)\.?\s?\d{4,5}\b"
)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "pii_email": _EMAIL,
    "pii_url": _URL,
    "pii_phone": _PHONE,
    "pii_ssn": _SSN,
    "pii_handle": _HANDLE,
}
_PII_MASKS = {
    "pii_email": "[EMAIL]",
    "pii_url": "[URL]",
    "pii_phone": "[PHONE]",
    "pii_ssn": "[ID]",
    "pii_handle": "[USER]",
}


def find_pii(text: str) -> list[str]:
    """Names of the PII patterns present in ``text`` (email checked before URL: ``a@b.com`` is an email)."""
    found = []
    rest = text
    for name, pat in PII_PATTERNS.items():
        if pat.search(rest):
            found.append(name)
            rest = pat.sub(" ", rest)
    return found


def scrub_pii(text: str, policy: Literal["reject", "mask"] = "reject") -> tuple[str | None, list[str]]:
    """``reject``: return (None, kinds) if any PII is present. ``mask``: replace with placeholders."""
    kinds = find_pii(text)
    if not kinds:
        return text, []
    if policy == "reject":
        return None, kinds
    for name in kinds:
        text = PII_PATTERNS[name].sub(_PII_MASKS[name], text)
    return text, kinds


# --------------------------------------------------------------------------- boilerplate
_BOILERPLATE_LINE = re.compile(
    r"^(?:photo|image|picture|illustration|video)\s*(?:by|credit|courtesy|source)\b"
    r"|^(?:click here|read more|subscribe|sign up|follow (?:us|me)|share (?:this|on)|advertisement"
    r"|related:|tags?:|filed under|posted (?:by|in|on)|copyright|all rights reserved|\(c\)|©"
    r"|this (?:article|story|post) (?:was originally|originally|first) (?:published|appeared)"
    r"|originally published|thanks for reading"
    r"|sent from my|get the latest|join our|download the app|see(?: also)?:|watch:|more:|recommended:)",
    re.IGNORECASE,
)
# "By Jane Doe and John Smith" -- case-sensitive on purpose ("By then, it was over." is prose)
_NAME = r"[A-Z][\w.'-]*"
_BYLINE = re.compile(rf"^By {_NAME}(?: {_NAME}){{0,3}}(?:(?:,| and| &) {_NAME}(?: {_NAME}){{0,3}})*\s*$")
# web widgets that end up inside article text; only dropped on short lines (<= 25 words)
_WIDGET_LINE = re.compile(
    r"subscrib|newsletter|\binbox\b|video will start|photo credit|\(image:|getty images|cookie policy"
    r"|privacy policy|terms of (?:use|service)|sign up for|follow us on|click to|expand photo|share this"
    r"|read more|advertisement|all rights reserved",
    re.IGNORECASE,
)


def strip_boilerplate(text: str) -> str:
    """Drop boilerplate lines and short contact/signature lines carrying URLs, emails or phone numbers."""
    kept = []
    for line in text.split("\n"):
        s = line.strip()
        if s and (_BOILERPLATE_LINE.search(s) or _BYLINE.match(s)):
            continue
        if s and word_count(s) <= 25 and _WIDGET_LINE.search(s):
            continue
        if s and word_count(s) <= 12 and find_pii(s):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


# --------------------------------------------------------------------------- quality checks
_PLACEHOLDER = re.compile(
    r"Generic_\w+|@[A-Z]+\d+|\b(?:PROPER_NAME|STUDENT_NAME|LOCATION_NAME|TEACHER_NAME|SCHOOL_NAME)\b"
    r"|\[(?:NAME|REDACTED|deleted|removed|IMAGE|image|PHOTO|LINK)\]",
)
_MARKUP = re.compile(
    r"</?[a-zA-Z][^>]{0,60}>"  # html tags
    r"|\$(?![\d\s.,])[^$\n]{1,200}\$"  # LaTeX math ("$4.70-$4.80" is money, not math)
    r"|\\(?:begin|end|frac|cite|ref|mathbf|mathrm|textit|emph|section)\b|\\\[|\\\("  # LaTeX commands
    r"|\[\[|\]\]|\{\{|\}\}"  # wiki/template markup
    r"|=20\b|=09\b|=3D"  # quoted-printable
    r"|^#{1,6} |^```",  # markdown headings / code fences
    re.MULTILINE,
)
_CODE_LINE = re.compile(
    r"[;{}]\s*$|^\s*(?:def|class|import|from|return|if|for|while|var|let|const|function|public|private"
    r"|#include|SELECT|print)\b.*[(:={]|=>|==|!=|&&|\|\||->|::",
)
# identifiers that only occur in code/tech text: snake_case, SCREAMING_CASE, lowerCamelCase, calls
# (not PascalCase or mRNA/NaCl-style tokens: those are brands, chemicals and genes in normal prose)
_IDENTIFIER = re.compile(
    r"\b[a-z]+_[a-z0-9_]+\b|\b[A-Z]+_[A-Z0-9_]+\b|\b[a-z]{2,}[A-Z][a-z]+[A-Za-z0-9]*\b"
    r"|\b\w+\(\)|\b\w+\.\w+\("
)
_LIST_LINE = re.compile(r"^\s*(?:[-*•▪◦]|\d{1,2}[.)]|[a-z][.)])\s+")


def check_quality(text: str, cfg: FilterConfig = DEFAULT_FILTERS) -> str | None:
    """Return the first failing reason code for a candidate excerpt, or None if it passes.

    Language is checked separately (``is_english``) because it is the slowest check.
    """
    n = word_count(text)
    if n < cfg.min_words:
        return "too_short"
    if n > cfg.max_words:
        return "too_long"
    if _PLACEHOLDER.search(text):
        return "placeholder"
    if "�" in text:  # mojibake ("don�t"): a decoding artifact of the crawl, not of the writer
        return "encoding"
    if _MARKUP.search(text):
        return "markup"
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if sum(1 for ln in lines if _CODE_LINE.search(ln)) >= 2:
        return "code"
    if text.count("|") >= 4 or text.count("`") >= 2:
        return "markup"
    if len(_IDENTIFIER.findall(text)) >= 4:
        return "code"
    non_space = [c for c in text if not c.isspace()]
    letters = [c for c in non_space if c.isalpha()]
    if not non_space or len(letters) / len(non_space) < cfg.min_alpha_ratio:
        return "symbols"
    if letters and sum(c.isupper() for c in letters) / len(letters) > cfg.max_caps_ratio:
        return "caps"
    if sum(1 for ln in lines if _LIST_LINE.match(ln)) >= 3:
        return "list_like"
    if len(lines) >= 6 and sum(1 for ln in lines if word_count(ln) < 6) / len(lines) > 0.5:
        return "list_like"
    paras = [p.strip().lower() for p in text.split("\n\n") if p.strip()]
    if len(paras) != len(set(paras)):
        return "repetitive"
    words = re.findall(r"[a-z']+", text.lower())
    if words and len(set(words)) / len(words) < cfg.min_type_token_ratio:
        return "repetitive"
    return None


# --------------------------------------------------------------------------- language id
@lru_cache(maxsize=1)
def _lang_identifier():
    from langid.langid import LanguageIdentifier, model

    return LanguageIdentifier.from_modelstring(model, norm_probs=True)


def detect_language(text: str) -> tuple[str, float]:
    lang, prob = _lang_identifier().classify(text[:3000])
    return str(lang), float(prob)


def is_english(text: str, cfg: FilterConfig = DEFAULT_FILTERS) -> bool:
    lang, prob = detect_language(text)
    return lang == cfg.lang and prob >= cfg.min_lang_prob


# --------------------------------------------------------------------------- paragraphs & excerpts
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")


def paragraphs(text: str) -> list[str]:
    """Blank lines separate paragraphs when present (single newlines are wrapping); otherwise every
    non-empty line is a paragraph (news-style one-paragraph-per-line text)."""
    text = text.strip()
    if not text:
        return []
    if re.search(r"\n\s*\n", text):
        parts = re.split(r"\n\s*\n", text)
        parts = [re.sub(r"\s*\n\s*", " ", p) for p in parts]
    else:
        parts = text.split("\n")
    return [p.strip() for p in parts if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(paragraph) if s.strip()]


def stable_int(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class _Unit:
    text: str
    words: int
    para: int  # paragraph index: units of one over-long paragraph share it


def _units(paras: list[str], max_words: int) -> list[_Unit]:
    units: list[_Unit] = []
    for i, p in enumerate(paras):
        w = word_count(p)
        if w <= max_words:
            units.append(_Unit(p, w, i))
            continue
        for s in split_sentences(p):
            units.append(_Unit(s, word_count(s), i))
    return units


def _join(units: list[_Unit]) -> str:
    out: list[str] = []
    prev = None
    for u in units:
        if prev is None:
            out.append(u.text)
        elif u.para == prev:
            out.append(" " + u.text)
        else:
            out.append("\n\n" + u.text)
        prev = u.para
    return "".join(out)


def excerpt(text: str, key: str, cfg: FilterConfig = DEFAULT_FILTERS, limit: int | None = None) -> list[str]:
    """Split a document into contiguous fragments of ``min_words..max_words`` words.

    * A document already within bounds is returned whole.
    * Otherwise fragments are built from the start of the document: units (paragraphs, or sentences of
      over-long paragraphs) are added until the fragment is closest to a per-fragment target drawn from
      N(target_mean, target_sd) clipped to the bounds. A unit longer than ``max_words`` (a run-on
      "sentence") can never be part of a fragment and acts as a boundary.
    * ``limit`` caps the number of fragments (default: all).
    """
    paras = paragraphs(text)
    total = sum(word_count(p) for p in paras)
    if total < cfg.min_words:
        return []
    if total <= cfg.max_words:
        return ["\n\n".join(paras)]
    units = _units(paras, cfg.max_words)
    rng = np.random.default_rng(stable_int(key))
    out: list[str] = []
    i = 0
    while i < len(units) and (limit is None or len(out) < limit):
        target = float(
            np.clip(rng.normal(cfg.target_mean_words, cfg.target_sd_words), cfg.min_words, cfg.max_words)
        )
        acc: list[_Unit] = []
        n = 0
        j = i
        while j < len(units):
            w = units[j].words
            if w > cfg.max_words or n + w > cfg.max_words:
                break
            if n >= cfg.min_words and abs(n + w - target) >= abs(n - target):
                break
            acc.append(units[j])
            n += w
            j += 1
        if n >= cfg.min_words:
            out.append(_join(acc))
            i = j
        else:
            # could not reach the minimum before a boundary: restart after the blocking unit
            i = max(j, i) + 1 if j < len(units) and units[j].words > cfg.max_words else i + 1
    return out


def fit_window(text: str, key: str, cfg: FilterConfig = DEFAULT_FILTERS) -> tuple[str | None, str]:
    """Fit a cached candidate (collected in a wider window) to the pool window ``cfg.min_words..max_words``.

    Returns ``(text, "")`` or ``(None, reason)``. A candidate inside the window is kept whole; a shorter one
    is dropped (``window_short``); a longer one is cut with ``excerpt`` -- the same contiguous,
    paragraph-aligned rule and length target as a direct build -- and the first piece that passes the
    per-text checks again (quality, PII) is kept. Language is not re-checked: the piece is a contiguous
    part of a text that already passed the language filter.
    """
    n = word_count(text)
    if n < cfg.min_words:
        return None, "window_short"
    if n <= cfg.max_words:
        return text, ""
    reasons: list[str] = []
    for piece in excerpt(text, key, cfg):
        pii = find_pii(piece)
        reason = check_quality(piece, cfg) or (pii[0] if pii else None)
        if reason is None:
            return piece, ""
        reasons.append(reason)
    return None, f"window_{reasons[0]}" if reasons else "window_cut"


# --------------------------------------------------------------------------- deduplication
def _dedupe_key(text: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _shingles(text: str, k: int) -> set[str]:
    toks = re.findall(r"[a-z0-9']+", text.lower())
    if len(toks) < k:
        return {" ".join(toks)}
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


class Deduplicator:
    """Exact (normalized hash) + near-duplicate (MinHash LSH over word shingles) filter.

    ``check_and_add`` is order-dependent by design: the first occurrence wins, so feed candidates in a
    deterministic order.
    """

    def __init__(self, cfg: FilterConfig = DEFAULT_FILTERS):
        from datasketch import MinHashLSH

        self.cfg = cfg
        self._exact: set[str] = set()
        self._lsh = MinHashLSH(threshold=cfg.near_dup_threshold, num_perm=cfg.minhash_perm)
        self._n = 0

    def minhash(self, text: str):
        from datasketch import MinHash

        m = MinHash(num_perm=self.cfg.minhash_perm, seed=1)
        m.update_batch([s.encode("utf-8") for s in _shingles(text, self.cfg.shingle_size)])
        return m

    def check_and_add(self, text: str) -> str | None:
        """Return ``"exact_dup"`` / ``"near_dup"`` if ``text`` duplicates an earlier one, else add it."""
        key = _dedupe_key(text)
        if key in self._exact:
            return "exact_dup"
        m = self.minhash(text)
        if self._lsh.query(m):
            return "near_dup"
        self._exact.add(key)
        self._lsh.insert(f"t{self._n}", m)
        self._n += 1
        return None

    def __len__(self) -> int:
        return self._n


# --------------------------------------------------------------------------- document pipeline
@dataclass
class DocResult:
    excerpts: list[str]
    rejected: list[str]  # reason codes (one per rejected document or excerpt)


def process_document(
    raw: str,
    key: str,
    cfg: FilterConfig = DEFAULT_FILTERS,
    preclean=None,
) -> DocResult:
    """Clean one document and return accepted excerpts (at most ``cfg.max_excerpts_per_doc``)."""
    text = preclean(raw) if preclean else raw
    text = strip_boilerplate(normalize_text(text or ""))
    if not text:
        return DocResult([], ["empty"])
    if _PLACEHOLDER.search(text):
        return DocResult([], ["placeholder"])
    pieces = excerpt(text, key, cfg)
    if not pieces:
        return DocResult([], ["too_short"])
    accepted: list[str] = []
    rejected: list[str] = []
    for piece in pieces:
        if len(accepted) >= cfg.max_excerpts_per_doc:
            break
        reason = check_quality(piece, cfg)
        if reason is None:
            scrubbed, kinds = scrub_pii(piece, cfg.pii_policy)
            if scrubbed is None:
                reason = kinds[0]
            else:
                piece = scrubbed
        if reason is None and not is_english(piece, cfg):
            reason = "not_english"
        if reason is None:
            accepted.append(piece)
        else:
            rejected.append(reason)
    return DocResult(accepted, rejected)
