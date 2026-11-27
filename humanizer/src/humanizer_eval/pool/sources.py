"""Declarative registry of human-text corpora (D-50).

Every source must carry *evidence* that its texts were published before 2022-01-01:

* ``date_field`` set  -> a per-row timestamp; rows without a parseable date, or dated on/after the
  cutoff, are dropped (strict);
* ``date_field`` None -> dataset-level evidence (corpus release/collection date) in ``date_evidence``.

Only repositories that load with ``datasets>=4`` (parquet/csv/jsonl, no loading scripts) are listed in
``SOURCES``; a repo whose script is unsupported but which also ships its raw files can be read with
``builder="json"`` (``webis/args_me``). Everything we tried and did not use is in ``REJECTED`` with the
reason, so the report and the code stay in sync.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime

GENRES: tuple[str, ...] = (
    "essay",
    "article_blog",
    "academic",
    "news",
    "email_letter",
    "review",
    "forum_qa",
    "creative",
    # production categories measured in 2026-11 (D-55, Q-2)
    "application",
    "business_marketing",
    "homework_answer",
)

CUTOFF = datetime(2022, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- date parsing
def parse_date(value: object) -> datetime | None:
    """Best-effort parse of the timestamp formats found in our sources; None if unparseable."""
    if value is None or value == "":
        return None
    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif hasattr(value, "to_pydatetime"):  # pandas.Timestamp
        dt = value.to_pydatetime()
    elif isinstance(value, int | float) or (isinstance(value, str) and value.strip().isdigit()):
        try:
            dt = datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    else:
        s = str(value).strip()
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%d,%B,%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d %B %Y", "%B %d, %Y"):
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except ValueError:
                    continue
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


_ARXIV_ID = re.compile(r"^(?:[a-z][a-z.\-]*/)?(\d{2})(\d{2})", re.IGNORECASE)


def arxiv_id_date(value: object) -> datetime | None:
    """arXiv ids encode the submission month: ``0704.0001`` -> 2007-04, ``hep-th/9901001`` -> 1999-01."""
    m = _ARXIV_ID.match(str(value or "").strip())
    if not m:
        return None
    yy, mm = int(m.group(1)), int(m.group(2))
    if not 1 <= mm <= 12:
        return None
    year = 1900 + yy if yy >= 91 else 2000 + yy
    return datetime(year, mm, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- per-source cleaning
_ENRON_CUT = re.compile(
    r"^[ \t>]*(?:-{2,}\s*Original Message\s*-{2,}|-{3,}\s*Forwarded by"
    r"|-{5,}|_{5,}|\*{5,}|={5,}|/{5,}|~{5,}|#{5,}"  # separator lines start quoted/forwarded material
    r"|From:\s.*\n[ \t>]*(?:Sent|To|Date):|.{0,80}\bon \d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}.*\n"
    r"[ \t]*To:)",
    re.MULTILINE | re.IGNORECASE,
)
_HEADER_LINE = re.compile(r"^[ \t]*(?:To|From|cc|bcc|Subject|Sent|Date|Re|Fw|Fwd):.*$", re.M | re.I)


def clean_enron(text: str) -> str:
    """Keep only the newly written part of an email: cut at the first quoted/forwarded block."""
    m = _ENRON_CUT.search(text)
    if m:
        text = text[: m.start()]
    text = _HEADER_LINE.sub("", text)
    lines = [ln for ln in text.split("\n") if not ln.lstrip().startswith(">")]
    return "\n".join(lines)


_MD_LINK = re.compile(r"\[([^\]]{1,200})\]\([^)]{1,500}\)")
_MD_EMPH = re.compile(r"(\*\*|__|~~|\*|`)")


def clean_reddit(text: str) -> str:
    """Reddit markdown -> plain text: unescape entities, drop quotes, unwrap links/emphasis."""
    text = html.unescape(text).replace("&#x200B;", "")
    lines = [ln for ln in text.split("\n") if not ln.lstrip().startswith(">")]
    text = "\n".join(lines)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_EMPH.sub("", text)
    return text


_HTML_TAG = re.compile(r"<[^>]{1,200}>")


def clean_html_comment(text: str) -> str:
    """Hacker News comments: <p> paragraph separators and HTML entities."""
    text = re.sub(r"<p>", "\n\n", text, flags=re.I)
    text = html.unescape(_HTML_TAG.sub("", text))
    # "> quoted" lines are someone else's words (often copied from the linked article)
    return "\n".join(ln for ln in text.split("\n") if not ln.lstrip().startswith(">"))


def clean_imdb(text: str) -> str:
    return re.sub(r"<br\s*/?>", "\n", text, flags=re.I)


def clean_blog(text: str) -> str:
    """Blog Authorship Corpus: ``urlLink`` tokens replace links; ``nbsp`` leftovers."""
    text = re.sub(r"\burlLink\b", "", text)
    return text.replace("&nbsp;", " ").replace("nbsp;", " ")


def unwrap_lines(text: str) -> str:
    """Join hard-wrapped lines (arXiv abstracts) into one paragraph."""
    return re.sub(r"\s*\n\s*", " ", text.strip())


def drop_first_line(text: str) -> str:
    """Pile PubMed abstracts start with the paper title on its own line."""
    parts = text.strip().split("\n", 1)
    return parts[1] if len(parts) == 2 else text


_DAILYMAIL_HEADER = re.compile(r"\b(?:PUBLISHED|UPDATED):\s*\.", re.I)


def clean_cnn_dm(text: str) -> str:
    """CNN/DailyMail: Daily Mail articles carry a byline/timestamp header; drop those documents."""
    if _DAILYMAIL_HEADER.search(text[:600]) or text.lstrip().startswith("By ."):
        return ""
    text = re.sub(r"^.{0,80}?\(CNN\)\s*--\s*", "", text.strip())
    # inline teasers such as "Watch a driver describe his narrow escape » ."
    text = re.sub(r"(?:^|(?<=[.!?\"]\s))[^.!?»]{0,160}»\s*\.?\s*", "", text)
    return re.sub(r"\bE-mail to a friend\b\.?", "", text)


# --------------------------------------------------------------------------- row -> documents
def stackexchange_posts(row: dict) -> list[str]:
    """Pile StackExchange rows are whole threads ``Q:\\n\\n...\\n\\nA:\\n\\n...``; return each post separately
    so one excerpt never mixes authors."""
    text = row.get("text") or ""
    posts = re.split(r"\n\nA:\n\n", text)
    if posts and posts[0].startswith("Q:\n\n"):
        posts[0] = posts[0][4:]
    return [p for p in (p.strip() for p in posts) if p]


def enron_internal_sender(row: dict) -> str | None:
    """Keep mail written by Enron staff; external senders are mostly newsletters, ads and news feeds."""
    sender = str(row.get("from") or "").strip().lower()
    return None if sender.endswith("@enron.com") else "external_sender"


def medium_documents(row: dict) -> list[str]:
    """Medium articles repeat the title as the first line."""
    text = (row.get("text") or "").strip()
    title = (row.get("title") or "").strip()
    if title and text.startswith(title):
        text = text[len(title) :].lstrip()
    return [text]


def reddit_documents(row: dict) -> list[str]:
    body = row.get("body") or ""
    if body.strip() in {"[deleted]", "[removed]", ""} or row.get("author") == "AutoModerator":
        return []
    return [body]


def writingprompts_story(row: dict) -> list[str]:
    """r/WritingPrompts: stories are the top-level comments (parent is the prompt submission ``t3_``)."""
    if not str(row.get("parent_id") or "").startswith("t3_"):
        return []
    return reddit_documents(row)


# --------------------------------------------------------------------------- essay: r/changemyview
# a separator line (CMV moderator footnote follows it) or an "EDIT:"/"UPDATE:" paragraph: replies to
# commenters (deltas awarded, clarifications) appended after the essay itself
_CMV_CUT = re.compile(
    r"^[ \t]*(?:_{3,}|-{3,}|\*{3,}|={3,})[ \t]*$|^\W{0,3}(?:edit|update|eta)\b", re.M | re.I
)


def cmv_documents(row: dict) -> list[str]:
    """r/changemyview submissions: the post body (``selftext``) is the essay; the title is the thesis."""
    return reddit_documents({"body": row.get("selftext"), "author": row.get("author")})


def clean_cmv(text: str) -> str:
    """Cut at the first edit/separator (edits placed on top are dropped), then Reddit markdown -> text.

    The cut runs on the raw markdown: ``clean_reddit`` would turn a ``_____`` rule into a stray ``_``."""
    text = text.strip()
    m = _CMV_CUT.search(text)
    while m is not None and m.start() == 0:
        text = text.split("\n\n", 1)[1].lstrip() if "\n\n" in text else ""
        m = _CMV_CUT.search(text)
    return clean_reddit(text[: m.start()] if m else text)


# --------------------------------------------------------------------------- essay: args.me
_ARGSME_CURATED = ("idebate.org", "debatewise.org", "debatepedia.org")


def argsme_portal(row: dict) -> str:
    ctx = row.get("context") if isinstance(row.get("context"), dict) else {}
    m = re.search(r"^https?://(?:www\.)?([^/]+)", str(ctx.get("sourceUrl") or ""))
    return m.group(1).lower() if m else ""


def argsme_curated_portal(row: dict) -> str | None:
    """Keep the edited debate portals; debate.org rounds quote the opponent inline (mixed authorship)."""
    portal = argsme_portal(row)
    if portal == "debate.org":
        return "debate_org_round"
    return None if portal in _ARGSME_CURATED else "unknown_portal"


def argsme_documents(row: dict) -> list[str]:
    """One argument = one premise text (the conclusion is only the claim/title)."""
    for p in row.get("premises") or []:
        if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
            return [p["text"]]
    return []


def argsme_acquired(value: object) -> datetime | None:
    """args.me ``context.acquisitionTime``: when the portal page was crawled (Apr 2019) -- an upper bound
    for the writing date."""
    return parse_date(value.get("acquisitionTime")) if isinstance(value, dict) else None


_REF_MARK = re.compile(r"\s?\[\d{1,2}\]")


def clean_argsme(text: str) -> str:
    """Undo HTML-stripping artifacts and drop the reference list.

    * ``don"t`` -> ``don't`` (apostrophes stored as double quotes);
    * ``end.Next`` -> paragraph break (paragraphs were concatenated without a separator);
    * the second ``[1]`` starts the reference list: cut there, then remove in-text ``[n]`` markers.
    """
    text = re.sub(r"(?<=[A-Za-z])\"(?=(?:s|t|re|ve|ll|d|m)\b)", "'", text)
    text = re.sub(r"(?<=[a-z]{2}[.!?])(?=[A-Z][a-z])", "\n\n", text)
    marks = [m.start() for m in re.finditer(r"\[1\]", text)]
    if len(marks) >= 2:
        text = text[: marks[1]]
    return _REF_MARK.sub("", text)


# --------------------------------------------------------------------------- C4 realnewslike: op-eds, letters
_LETTER_URL = re.compile(
    r"letters?-to-(?:the-)?editor|/letters?/|/letters?-?to-?editor|/readers?-?letters?/|/your-?letters?/"
    r"|/letters?-",
    re.I,
)
_OPINION_URL = re.compile(r"/opinions?/|/op-?eds?/|/commentary/|/editorials?/", re.I)
_INTERVIEW_URL = re.compile(r"/(?:q-?a|q-and-a|interviews?)/|-q-?and-?a-|-interview-", re.I)  # two voices
# section listings and daily round-ups put several authors on one page
_LISTING_URL = re.compile(
    r"\?|/(?:category|categories|tag|tags|section|topic|author|page)/"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|letters?-to-(?:the-)?editor(?:-\d+)*/?$|/letters?/?$|letters?-(?:for|from)-(?:\w+-)?\d"
    r"|readers?-(?:letters|write|respond)|letters?-(?:\d{1,2}-){1,2}\d{2,4}",
    re.I,
)
_LETTER_MAX_WORDS = 350  # a single letter fits the excerpt window; longer pages are usually round-ups


def c4_letter_page(row: dict) -> str | None:
    """Letters to the editor: letter URL, not a listing/round-up, and the whole page is one letter."""
    url = str(row.get("url") or "")
    if not _LETTER_URL.search(url):
        return "not_letter_url"
    if _LISTING_URL.search(url):
        return "listing_url"
    if len(str(row.get("text") or "").split()) > _LETTER_MAX_WORDS:
        return "page_too_long"
    return None


def c4_opinion_page(row: dict) -> str | None:
    """Op-eds/commentary/editorials (letters go to email_letter, listings are dropped)."""
    url = str(row.get("url") or "")
    if not _OPINION_URL.search(url):
        return "not_opinion_url"
    if _LETTER_URL.search(url):
        return "letter_url"
    if _LISTING_URL.search(url):
        return "listing_url"
    if _INTERVIEW_URL.search(url):
        return "interview_url"
    return None


_WEATHER_LINE = re.compile(
    r"\bchance of (?:rain|snow|showers|precipitation)\b|\bwinds? (?:[nsew]{1,3}|light and variable)\b"
    r"|\b(?:high|low) (?:of |near |around )?-?\d{1,3}\s?f\b",
    re.I,
)
# section headline kept by C4 because it ends in punctuation ("Letter: Are we truly free?") and print
# credits ("A version of this article appeared in the print edition of ...")
_KICKER_LINE = re.compile(
    r"^(?:letters?|op-?ed|opinion|editorial|commentary|guest (?:column|opinion)|viewpoint|(?:your|my) view)"
    r"\s*[:|–—-]"
    r"|^(?:a version of this (?:article|story|column|op-?ed|editorial|piece) (?:appeared|was published))"
    r"|^this (?:article|story|column|piece) (?:appeared|was published|first appeared)",
    re.I,
)
# desk-written standfirst summarizing the piece ("Writer feels that ...", "A Mesquite resident says ...")
_DEK_LINE = re.compile(
    r"\b(?:writer|reader|author|columnist|resident|contributor)s?\b.{0,120}?"
    r"\b(?:says?|writes?|argues?|feels?|believes?|takes?|thinks?|contends?|urges?|explains?|wonders?)\b"
    r"|\b(?:writes?|says) the (?:author|writer)\b",
    re.I,
)
_SIGNATURE_LINE = re.compile(r"^[A-Z][\w.'-]+(?: [A-Z][\w.'-]+){0,3},(?: [A-Z][\w.'-]+){1,3}\.?$")
_BIO_LINE = re.compile(
    r"^(?:(?:The )?(?:writer|author|columnist)s?|[A-Z][\w.'-]+(?: [A-Z][\w.'-]+){1,3})"
    r" (?:is|are|was|lives|teaches|works|writes|serves|served|chairs|directs|represents)\b"
)


def clean_c4_news(text: str) -> str:
    """C4 op-ed/letter pages -> the author's own text: drop weather widgets and section kickers, a
    desk-written standfirst, and trailing bio/signature lines (``Jane Doe, Springfield.``,
    ``The writer is a retired teacher.``). C4 already removed lines without end punctuation; some
    pages still carry HTML entities (``&apos;``), which are decoded."""
    text = html.unescape(text)
    lines = [
        ln.strip()
        for ln in text.split("\n")
        if ln.strip() and not _WEATHER_LINE.search(ln) and not _KICKER_LINE.match(ln.strip())
    ]
    if lines and len(lines[0].split()) <= 40 and _DEK_LINE.search(lines[0]):
        lines = lines[1:]
    while (
        lines
        and len(lines[-1].split()) <= 40
        and (_SIGNATURE_LINE.match(lines[-1]) or _BIO_LINE.match(lines[-1]))
    ):
        lines.pop()
    return "\n\n".join(lines)


C4_REALNEWS_FILES: tuple[str, ...] = tuple(
    f"realnewslike/c4-train.{i:05d}-of-00512.json.gz" for i in range(512)
)


# --------------------------------------------------------------------------- academic: paper body paragraphs
# Production academic texts are mostly 1-3 paragraphs from the body of a paper (introduction, literature
# review, methods, discussion), not abstracts. A paper is parsed into a sequence of blocks: a cleaned body
# paragraph, or ``None`` for anything that is not body text (a heading, a caption, a table, an equation, a
# paragraph that failed the checks, a section that is not intro/methods/discussion). A paper row yields a
# few *windows*: 1-3 consecutive body paragraphs starting at a seeded random body paragraph, holding as many
# paragraphs (<= 3) as fit the cache window; ``clean.fit_window`` later cuts a window to the pool window at a
# paragraph boundary (a paragraph is never cut: longer ones are not body blocks).
PAPER_MAX_PARAS = 3
PAPER_PARA_MIN_WORDS = 25
PAPER_PARA_MAX_WORDS = 350
PAPER_WINDOW_MIN_WORDS = 150
PAPER_WINDOW_MAX_WORDS = 500
PAPER_WINDOWS_PER_DOC = 4

_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "back",
        r"reference|bibliograph|literature cited|acknowledg|funding|financial (?:support|disclosure)|grant"
        r"|conflicts? of interest|competing interest|declaration|disclosure|authors?\W? contribution"
        r"|contributions? of authors|ethic|consent|data (?:availability|access|sharing)|availability of"
        r"|supplementar|supporting information|additional (?:information|file)|abbreviation|appendix"
        r"|publisher.?s note|footnote|^notes?$|note added|electronic supplementary|author information"
        r"|figure legends?|table legends?|^figures?$|^tables?$|biograph|about the author|orcid|copyright"
        r"|registration|keywords?|highlights|^abstract|summary$|graphical abstract|change history"
        r"|correction|erratum|open access|peer review|pre-?publication|provenance",
    ),
    (
        "results",
        r"^results?$|^results? (?:and|&) (?:analysis|findings)|^findings$"
        r"|^(?:main )?results\b(?!.*discussion)"
        r"|^experimental results|^evaluation|^experiments?$|^numerical|^simulation results"
        r"|^case (?:report|presentation|description)s?$",
    ),
    (
        "intro",
        r"introduc|background|literature|related work|previous (?:work|studies|research)|state of the art"
        r"|theor(?:y|etical)|conceptual|framework|overview|motivation|rationale|context|history|^aims?\b"
        r"|objective|purpose",
    ),
    (
        "methods",
        r"method|material|patients|participants|subjects|study (?:design|population|area|setting|site)"
        r"|procedure|design|sample|setting|data (?:collection|source|analysis)|measure|instrument|statistic"
        r"|analysis|protocol|intervention|recruit|survey|questionnaire|approach|experimental",
    ),
    (
        "discussion",
        r"discussion|conclu|implication|limitation|strength|future (?:work|research|direction)"
        r"|interpretation|recommendation|summary and|concluding|outlook|perspective|remarks",
    ),
)
_ROLE_RE = tuple((role, re.compile(p, re.I)) for role, p in _ROLE_PATTERNS)
_BODY_ROLES = frozenset({"intro", "methods", "discussion", "body"})


def heading_role(heading: str) -> str | None:
    """Section role of a heading (back/results/intro/methods/discussion), None if unknown."""
    h = re.sub(r"^[\s#*]*(?:[0-9IVXivx]+[.)]?(?:\d+[.)]?)*|[A-H][.)])\s+", "", heading.strip())
    h = h.strip(" :.*").lower()
    for role, rx in _ROLE_RE:
        if rx.search(h):
            return role
    return None


# a citation used as a noun ("as shown in [12]", "see [@smith]") cannot be deleted without breaking the
# sentence; such paragraphs are skipped rather than repaired
_CITE_LEAD = (
    r"(?:\b(?:in|by|see|from|of|to|with|on|as|than|and|or|cf\.?|refs?\.?|references?|studies|study|works?)"
    r"\s*)"
)
_NARRATIVE_CITE_PANDOC = re.compile(_CITE_LEAD + r"(?:\\?\[\[@|\(\[@|\[@|\^?\[\d[^\]]{0,30}\]\(#)", re.I)
_NARRATIVE_CITE_GROBID = re.compile(_CITE_LEAD + r"\[\s*\d", re.I)
_AUTHOR_YEAR_PANDOC = re.compile(r"(?:et al\.?|\b[A-Z][a-z]+)\s*\[@")  # the year was lost upstream
# a superscript citation flattened by PDF extraction ("magnetization 21."), except after count words
_SUP_SAFE = (
    r"day|days|week|weeks|month|months|year|years|section|sections|table|tables|figure|figures|fig|group"
    r"|groups|phase|step|steps|level|levels|grade|type|stage|chapter|equation|eq|model|wave|cohort|version"
    r"|item|items|class|session|sessions|trial|experiment|study|hypothesis|question|questions|patient|case"
    r"|cases|point|points|round|block|line|row|column|appendix|part|article|page|pages|of|to|and|or|than"
    r"|from|between|aged|age|ages|about|approximately|over|under|by|in|at|on|with|were|was|are|is|be|for"
    r"|only|least|most|total"
)
_SUP_CITE = re.compile(r"\b(?!(?:" + _SUP_SAFE + r")\b)[a-z]{3,} \d{1,3}(?:[,–\-]\d{1,3})*[.;,:](?=\s|$)")


def clean_pandoc_inline(text: str) -> str | None:
    """One pandoc-markdown paragraph (PMC XML -> markdown) -> plain prose; None if it cannot be repaired."""
    t = re.sub(r"[ \t]+", " ", text)
    if len(t) > 8000 or _NARRATIVE_CITE_PANDOC.search(t) or _AUTHOR_YEAR_PANDOC.search(t):
        return None
    t = re.sub(r"\s*\{#[^}]*\}", "", t)  # {#sec1} anchors
    t = re.sub(
        r"\^?\\?\[(\[@[^\]\n]{0,80}\](?:[,;\-–\s]*\[@[^\]\n]{0,80}\])*)\\?\]\^?", "", t
    )  # \[[@a],[@b]\]
    t = re.sub(r"\s*\((?:\[@[^\]\n]{0,80}\][,;\-–\s]*)+\)", "", t)  # ([@B1], [@B2])
    t = re.sub(r"\^?(?:\[@[^\]\n]{0,80}\][,;\-–\s]*)+\^?", "", t)  # bare [@key], ^[@B1]^
    t = re.sub(r'\^?\[[^\]]{0,30}\]\(#[^)\s]{0,80}\)\{ref-type="(?:bibr|ref)"\}\^?', "", t)  # [16](#b16){..}
    t = re.sub(r"\[([^\]]{0,40})\]\(#[^)\s]{0,80}\)\{[^}\n]{0,80}\}", r"\1", t)  # Figure [1](#f1){..} -> 1
    t = re.sub(r"\[([^\]]{0,60})\]\{[^}\n]{0,80}\}", r"\1", t)  # [text]{.smallcaps}
    t = re.sub(r"\*\*([^*\n]{1,800})\*\*", r"\1", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]{1,800})\*(?![\w*])", r"\1", t)
    t = re.sub(r"(?<![\w_])_([^_\n]{1,200})_(?![\w_])", r"\1", t)
    t = re.sub(r"\^([^^\s]{1,20})\^", r"\1", t)  # ^st^ -> st
    t = re.sub(r"~([^~\s]{1,20})~", r"\1", t)  # CO~2~ -> CO2
    t = re.sub(r"\\(?=[^\w\s])", "", t)  # \[ \' escapes
    t = t.replace("\u00ad", "")
    t = re.sub(r"\s+([,.;:)])", r"\1", t)
    t = re.sub(r"\(\s*\)|\[\s*\]", "", t)
    return re.sub(r" {2,}", " ", t).strip()


def clean_grobid_inline(text: str) -> str | None:
    """One Grobid (PDF -> text) paragraph of S2ORC/peS2o -> plain prose; None if it cannot be repaired."""
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"(?<=[a-z]{2}[.!?])(?=[A-Z][a-z])", " ", t)  # sentences joined without a space
    if len(t) > 8000 or _NARRATIVE_CITE_GROBID.search(t) or _SUP_CITE.search(t):
        return None
    t = re.sub(r"\s*(?:\[\s*\d{1,3}(?:\s*[,–\-]\s*\d{1,3})*\s*\])+", "", t)  # [1], [2][3], [26-29, 31]
    t = re.sub(r"\s+([,.;:)])", r"\1", t)
    t = re.sub(r"\(\s*\)", "", t)
    return re.sub(r" {2,}", " ", t).strip()


_PAPER_MATH = re.compile(r"[∑∫∂∇√∞≤≥≈≠±×→←⇒∈∉⊂⊆∀∃∧∨⊗⊕∝ℏ]|\$|\\[a-zA-Z]{2,}|_\{|\^\{")
_GREEK = re.compile(r"[α-ωΑ-Ω]")
_PAPER_BAD_CHARS = re.compile(r"[\[\]{}@#|\\*^]|<[a-z/!]|https?://|www\.|\S+@\S+\.\w+")
_PAPER_STAT = re.compile(r"\b[pP]\s*[<=>≤]\s*0?\.\d|\b(?:OR|HR|RR|CI|SD|SE|IQR)\b|\b\d+(?:\.\d+)?\s*%|±")
_PAPER_CAPTION = re.compile(r"^(?:fig(?:ure)?|table|scheme|chart|plate|supplementary|panel)\s*S?\d", re.I)
_INLINE_CAPTION = re.compile(r"\b(?:Table|Figure|Fig\.)\s*S?\d+[a-z]?[.:]\s+[A-Z]")
_LOWER_SENTENCE = re.compile(
    r"\b(?!(?:approx|resp|incl|spp|subsp|var|cf|vs|etc|ibid)\b)[a-z]{3,}[.!?]\s+[a-z]"
)
_JOINED_SENTENCE = re.compile(r"[a-z]{2}[.!?][A-Z][a-z]")


def paper_paragraph_ok(p: str) -> bool:
    """A plain prose paragraph of a paper: 25-350 words and >= 2 sentences, starts and ends like a sentence,
    no caption, no PDF-extraction noise, no leftover markup, maths, statistics or number dumps."""
    words = p.split()
    n = len(words)
    if not PAPER_PARA_MIN_WORDS <= n <= PAPER_PARA_MAX_WORDS:
        return False
    if not re.match(r"^[\"“‘(]?[A-Z0-9]", p) or not re.search(r"[.?!][\"”’)]?$", p):
        return False
    if _PAPER_CAPTION.match(p) or _INLINE_CAPTION.search(p) or re.match(r"^\d{1,2}[.)]\s", p):
        return False  # a caption or a numbered list item
    if _LOWER_SENTENCE.search(p) or _JOINED_SENTENCE.search(p) or re.search(r"[^.]\.\.(?!\.)", p):
        return False
    if _PAPER_BAD_CHARS.search(p) or _PAPER_MATH.search(p) or len(_GREEK.findall(p)) > 3:
        return False
    single = sum(1 for w in words if len(w.strip(".,;:()")) == 1 and w.strip(".,;:()").lower() not in "ai")
    if single / n > 0.06 or sum(c.isdigit() for c in p) / len(p) > 0.05:
        return False
    if len(_PAPER_STAT.findall(p)) > 3 or len(re.findall(r"[.?!](?:\s|$)", p)) < 2:
        return False
    return sum(1 for w in words if w[:1].isupper()) / n <= 0.4  # not a title-case run


_SETEXT = re.compile(r"^(?:=+|-+)\s*$")
_NON_TEXT_BLOCK = re.compile(r"^\s*(?:!\[|\||\+[-=]|[-*>:<] |\[\^|\d+[.)]\s)")


def pandoc_blocks(text: str) -> list[str | None]:
    """Pandoc markdown (Common Pile PubMed Central) -> body paragraphs and ``None`` breaks.

    YAML front matter and everything before the first heading are skipped; setext (``===``/``---``) and
    ATX (``#``) headings set the section role (a level-1 heading the role of its subsections; an unknown
    level-1 heading is topical body text); an empty ``######`` heading holds a table/figure title, so the
    block after it is skipped; lists, tables, images, quotes and display maths are breaks."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        text = text[end + 5 :] if end > 0 else ""
    out: list[str | None] = []
    role = top = "front"
    skip_next = False
    for block in re.split(r"\n\s*\n", text):
        block = block.strip("\n")
        if not block.strip():
            continue
        lines = block.split("\n")
        heading, level = None, 0
        if len(lines) >= 2 and _SETEXT.match(lines[1]) and len(lines[0]) < 200:
            heading, level = lines[0], 1 if lines[1].startswith("=") else 2
            block = "\n".join(lines[2:])
        elif re.match(r"^#{1,6}(?:\s|$)", block):
            m = re.match(r"^(#{1,6})\s*(.*)", lines[0])
            heading, level = (m.group(2), len(m.group(1))) if m else ("", 6)
            block = "\n".join(lines[1:])
        if heading is not None:
            out.append(None)
            heading = re.sub(r"\s*\{#[^}]*\}", "", heading).strip()
            if level == 6 or not heading:  # table/figure title holder: the next block is the float
                skip_next = True
                continue
            r = heading_role(heading)
            if level == 1:
                top = role = r or "body"
            else:
                role = "back" if "back" in (top, r) else (r or top)
            if not block.strip():
                continue
        if skip_next:
            skip_next = False
            out.append(None)
            continue
        if (
            role not in _BODY_ROLES
            or _NON_TEXT_BLOCK.match(block)
            or re.search(r"^\s*-{3,}(?:\s+-{3,})+\s*$", block, re.M)
            or "$$" in block
            or "\\begin" in block
        ):
            out.append(None)
            continue
        para = clean_pandoc_inline(" ".join(ln.strip() for ln in block.split("\n")))
        out.append(para if para and paper_paragraph_ok(para) else None)
    return out


def grobid_blocks(text: str) -> list[str | None]:
    """peS2o / S2ORC (Grobid) text -> body paragraphs and ``None`` breaks.

    Block 0 is the title, block 1 the abstract (skipped); later blocks are sections separated by a blank
    line, whose first line is the heading when it is short and has no closing punctuation; paragraphs inside
    a section are separated by single newlines. A paper without any recognisable IMRaD heading counts as
    body text."""
    sections = re.split(r"\n\s*\n", text.strip())[2:]

    def is_heading(line: str) -> bool:
        n = len(line.split())
        return n <= 6 or (n <= 15 and not re.search(r"[.?!]$", line))

    firsts = [s.strip().split("\n", 1)[0].strip() for s in sections]
    imrad = any(
        is_heading(f) and heading_role(f) in {"intro", "methods", "discussion", "results"} for f in firsts
    )
    role = "intro" if imrad else "body"
    out: list[str | None] = []
    for sec in sections:
        lines = [ln.strip() for ln in sec.split("\n") if ln.strip()]
        if not lines:
            continue
        out.append(None)
        if is_heading(lines[0]):
            r = heading_role(lines[0])
            if r is not None:
                role = r
            elif not imrad and role != "back":
                role = "body"
            lines = lines[1:]
        for ln in lines:
            para = clean_grobid_inline(ln) if role in _BODY_ROLES else None
            out.append(para if para and paper_paragraph_ok(para) else None)
    return out


def paragraph_windows(blocks: list[str | None], key: str, n: int = PAPER_WINDOWS_PER_DOC) -> list[str]:
    """Up to ``n`` windows of 1-3 consecutive body paragraphs (``None`` breaks a run), 150-500 words each,
    starting at body paragraphs in a seeded random order."""
    import random

    from humanizer_eval.pool.clean import stable_int

    starts = [i for i, p in enumerate(blocks) if p is not None]
    random.Random(stable_int(key)).shuffle(starts)  # noqa: S311 (seeded sampling, not crypto)
    out: list[str] = []
    for i in starts:
        window: list[str] = []
        words = 0
        for p in blocks[i : i + PAPER_MAX_PARAS]:
            if p is None or words + len(p.split()) > PAPER_WINDOW_MAX_WORDS:
                break
            window.append(p)
            words += len(p.split())
        if words >= PAPER_WINDOW_MIN_WORDS:
            out.append("\n\n".join(window))
            if len(out) >= n:
                break
    return out


def json_line(row: dict) -> dict | None:
    """``text``-builder row holding one JSON object per line -> that object."""
    import json

    try:
        value = json.loads(row.get("text") or "")
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def pmc_documents(row: dict) -> list[str]:
    """Common Pile PubMed Central article -> paragraph windows (one is used per article)."""
    return paragraph_windows(pandoc_blocks(str(row.get("text") or "")), f"pmc:{row.get('id')}")


def pes2o_documents(row: dict) -> list[str]:
    """Common Pile peS2o paper (Grobid full text) -> paragraph windows."""
    return paragraph_windows(grobid_blocks(str(row.get("text") or "")), f"pes2o:{row.get('id')}")


PES2O_STEM_FIELDS = frozenset(
    {"Physics", "Computer Science", "Engineering", "Mathematics", "Materials Science", "Chemistry"}
)
PES2O_SOCIAL_FIELDS = frozenset(
    {
        "Psychology",
        "Education",
        "Economics",
        "Sociology",
        "Political Science",
        "Business",
        "Philosophy",
        "History",
        "Linguistics",
        "Law",
        "Art",
        "Geography",
    }
)
_PES2O_EXCLUDED_FIELDS = frozenset({"Medicine", "Biology"})  # biomedicine comes from PubMed Central


def pes2o_created(value: object) -> datetime | None:
    """peS2o ``created``; the placeholders ``0001-01-01`` and ``1970-01-01`` count as missing."""
    dt = parse_date(value)
    if dt is None or dt.year < 1900 or (dt.year, dt.month, dt.day) == (1970, 1, 1):
        return None
    return dt


def pes2o_filter(fields: frozenset[str]) -> RowFilter:
    """peS2o row filter: one of ``fields`` among the S2 fields of study, none of Medicine/Biology, and the
    publication year (``metadata.year``) before 2022 as well as ``created``."""

    def check(row: dict) -> str | None:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        fos = set(meta.get("s2fieldsofstudy") or ())
        if not fos & fields:
            return "other_field"
        if fos & _PES2O_EXCLUDED_FIELDS:
            return "biomedical_field"
        try:
            year = int(meta.get("year"))
        except (TypeError, ValueError):
            return "year_missing"
        return None if year < CUTOFF.year else "date_too_late"

    return check


# --------------------------------------------------------------------------- essay: PERSUADE 2.0 by prompt
#: PERSUADE prompts that ask for an explanatory essay / an analysis of a source text; the other 12 prompts
#: ask for an argument (position, letter to an official, persuasion)
PERSUADE_EXPOSITORY = frozenset({"Car-free cities", "Seeking multiple opinions"})
PERSUADE_ANALYSIS = frozenset({"Exploring Venus"})


def persuade_prompts(prompts: frozenset[str] | None, exclude: frozenset[str] = frozenset()) -> RowFilter:
    def check(row: dict) -> str | None:
        name = str(row.get("prompt_name") or "")
        if name in exclude or (prompts is not None and name not in prompts):
            return "other_prompt"
        return None

    return check


# --------------------------------------------------------------------------- essay: ASAP-AES by essay set
# Hewlett ASAP (2012), grades 7-10: sets 1-2 persuasive (mostly @CAPS-anonymized), 3-5 responses analysing a
# literary text (story/memoir), 6 an explanatory response to an informational text, 7-8 narratives
ASAP_LITERARY = frozenset({3, 4, 5})
ASAP_EXPOSITORY = frozenset({6})
ASAP_NARRATIVE = frozenset({7, 8})
ASAP_FILES = ("training_set_rel3.tsv", "valid_set.tsv", "test_set.tsv")


def asap_sets(sets: frozenset[int]) -> RowFilter:
    def check(row: dict) -> str | None:
        try:
            ok = int(float(str(row.get("essay_set")))) in sets
        except ValueError:
            ok = False
        return None if ok else "other_set"

    return check


def drop_title_line(text: str) -> str:
    """Student essays often open with a title on its own line ("The Debate of the Faces"): drop a first
    line of <= 12 words without closing punctuation."""
    first, sep, rest = text.strip().partition("\n")
    if sep and len(first.split()) <= 12 and not re.search(r"[.!?:\"”']\s*$", first):
        return rest
    return text


def fix_cp1252(text: str) -> str:
    """ASAP is read as latin-1 but written in cp1252: U+0080-U+009F are cp1252 punctuation (\\x92 -> ’)."""

    def repl(m: re.Match[str]) -> str:
        try:
            return m.group().encode("latin-1").decode("cp1252")
        except UnicodeDecodeError:
            return ""

    return re.sub(r"[\x80-\x9f]", repl, text)


# --------------------------------------------------------------------------- application: EssayForum
# essayforum.com posts (admission essays, personal statements, statements of purpose, scholarship essays)
# often open with a request or with the prompt, and sometimes end with thanks; newlines and "/" were removed
# by the dataset's own cleaning.
_EF_SEPARATOR = re.compile(r"(?:-{3,}|_{3,}|={3,}|\*{3,}|~{3,})[\s\-_=*~]*")
_EF_META = re.compile(
    r"\b(?:feedback|proof-?read\w*|critique|critici[sz]e|criticism|comments?|suggestions?|advice|advise"
    r"|thanks?|thank you|appreciated?|help me|any help|word (?:count|limit)|words? (?:max|limit|minimum)"
    r"|character (?:count|limit)|prompt|my (?:essay|draft|statement)|this essay|essays?|posted|re-?post"
    r"|edited|revised?|revision|grammar|please|thread|below|here is|here's)\b",
    re.I,
)
_EF_PROMPT = re.compile(
    r"^(?:(?:prompt|question|topic|essay(?: question)?|option \w+|q\d)\s*[:#-]"
    r"|(?:describe|discuss|tell us|tell me|explain|share|reflect|write|imagine|consider|choose|select"
    r"|elaborate|evaluate|identify|state|briefly|what|why|how|who|which|in \d+ words|using \d+ words)\b)",
    re.I,
)
_EF_PROMPT_YOU = re.compile(
    r"\byour?\b.*\b(?:describe|discuss|explain|tell|share|reflect|elaborate|evaluate|identify|consider"
    r"|choose)\b|\b(?:describe|discuss|explain|tell us|share|reflect|elaborate)\b.*\byour?\b",
    re.I,
)
_EF_REQUEST = re.compile(
    r"\b(?:please|feedback|proof-?read\w*|critique|thank you|thanks|any (?:help|comments|suggestions|advice)"
    r"|word (?:count|limit))\b",
    re.I,
)
EF_SLASH_MERGED = re.compile(
    r"\b(?:andor|hisher|heshe|himher|sirmadam|sheher|hehim|theythem|himherself|heorshe)\b", re.I
)


def clean_essayforum(text: str) -> str:
    """Keep the essay itself: cut a request/prompt preamble (before a ``-----`` separator, or leading
    sentences that ask for feedback or read like the prompt), trailing thanks, and anything from a later
    request sentence on; restore the space of sentences joined by the dataset's newline removal."""
    t = text.strip()
    m = _EF_SEPARATOR.search(t)
    if m and m.start() < 0.4 * len(t):
        t = t[m.end() :]
    t = _EF_SEPARATOR.sub(" ", t)
    t = re.sub(r"(?<=[a-z]{2}[.!?])(?=[A-Z][a-z])", " ", t)
    sents = [s for s in re.split(r"(?<=[.!?])[\"')\]]?\s+", t) if s.strip()]
    i = 0
    while i < min(len(sents), 8) and (
        _EF_META.search(sents[i]) or _EF_PROMPT.match(sents[i]) or _EF_PROMPT_YOU.search(sents[i])
    ):
        i += 1
    sents = sents[i:]
    for j, sent in enumerate(sents):
        if j and _EF_REQUEST.search(sent):
            sents = sents[:j]
            break
    return " ".join(sents)


def essayforum_row(row: dict) -> str | None:
    """Drop posts where the dataset's "/" removal merged words (``andor``, ``SirMadam``)."""
    return "slash_merged" if EF_SLASH_MERGED.search(str(row.get("Cleaned Essay") or "")) else None


# --------------------------------------------------------------------------- business / marketing
def amazon_description(row: dict) -> list[str]:
    """Amazon 2018 product metadata: ``description`` (a string, or a list of paragraphs)."""
    d = row.get("description")
    text = "\n\n".join(str(x) for x in d) if isinstance(d, list) else str(d or "")
    return [text] if text.strip() else []


def clean_amazon(text: str) -> str:
    """Product description HTML -> text: line-break tags and inline ``*`` bullets become lines."""
    t = re.sub(r"<br\s*/?>|</?p>|</?li>|</?ul>|</?div>", "\n", text, flags=re.I)
    t = html.unescape(_HTML_TAG.sub("", t))
    return re.sub(r"\s*\*\s*(?=[A-Z])", "\n", t)


_PRESS_URL = re.compile(r"press-releases?|/press/|/news-releases?/|/pressroom/|/media-releases?/", re.I)
_DATELINE = re.compile(
    r"^[^\n]{0,160}?(?:--\s*\((?:Marketwired?|BUSINESS WIRE|GLOBE NEWSWIRE|ACCESSWIRE|PRWEB)[^)]{0,60}\)"
    r"\s*-{1,2}|/\s*PRNewswire[^/]*/\s*--?|\((?:GLOBE NEWSWIRE|BUSINESS WIRE|PRWEB)\)\s*-{1,2}"
    r"|/\s*ACCESSWIRE\s*/[^/]{0,40}/"
    # "(FSCWire via Comtex) - Vancouver, British Columbia (FSCwire) - "
    r"|\([A-Z][A-Za-z]{0,20}(?i:wire)(?:\s+via\s+Comtex)?\)\s*[-–—]{1,2}"
    r"(?:\s*[^-–—\n]{0,60}\([A-Z][A-Za-z]{0,20}(?i:wire)\)\s*[-–—]{1,2})?)"
    r"\s*",
)
_DATELINE_PLAIN = re.compile(
    r"^[A-Z][A-Z .,'()-]{2,60}(?:\([^)]{0,40}\))?\s*[-–—]{1,2}\s+(?=[A-Z])"  # "MONTREAL (April, 2018) – "
)
_PRESS_TAIL = re.compile(
    r"^(?:about [A-Z0-9][^\n]{0,80}$|about us\b|media contacts?\b|contacts?:|press contacts?\b"
    r"|for more information"
    r"|source:|###|- ?30 ?-|forward-looking statements?\b|safe harbor\b|this press release contains)",
    re.I | re.M,
)


# government and intergovernmental press offices (court cases, UN briefings) are not business writing
_PUBLIC_SECTOR_HOST = re.compile(
    r"\.gov(?:\.[a-z]{2})?$|\.mil$|(?:^|\.)(?:un|who|nato|wto|imf|worldbank|oecd)\.(?:org|int)$|\.europa\.eu$|\.int$"
)


def c4_press_page(row: dict) -> str | None:
    url = str(row.get("url") or "")
    if not _PRESS_URL.search(url):
        return "not_press_url"
    m = re.match(r"^https?://([^/:]+)", url)
    return "public_sector" if m and _PUBLIC_SECTOR_HOST.search(m.group(1).lower()) else None


def clean_press(text: str) -> str:
    """Press release -> the announcement: drop the wire dateline and cut at the boilerplate tail (About the
    company, contacts, SOURCE, forward-looking statements)."""
    t = html.unescape(text).strip()
    t = _DATELINE.sub("", t, count=1)
    t = _DATELINE_PLAIN.sub("", t, count=1)
    m = _PRESS_TAIL.search(t)
    if m:
        t = t[: m.start()]
    return clean_c4_news(t)


_ABOUT_URL = re.compile(
    r"/about(?:-?us)?/?$|/about-[a-z0-9-]+/?$|/our-story/?$|/who-we-are/?$|/company/?$|/services?/?(?:[a-z0-9-]+/?)?$",
    re.I,
)


def c4_about_page(row: dict) -> str | None:
    """Business website copy: About-us / our-story / services pages of C4 (Common Crawl, Apr 2019)."""
    return None if _ABOUT_URL.search(str(row.get("url") or "")) else "not_about_url"


C4_EN_FILES: tuple[str, ...] = tuple(f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(0, 1024, 16))

_EDGAR_BOILERPLATE = re.compile(
    r"forward-looking|should be read in conjunction|private securities litigation reform act"
    r"|incorporated (?:herein )?by reference|^not applicable|^none\.?$|safe harbor|see (?:item|note)\s",
    re.I,
)
_EDGAR_CONTACT = re.compile(r"telephone|website|web site|www\.|internet address|e-mail|facsimile", re.I)


def year_date(value: object) -> datetime | None:
    """A bare year (EDGAR-CORPUS ``year``: the filing year) -> Jan 1 of that year."""
    m = re.match(r"^\s*(\d{4})\s*$", str(value or ""))
    return datetime(int(m.group(1)), 1, 1, tzinfo=UTC) if m else None


def clean_edgar(text: str) -> str:
    """10-K section -> narrative paragraphs: drop the ITEM heading, sub-headings and table rows (lines
    without closing punctuation), number-heavy lines, contact lines and standard legal boilerplate."""
    out = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or not re.search(r"[.!?:][\"')”]?$", s) or re.match(r"^item\s+\d", s, re.I):
            continue
        words = s.split()
        if len(words) < 8 or sum(any(c.isdigit() for c in w) for w in words) > 0.2 * len(words):
            continue
        letters = [c for c in s if c.isalpha()]
        if sum(c.isupper() for c in letters) > 0.5 * len(letters):  # an upper-case heading with a period
            continue
        if _EDGAR_BOILERPLATE.search(s) or _EDGAR_CONTACT.search(s):
            continue
        out.append(s)
    return "\n\n".join(out)


def edgar_section(name: str) -> RowDocuments:
    def documents(row: dict) -> list[str]:
        text = row.get(name)
        return [text] if isinstance(text, str) and text.strip() else []

    return documents


EDGAR_YEARS = tuple(range(2001, 2021))  # the corpus starts in 1993; recent filings read more like today's


# --------------------------------------------------------------------------- homework answers
_MOD_TEMPLATE = re.compile(
    r"submission has been removed|please read this entire message|thank you for your submission"
    r"|your (?:post|question|submission|comment) (?:has been|was) removed|this subreddit does not"
    r"|i am a bot|\bperformed automatically\b|message the moderators|contact the moderators",
    re.I,
)
_REDDIT_BOTS = frozenset({"AutoModerator", "AskScienceModerator", "[deleted]"})


def reddit_answer(min_score: int) -> RowFilter:
    """A direct answer: a top-level comment (parent is the question post) by a person, not a moderator
    template, with at least ``min_score`` points."""

    def check(row: dict) -> str | None:
        if not str(row.get("parent_id") or "").startswith("t3_"):
            return "not_top_level"
        if row.get("author") in _REDDIT_BOTS:
            return "bot"
        if _MOD_TEMPLATE.search(str(row.get("body") or "")):
            return "mod_template"
        try:
            score = int(str(row.get("score") or "0"))
        except ValueError:
            score = 0
        return None if score >= min_score else "low_score"

    return check


def se_answer(row: dict) -> list[str]:
    """flax StackExchange dump: each line is a JSON pair ``[title + body, best-voted answer]``. Indented
    lines (quoted material, code) are not the answerer's words and are dropped."""
    import json

    try:
        pair = json.loads(row.get("text") or "")
    except ValueError:
        return []
    if not (isinstance(pair, list) and len(pair) == 2 and isinstance(pair[1], str)):
        return []
    lines = [ln for ln in html.unescape(pair[1]).split("\n") if not re.match(r"^(?: {2,}|\t)\S", ln)]
    text = "\n".join(lines)
    return [text] if text.strip() else []


_SE_FLAX = "flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl"
SE_SCIENCE_SITES = ("physics", "chemistry", "biology", "astronomy", "earthscience")
SE_HUMANITIES_SITES = ("history", "philosophy", "english", "linguistics", "economics")


def geclm_files(sub: str, shards: int, used: range) -> tuple[str, ...]:
    """File names of one subreddit in ``HuggingFaceGECLM/REDDIT_comments``
    (``data/<sub>-NNNNN-of-NNNNN-<hash>.parquet``; the hash suffixes are listed in ``_GECLM_HASHES``)."""
    return tuple(f"data/{sub}-{i:05d}-of-{shards:05d}-{_GECLM_HASHES[sub][i]}.parquet" for i in used)


_GECLM_HASHES: dict[str, tuple[str, ...]] = {
    "explainlikeimfive": (
        "21d7b92cb6f7a8ba", "8e9fcac1ac51302d", "3035f5b3f6424b27", "8dda4e148fb72dd8", "fcabc283419ad6be",
        "b35d7d10ff7e1ac7", "f5b0c96013ea3d99", "cd5562ee2c9dfa76", "7c7ad6ac471a7939", "f4f4a8df3d44fe2f",
        "5c425c8093c7b6fd", "050d6c50793d3ea5", "cdbb55dabc286b10", "1a6f7c25956c066a", "906d2dc58f521a00",
        "3743f03e855fb782", "cf9d1ebec8a07140",
    ),
    "askscience": (
        "47a34646ed6e11b0", "08f9e5b175e2a903", "6a325b8989ac345d", "9b14fd494e91e8d6", "e7a25f47870bd937",
        "733061c82dd94184", "1bc8e1f81c768875",
    ),
}  # fmt: skip


# --------------------------------------------------------------------------- registry types
DateParser = Callable[[object], "datetime | None"]
RowDocuments = Callable[[dict], list[str]]
Preclean = Callable[[str], str]
RowFilter = Callable[[dict], "str | None"]
RowParser = Callable[[dict], "dict | None"]


@dataclass(frozen=True)
class Source:
    """One corpus on the Hugging Face Hub."""

    id: str
    genre: str
    hf_id: str
    text_field: str
    license: str
    date_evidence: str
    config: str | None = None
    id_field: str | None = None  # native document id, kept as source_doc_id
    split: str = "train"
    data_files: tuple[str, ...] | None = None
    builder: str | None = None  # e.g. "json": read ``data_files`` of ``hf_id`` raw, bypassing its script
    builder_kwargs: tuple[tuple[str, str], ...] = ()  # extra builder options, e.g. (("sep", "\t"),)
    interleave_files: bool = False  # one stream per data file, round-robin (topic diversity)
    stride: int = 1  # use every n-th row: spreads a sample over corpora stored in sorted order
    oversample: float = 1.0  # extra candidates for sources with many duplicates (e.g. Enron)
    # rows to scan per wanted candidate before giving up (None: BuildConfig default); raise it for sources
    # whose ``row_filter`` keeps a tiny share of rows (URL-selected C4 pages, curated args.me portals)
    max_rows_per_candidate: int | None = None
    # stream ``data_files`` as this many contiguous slices in parallel workers (quota and row budget are
    # split evenly): one stream of all 512 C4 realnewslike shards takes ~1 h
    parallel_files: int = 1
    date_field: str | None = None
    date_parser: DateParser = parse_date
    documents: RowDocuments | None = None
    preclean: Preclean | None = None
    row_filter: RowFilter | None = None  # row -> rejection reason, or None to keep
    # raw row -> row dict (None: unparseable); e.g. JSON lines read with the ``text`` builder, when Arrow
    # type inference of the json builder fails on a column (Common Pile PubMed dates)
    row_parser: RowParser | None = None
    weight: float = 1.0  # share of the genre's quota
    # kind of text inside the genre (essay: argumentative/expository/narrative/literary; academic:
    # paper/abstract), used to report the achieved within-genre mix against the production mix
    subtype: str = ""
    enabled: bool = True
    notes: str = ""

    def parse_row(self, row: dict) -> dict | None:
        return self.row_parser(row) if self.row_parser is not None else row

    def row_documents(self, row: dict) -> list[str]:
        if self.documents is not None:
            return self.documents(row)
        value = row.get(self.text_field)
        return [value] if isinstance(value, str) and value.strip() else []

    def row_id(self, row: dict, index: int) -> str:
        value = row.get(self.id_field) if self.id_field else None
        return str(value) if value not in (None, "") else f"row{index}"

    def row_date(self, row: dict) -> datetime | None:
        if not self.date_field:
            return None
        return self.date_parser(row.get(self.date_field))

    def check_date(self, row: dict) -> tuple[bool, str]:
        """(ok, reason). Dataset-level evidence always passes; row-level evidence is strict."""
        if not self.date_field:
            return True, "dataset"
        dt = self.row_date(row)
        if dt is None:
            return False, "date_missing"
        if dt >= CUTOFF:
            return False, "date_too_late"
        return True, "row"


@dataclass(frozen=True)
class RejectedSource:
    hf_id: str
    genre: str
    status: str  # "works_not_used" | "broken" | "unsafe"
    reason: str


# --------------------------------------------------------------------------- the registry
_REDDIT_SUBS = ("AskHistorians", "DIY", "Fitness", "LifeProTips", "EatCheapAndHealthy", "Games")

SOURCES: tuple[Source, ...] = (
    # ---- essay. Production mix (2026-11): 49% expository, 24% argumentative, 17% narrative/personal, 10%
    # literary analysis. Non-argumentative student essays first (scarce: every one is used), then the
    # argumentative sources of Q-11. Registry order = dedupe priority and top-up order: a shortfall of the
    # scarce subtypes is topped up from CMV first (adult essays), not from school essays.
    Source(
        id="persuade2_expository",
        subtype="expository",
        id_field="essay_id_comp",
        genre="essay",
        hf_id="nlpatunt/D_persuade_2",
        data_files=("train.csv", "validation.csv", "test.csv"),
        text_field="full_text",
        row_filter=persuade_prompts(PERSUADE_EXPOSITORY),
        preclean=drop_title_line,
        license="upstream PERSUADE 2.0: CC BY 4.0 (per its GitHub repository); mirror card: none",
        date_evidence="PERSUADE 2.0 essays by US students (grades 6-12), written 2010-2020; corpus used in "
        "the Kaggle Feedback Prize (Dec 2021)",
        weight=0.12,
        notes="prompts 'Car-free cities' (explanatory essay on a source set) and 'Seeking multiple "
        "opinions'; "
        "the mirror's `gpt4_summary` column is LLM text and is not read",
    ),
    Source(
        id="persuade2_analysis",
        subtype="literary",
        id_field="essay_id_comp",
        genre="essay",
        hf_id="nlpatunt/D_persuade_2",
        data_files=("train.csv", "validation.csv", "test.csv"),
        text_field="full_text",
        row_filter=persuade_prompts(PERSUADE_ANALYSIS),
        preclean=drop_title_line,
        license="upstream PERSUADE 2.0: CC BY 4.0 (per its GitHub repository); mirror card: none",
        date_evidence="PERSUADE 2.0 essays by US students (grades 6-12), written 2010-2020; corpus used in "
        "the Kaggle Feedback Prize (Dec 2021)",
        weight=0.055,
        notes="prompt 'Exploring Venus': evaluate how well an article's author supports a claim (analysis of "
        "a source text; counted with literary analysis)",
    ),
    Source(
        id="asap_literary",
        subtype="literary",
        id_field="essay_id",
        genre="essay",
        hf_id="TasfiaS/ASAP-AES",
        builder="csv",
        builder_kwargs=(("sep", "\t"), ("encoding", "latin-1")),
        data_files=ASAP_FILES,
        interleave_files=True,  # the three files have different columns: one stream each
        text_field="essay",
        row_filter=asap_sets(ASAP_LITERARY),
        preclean=fix_cp1252,
        max_rows_per_candidate=400,
        license="Hewlett Foundation ASAP competition data (Kaggle, 2012); no open license stated; mirror "
        "card: none",
        date_evidence="ASAP-AES: essays by US grade 7-10 students, released for the Kaggle competition in "
        "Feb 2012",
        weight=0.03,
        notes="essay sets 3-5: responses analysing a story/memoir; essays with @CAPS-style anonymization "
        "tokens are rejected (only ~20-30% are >= 150 words)",
    ),
    Source(
        id="asap_expository",
        subtype="expository",
        id_field="essay_id",
        genre="essay",
        hf_id="TasfiaS/ASAP-AES",
        builder="csv",
        builder_kwargs=(("sep", "\t"), ("encoding", "latin-1")),
        data_files=ASAP_FILES,
        interleave_files=True,
        text_field="essay",
        row_filter=asap_sets(ASAP_EXPOSITORY),
        preclean=fix_cp1252,
        max_rows_per_candidate=400,
        license="Hewlett Foundation ASAP competition data (Kaggle, 2012); no open license stated; mirror "
        "card: none",
        date_evidence="ASAP-AES: essays by US grade 7-10 students, released for the Kaggle competition in "
        "Feb 2012",
        weight=0.02,
        notes="essay set 6: explaining the obstacles described in an informational text",
    ),
    Source(
        id="asap_narrative",
        subtype="narrative",
        id_field="essay_id",
        genre="essay",
        hf_id="TasfiaS/ASAP-AES",
        builder="csv",
        builder_kwargs=(("sep", "\t"), ("encoding", "latin-1")),
        data_files=ASAP_FILES,
        interleave_files=True,
        text_field="essay",
        row_filter=asap_sets(ASAP_NARRATIVE),
        preclean=fix_cp1252,
        max_rows_per_candidate=400,
        license="Hewlett Foundation ASAP competition data (Kaggle, 2012); no open license stated; mirror "
        "card: none",
        date_evidence="ASAP-AES: essays by US grade 7-10 students, released for the Kaggle competition in "
        "Feb 2012",
        weight=0.005,
        notes="essay sets 7-8: personal narratives (patience, laughter); most carry @CAPS tokens "
        "(~120 usable)",
    ),
    Source(
        id="cmv",
        subtype="argumentative",
        stride=40,  # the dump is in time order: at divisor 10 (stride 4) the sample spans 2013-2021
        id_field="id",
        genre="essay",
        hf_id="HuggingFaceGECLM/REDDIT_submissions",
        data_files=("data/changemyview-*.parquet",),
        text_field="selftext",
        documents=cmv_documents,
        preclean=clean_cmv,
        date_field="created_utc",
        license="Reddit content via Pushshift; mirror card: no license",
        date_evidence="row-level `created_utc` of the submission (dump: 2013-2023), filtered < 2022-01-01",
        weight=0.28,
        notes="r/changemyview opening posts: argued opinion essays by adults; edits/footnotes cut",
    ),
    Source(
        id="c4_opinion",
        subtype="argumentative",
        id_field="url",
        genre="essay",
        hf_id="allenai/c4",
        data_files=C4_REALNEWS_FILES,
        text_field="text",
        row_filter=c4_opinion_page,
        preclean=clean_c4_news,
        date_field="timestamp",
        max_rows_per_candidate=400,
        license="C4: ODC-BY 1.0 (+ Common Crawl terms); articles (c) publishers",
        date_evidence="row-level `timestamp` = Common Crawl fetch time (Apr 2019), filtered < 2022-01-01",
        weight=0.14,
        notes="C4 realnewslike pages under /opinion/, /op-ed/, /commentary/, /editorial/ (not letters)",
    ),
    Source(
        id="argsme",
        subtype="argumentative",
        id_field="id",
        genre="essay",
        hf_id="webis/args_me",
        builder="json",
        data_files=("args-me.jsonl",),
        text_field="premises",
        documents=argsme_documents,
        row_filter=argsme_curated_portal,
        preclean=clean_argsme,
        date_field="context",
        date_parser=argsme_acquired,
        max_rows_per_candidate=5000,  # the edited portals start after 333k debate.org rows
        license="CC BY 4.0 (args.me corpus, Webis)",
        date_evidence="row-level `context.acquisitionTime` = crawl of the debate portal (Apr 2019), "
        "filtered < 2022-01-01",
        weight=0.15,
        notes="idebate.org, debatewise.org, debatepedia.org arguments only (~5.7k usable); raw jsonl read "
        "with the json builder because the repo's loading script is unsupported",
    ),
    Source(
        id="persuade2",
        subtype="argumentative",
        stride=7,
        id_field="essay_id_comp",
        genre="essay",
        hf_id="nlpatunt/D_persuade_2",
        data_files=("train.csv", "validation.csv", "test.csv"),
        text_field="full_text",
        row_filter=persuade_prompts(None, exclude=PERSUADE_EXPOSITORY | PERSUADE_ANALYSIS),
        preclean=drop_title_line,
        license="upstream PERSUADE 2.0: CC BY 4.0 (per its GitHub repository); mirror card: none",
        date_evidence="argumentative essays by US students (grades 6-12), written 2010-2020; used in "
        "Kaggle Feedback Prize (launched Dec 2021)",
        weight=0.10,
        notes="the 12 argumentative prompts; this mirror carries `prompt_name` (realbenpope/"
        "PERSUADE_manageable, used before, does not)",
    ),
    Source(
        id="ellipse",
        subtype="argumentative",
        stride=3,
        id_field="text_id",
        genre="essay",
        hf_id="tcapelle/feedback-prize-english-language-learning-fluency",
        text_field="full_text",
        license="upstream ELLIPSE corpus (Kaggle Feedback Prize ELL): CC BY 4.0 (per its GitHub repository); "
        "no card",
        date_evidence="ELL student essays (grades 8-12) from pre-2021 assessments; released Aug 2022, "
        "before ChatGPT; mirror adds LLM-written score rationales in other columns (not used)",
        weight=0.1,
        notes="~85% argumentative, ~11% expository/descriptive, ~4% narrative (200 essays read); no prompt "
        "column to route them",
    ),
    # ---- article / blog
    Source(
        id="medium",
        stride=20,
        id_field="url",
        genre="article_blog",
        hf_id="fabiochiu/medium-articles",
        text_field="text",
        documents=medium_documents,
        date_field="timestamp",
        license="mirror card: MIT; posts (c) their authors",
        date_evidence="row-level `timestamp` (publication time), filtered < 2022-01-01",
        weight=0.6,
    ),
    Source(
        id="blog_authorship",
        stride=50,
        genre="article_blog",
        hf_id="tasksource/blog_authorship_corpus",
        text_field="text",
        preclean=clean_blog,
        license="mirror card: Apache-2.0; upstream: non-commercial research use",
        date_evidence="Blog Authorship Corpus: blogger.com posts collected Aug 2004 (Schler et al., 2006)",
        weight=0.4,
    ),
    # ---- academic. Production: ~88% sections of papers/theses (1-3 paragraphs), ~12% abstracts. Paper
    # paragraphs: biomedicine from PubMed Central, physics/CS/engineering and social sciences/humanities
    # from peS2o (S2ORC), split by field of study; then the abstract sources.
    Source(
        id="pmc_papers",
        subtype="paper",
        id_field="id",
        genre="academic",
        hf_id="common-pile/pubmed",
        builder="text",  # the json builder infers `created` as a timestamp and fails on "2021-8-13"
        row_parser=json_line,
        data_files=tuple(f"data/{i:05d}_pubmedcentral.jsonl.gz" for i in range(12)),
        interleave_files=True,  # shards are in PMC-id (= time) order: 2003 ... 2022
        parallel_files=6,
        text_field="text",
        documents=pmc_documents,
        date_field="created",
        license="Common Pile v0.1: openly licensed PMC articles only; per-article license in "
        "`metadata.license` (CC BY / CC BY-SA / CC0)",
        date_evidence="row-level `created` = article publication date, filtered < 2022-01-01",
        weight=0.34,
        notes="pandoc markdown converted from the publisher XML; 1-3 consecutive paragraphs of introduction/"
        "methods/discussion (no results, headings, captions, tables, maths, reference-heavy paragraphs)",
    ),
    Source(
        id="pes2o_social",
        subtype="paper",
        id_field="id",
        genre="academic",
        hf_id="common-pile/peS2o",
        builder="json",
        data_files=tuple(f"v0/documents/peS2o-{i:04d}.json.gz" for i in range(51, 102)),
        parallel_files=6,
        text_field="text",
        documents=pes2o_documents,
        row_filter=pes2o_filter(PES2O_SOCIAL_FIELDS),
        date_field="created",
        date_parser=pes2o_created,
        license="Common Pile v0.1: openly licensed S2ORC papers; per-paper `metadata.oa_license` (CC BY / "
        "CC BY-SA / public domain)",
        date_evidence="row-level `created` (publication date) and `metadata.year`, both < 2022",
        weight=0.30,
        notes="fields of study: psychology, education, economics, sociology, political science, business, "
        "philosophy, history, linguistics, law, art, geography (not medicine/biology); Grobid PDF text",
    ),
    Source(
        id="pes2o_stem",
        subtype="paper",
        id_field="id",
        genre="academic",
        hf_id="common-pile/peS2o",
        builder="json",
        data_files=tuple(f"v0/documents/peS2o-{i:04d}.json.gz" for i in range(51)),
        parallel_files=6,
        text_field="text",
        documents=pes2o_documents,
        row_filter=pes2o_filter(PES2O_STEM_FIELDS),
        date_field="created",
        date_parser=pes2o_created,
        license="Common Pile v0.1: openly licensed S2ORC papers; per-paper `metadata.oa_license` (CC BY / "
        "CC BY-SA / public domain)",
        date_evidence="row-level `created` (publication date) and `metadata.year`, both < 2022",
        weight=0.24,
        notes="fields of study: physics, computer science, engineering, mathematics, materials science, "
        "chemistry (not medicine/biology); maths-heavy paragraphs are skipped",
    ),
    Source(
        id="pubmed_abstracts",
        subtype="abstract",
        genre="academic",
        hf_id="timaeus/pile-pubmed_abstracts",
        text_field="text",
        preclean=drop_first_line,
        license="PubMed abstracts via The Pile (publisher terms); mirror card: none",
        date_evidence="The Pile subset (Pile released 2020-12-31)",
        weight=0.05,
    ),
    Source(
        id="arxiv_abstracts",
        subtype="abstract",
        stride=300,
        id_field="id",
        genre="academic",
        hf_id="gfissore/arxiv-abstracts-2021",
        text_field="abstract",
        preclean=unwrap_lines,
        date_field="id",
        date_parser=arxiv_id_date,
        license="CC0-1.0 (arXiv metadata)",
        date_evidence="row-level: arXiv id encodes submission month (all <= 2021 in this 2021 snapshot)",
        weight=0.04,
    ),
    Source(
        id="nih_exporter",
        subtype="abstract",
        genre="academic",
        hf_id="timaeus/pile-nih_exporter",
        text_field="text",
        license="NIH grant abstracts (US government, public domain) via The Pile",
        date_evidence="The Pile subset (Pile released 2020-12-31)",
        weight=0.03,
    ),
    # ---- news
    Source(
        id="cc_news",
        stride=100,
        id_field="url",
        genre="news",
        hf_id="vblagoje/cc_news",
        text_field="text",
        date_field="date",
        license="unknown (CommonCrawl News; articles (c) publishers)",
        date_evidence="row-level `date` (2017-2018), filtered < 2022-01-01",
        weight=0.7,
    ),
    Source(
        id="cnn",
        stride=50,
        id_field="id",
        genre="news",
        hf_id="abisee/cnn_dailymail",
        config="3.0.0",
        text_field="article",
        preclean=clean_cnn_dm,
        license="dataset: Apache-2.0; articles (c) CNN",
        date_evidence="CNN/DailyMail corpus (Hermann et al., 2015): articles 2007-2015; Daily Mail part "
        "dropped "
        "(byline/timestamp boilerplate)",
        weight=0.3,
    ),
    # ---- email / letter: Enron business email + published letters to the editor (Q-11)
    Source(
        id="enron",
        row_filter=enron_internal_sender,
        oversample=1.7,
        stride=3,
        id_field="message_id",
        genre="email_letter",
        hf_id="corbt/enron-emails",
        text_field="body",
        preclean=clean_enron,
        date_field="date",
        license="Enron corpus: public record released by FERC (2003); mirror card: none",
        date_evidence="row-level `date` (1999-2002; a few bogus 1980 dates, still < 2022)",
        weight=0.55,
    ),
    Source(
        id="c4_letters",
        id_field="url",
        genre="email_letter",
        hf_id="allenai/c4",
        data_files=C4_REALNEWS_FILES,
        text_field="text",
        row_filter=c4_letter_page,
        preclean=clean_c4_news,
        date_field="timestamp",
        max_rows_per_candidate=3000,
        parallel_files=8,
        license="C4: ODC-BY 1.0 (+ Common Crawl terms); letters (c) their authors/publishers",
        date_evidence="row-level `timestamp` = Common Crawl fetch time (Apr 2019), filtered < 2022-01-01",
        weight=0.45,
        notes="letters to the editor from C4 realnewslike (~1 per 900 pages): letter URLs, no listings or "
        "round-ups, whole page <= 350 words (one letter, one author)",
    ),
    # ---- review
    Source(
        id="yelp",
        id_field="review_id",
        genre="review",
        hf_id="yashraizad/yelp-open-dataset-reviews",
        text_field="text",
        date_field="date",
        license="Yelp Open Dataset terms (non-commercial); mirror card: Apache-2.0",
        date_evidence="row-level `date`, filtered < 2022-01-01",
        weight=0.65,
    ),
    Source(
        id="imdb",
        stride=8,
        genre="review",
        hf_id="stanfordnlp/imdb",
        config="plain_text",
        text_field="text",
        preclean=clean_imdb,
        license="other (IMDb reviews, ACL 2011 release)",
        date_evidence="Large Movie Review Dataset (Maas et al., 2011)",
        weight=0.35,
    ),
    # ---- forum / Q&A
    Source(
        id="reddit",
        id_field="id",
        genre="forum_qa",
        hf_id="HuggingFaceGECLM/REDDIT_comments",
        data_files=tuple(f"data/{s}-00000-of-*.parquet" for s in _REDDIT_SUBS),
        interleave_files=True,
        text_field="body",
        documents=reddit_documents,
        preclean=clean_reddit,
        date_field="created_utc",
        license="Reddit content via Pushshift; mirror card: no license",
        date_evidence="row-level `created_utc`, filtered < 2022-01-01",
        weight=0.6,
        notes="subreddits: " + ", ".join(_REDDIT_SUBS),
    ),
    Source(
        id="stackexchange",
        genre="forum_qa",
        hf_id="timaeus/pile-stackexchange",
        text_field="text",
        documents=stackexchange_posts,
        license="CC BY-SA (Stack Exchange) via The Pile",
        date_evidence="The Pile subset (Pile released 2020-12-31)",
        weight=0.25,
    ),
    Source(
        id="hackernews",
        id_field="id",
        genre="forum_qa",
        hf_id="jkeisling/hacker-news-corpus-2007-2022",
        text_field="text",
        preclean=clean_html_comment,
        date_field="timestamp",
        license="mirror card: MIT; comments (c) their authors",
        date_evidence="row-level `timestamp`, filtered < 2022-01-01 (corpus runs to Nov 2022)",
        weight=0.15,
    ),
    # ---- creative
    Source(
        id="writingprompts",
        genre="creative",
        hf_id="sileod/reddit-WritingPrompts",
        config="comments",
        data_files=tuple(f"comments/train-0000{i}-of-00009.parquet" for i in range(1, 8)),
        interleave_files=True,
        id_field="id",
        text_field="body",
        documents=writingprompts_story,
        preclean=clean_reddit,
        date_field="created_utc",
        license="Reddit r/WritingPrompts content via Pushshift; mirror card: no license",
        date_evidence="row-level `created_utc` of the story comment, filtered < 2022-01-01",
        weight=1.0,
        notes="top-level comments only (the stories); untokenized original text",
    ),
    # ---- application: admission essays, personal statements, statements of purpose (2026-11 production
    # category). The only pre-2022 human source found; cover/motivation letter sets are templated or LLM.
    Source(
        id="essayforum",
        subtype="admission_essay",
        genre="application",
        hf_id="nid989/EssayFroum-Dataset",
        data_files=("EssayForum_Essays_cleaned.csv",),
        text_field="Cleaned Essay",
        row_filter=essayforum_row,
        preclean=clean_essayforum,
        license="card: apache-2.0; posts (c) their authors (essayforum.com)",
        date_evidence="corpus release: the repo was created and its only data commit made on 2022-09-02 "
        "(before ChatGPT, Nov 2022); no per-post date, so posts of Jan-Aug 2022 cannot be excluded",
        weight=1.0,
        notes="essayforum.com threads asking for feedback on college admission essays, supplements, "
        "statements of purpose, scholarship essays (a few cover letters); the request/prompt preamble is "
        "cut; "
        "only the `Cleaned Essay` column (the `Correct Grammar` column is machine-corrected)",
    ),
    # ---- business / marketing (2026-11 production category): product descriptions, press releases,
    # company website copy, annual-report narrative
    Source(
        id="amazon_sports",
        subtype="product_description",
        id_field="asin",
        genre="business_marketing",
        hf_id="smartcat/Amazon_Sports_and_Outdoors_2018",
        config="metadata",
        text_field="description",
        documents=amazon_description,
        preclean=clean_amazon,
        license="Amazon Review Data 2018 (McAuley lab, research use); mirror card: none",
        date_evidence="Amazon Review Data 2018 (Ni, Li & McAuley, EMNLP 2019): product metadata crawled 2018",
        weight=0.15,
    ),
    Source(
        id="amazon_toys",
        subtype="product_description",
        id_field="asin",
        genre="business_marketing",
        hf_id="smartcat/Amazon_Toys_and_Games_2018",
        config="metadata",
        text_field="description",
        documents=amazon_description,
        preclean=clean_amazon,
        license="Amazon Review Data 2018 (McAuley lab, research use); mirror card: none",
        date_evidence="Amazon Review Data 2018 (Ni, Li & McAuley, EMNLP 2019): product metadata crawled 2018",
        weight=0.15,
    ),
    Source(
        id="amazon_beauty",
        subtype="product_description",
        id_field="asin",
        genre="business_marketing",
        hf_id="smartcat/Amazon_All_Beauty_2018",
        config="metadata",
        text_field="description",
        documents=amazon_description,
        preclean=clean_amazon,
        max_rows_per_candidate=200,
        license="Amazon Review Data 2018 (McAuley lab, research use); mirror card: none",
        date_evidence="Amazon Review Data 2018 (Ni, Li & McAuley, EMNLP 2019): product metadata crawled 2018",
        weight=0.05,
    ),
    Source(
        id="c4_press",
        subtype="press_release",
        id_field="url",
        genre="business_marketing",
        hf_id="allenai/c4",
        data_files=C4_REALNEWS_FILES,
        text_field="text",
        row_filter=c4_press_page,
        preclean=clean_press,
        date_field="timestamp",
        max_rows_per_candidate=600,
        parallel_files=8,
        license="C4: ODC-BY 1.0 (+ Common Crawl terms); releases (c) their issuers",
        date_evidence="row-level `timestamp` = Common Crawl fetch time (Apr 2019), filtered < 2022-01-01",
        weight=0.20,
        notes="pages under press-release(s)/, /press/, /news-releases/, /pressroom/, /media-releases/, "
        "not on government/UN-type hosts; wire dateline and the About/Contact/SOURCE tail are cut",
    ),
    Source(
        id="c4_about",
        subtype="website_copy",
        id_field="url",
        genre="business_marketing",
        hf_id="allenai/c4",
        builder="json",
        data_files=C4_EN_FILES,
        text_field="text",
        row_filter=c4_about_page,
        preclean=clean_c4_news,
        date_field="timestamp",
        max_rows_per_candidate=600,
        parallel_files=8,
        license="C4: ODC-BY 1.0 (+ Common Crawl terms); pages (c) their owners",
        date_evidence="row-level `timestamp` = Common Crawl fetch time (Apr 2019), filtered < 2022-01-01",
        weight=0.20,
        notes="C4 `en` (every 16th shard) pages at /about(-us)/, /our-story/, /who-we-are/, /company/, "
        "/services/: company and service descriptions",
    ),
    Source(
        id="edgar_business",
        subtype="business_report",
        id_field="filename",
        genre="business_marketing",
        hf_id="eloukas/edgar-corpus",
        builder="json",
        data_files=tuple(f"{y}/test.jsonl" for y in EDGAR_YEARS),
        interleave_files=True,
        parallel_files=4,
        text_field="section_1",
        documents=edgar_section("section_1"),
        preclean=clean_edgar,
        date_field="year",
        date_parser=year_date,
        license="EDGAR-CORPUS: Apache-2.0; 10-K filings are public SEC records",
        date_evidence="row-level `year` = filing year (2001-2020 files used), filtered < 2022",
        weight=0.15,
        notes="10-K Item 1 (Business): company description; headings, tables, contact lines and legal "
        "boilerplate dropped",
    ),
    Source(
        id="edgar_mdna",
        subtype="business_report",
        id_field="filename",
        genre="business_marketing",
        hf_id="eloukas/edgar-corpus",
        builder="json",
        data_files=tuple(f"{y}/validate.jsonl" for y in EDGAR_YEARS),
        interleave_files=True,
        parallel_files=4,
        text_field="section_7",
        documents=edgar_section("section_7"),
        preclean=clean_edgar,
        date_field="year",
        date_parser=year_date,
        license="EDGAR-CORPUS: Apache-2.0; 10-K filings are public SEC records",
        date_evidence="row-level `year` = filing year (2001-2020 files used), filtered < 2022",
        weight=0.10,
        notes="10-K Item 7 (management's discussion and analysis); other filings than edgar_business",
    ),
    # ---- homework answer (2026-11 production category): direct explanatory answers to study questions
    Source(
        id="eli5",
        subtype="eli5",
        id_field="id",
        genre="homework_answer",
        hf_id="HuggingFaceGECLM/REDDIT_comments",
        data_files=geclm_files("explainlikeimfive", 17, range(16)),
        interleave_files=True,  # files are in time order: one stream each spreads the sample over 2011-2021
        parallel_files=4,
        text_field="body",
        documents=reddit_documents,
        row_filter=reddit_answer(min_score=3),
        preclean=clean_reddit,
        date_field="created_utc",
        max_rows_per_candidate=300,
        license="Reddit content via Pushshift; mirror card: no license",
        date_evidence="row-level `created_utc`, filtered < 2022-01-01 (files 0-14 are all earlier, 15 "
        "straddles)",
        weight=0.35,
        notes="r/explainlikeimfive top-level answers with score >= 3 (moderator templates and bots dropped)",
    ),
    Source(
        id="askscience",
        subtype="askscience",
        id_field="id",
        genre="homework_answer",
        hf_id="HuggingFaceGECLM/REDDIT_comments",
        data_files=geclm_files("askscience", 7, range(7)),
        interleave_files=True,
        parallel_files=3,
        text_field="body",
        documents=reddit_documents,
        row_filter=reddit_answer(min_score=3),
        preclean=clean_reddit,
        date_field="created_utc",
        max_rows_per_candidate=300,
        license="Reddit content via Pushshift; mirror card: no license",
        date_evidence="row-level `created_utc`, filtered < 2022-01-01",
        weight=0.20,
        notes="r/askscience top-level answers with score >= 3 (AskScienceModerator and templates dropped)",
    ),
    Source(
        id="se_science",
        subtype="stackexchange",
        genre="homework_answer",
        hf_id=_SE_FLAX,
        builder="text",
        data_files=tuple(f"{s}.stackexchange.com.jsonl.gz" for s in SE_SCIENCE_SITES),
        interleave_files=True,
        text_field="text",
        documents=se_answer,
        license="card: CC BY-NC-SA 4.0; upstream Stack Exchange content CC BY-SA",
        date_evidence="corpus release: data files committed 2021-07-11 from the archive.org dump of "
        "2021-07-01",
        weight=0.25,
        notes="best-voted answers of physics, chemistry, biology, astronomy, earthscience (LaTeX answers are "
        "rejected by the markup filter)",
    ),
    Source(
        id="se_humanities",
        subtype="stackexchange",
        genre="homework_answer",
        hf_id=_SE_FLAX,
        builder="text",
        data_files=tuple(f"{s}.stackexchange.com.jsonl.gz" for s in SE_HUMANITIES_SITES),
        interleave_files=True,
        text_field="text",
        documents=se_answer,
        license="card: CC BY-NC-SA 4.0; upstream Stack Exchange content CC BY-SA",
        date_evidence="corpus release: data files committed 2021-07-11 from the archive.org dump of "
        "2021-07-01",
        weight=0.20,
        notes="best-voted answers of history, philosophy, english, linguistics, economics",
    ),
)


REJECTED: tuple[RejectedSource, ...] = (
    RejectedSource(
        "euclaise/writingprompts",
        "creative",
        "works_not_used",
        'WritingPrompts (Fan et al., 2018) loads, but is PTB-tokenized (`` quotes \'\', "did n\'t", " ,") -- '
        "an artifact detectors would see; replaced by raw r/WritingPrompts comments with timestamps",
    ),
    RejectedSource(
        "euclaise/WritingPromptsX",
        "creative",
        "unsafe",
        "raw r/WritingPrompts text but no per-row date; built Sep 2023 -> may contain post-ChatGPT stories",
    ),
    RejectedSource(
        "monology/pile-uncopyrighted",
        "*",
        "works_not_used",
        "loads (needs zstandard); same documents as the per-subset timaeus/pile-* mirrors used here, which "
        "avoid scanning 30 x 10 GB shards; use it at full scale filtered by meta.pile_set_name",
    ),
    RejectedSource(
        "nlpatunt/D_ASAP-AES",
        "essay",
        "works_not_used",
        "ASAP-AES (2012) loads, but 64% of essays contain @CAPS1/@LOCATION1 anonymization tokens and the "
        "rest skew short (<150 words)",
    ),
    RejectedSource(
        "qwedsacf/ivypanda-essays",
        "essay",
        "unsafe",
        "no per-row date; crawled Jan 2023 (after ChatGPT); essays cite 2021+ sources -> human origin not "
        "guaranteed",
    ),
    RejectedSource(
        "knarasi1/student_and_llm_essays", "essay", "unsafe", "mixes human and LLM essays, prompt-formatted"
    ),
    RejectedSource(
        "baber/paul_graham_essays", "essay", "works_not_used", "single author, ~200 essays, some after 2021"
    ),
    RejectedSource(
        "srky/ICNALE_writing_score", "essay", "works_not_used", "only a few hundred short L2-learner essays"
    ),
    RejectedSource(
        "introvoyz041/PERSUADE_corpus_2.0", "essay", "broken", "contains only PDFs of the corpus paper"
    ),
    RejectedSource(
        "mariolinov/Ellipse", "essay", "broken", "unrelated (image tensors), not the ELLIPSE corpus"
    ),
    RejectedSource(
        "ShashiVish/cover-letter-dataset", "email_letter", "unsafe", "templated/synthetic cover letters"
    ),
    RejectedSource(
        "Yale-LILY/aeslc",
        "email_letter",
        "works_not_used",
        "loads; subset of the same Enron corpus (duplicate source)",
    ),
    RejectedSource(
        "timaeus/pile-enron_emails",
        "email_letter",
        "works_not_used",
        "loads; same Enron corpus with raw headers, corbt/enron-emails has parsed body + date",
    ),
    RejectedSource(
        "McAuley-Lab/Amazon-Reviews-2023",
        "review",
        "broken",
        "loading script (unsupported by datasets>=4); raw jsonl could be read via data_files with a "
        "timestamp < 2022 filter at full scale",
    ),
    RejectedSource("webis/tldr-17", "forum_qa", "broken", "loading script (unsupported by datasets>=4)"),
    RejectedSource(
        "sentence-transformers/eli5",
        "forum_qa",
        "works_not_used",
        "loads; answers are short (7% >= 150 words), no per-row date",
    ),
    RejectedSource(
        "barilan/blog_authorship_corpus",
        "article_blog",
        "broken",
        "loading script; the tasksource mirror (csv) is used instead",
    ),
    RejectedSource(
        "timaeus/pile-pile-cc / allenai/c4 (en) / Skylion007/openwebtext",
        "article_blog",
        "works_not_used",
        "load; generic web text (2019-2020), not genre-pure -- fallback for article_blog at full scale "
        "(C4 realnewslike is used, URL-selected, for op-eds and letters)",
    ),
    RejectedSource(
        "stanford-oval/ccnews",
        "news",
        "works_not_used",
        "loads; 2016-2024 multilingual with `published_date` + `language` -> full-scale news source "
        "(filter en, < 2022)",
    ),
    RejectedSource(
        "timaeus/pile-gutenberg_pg-19",
        "creative",
        "works_not_used",
        "loads; pre-1919 books -> archaic register and transcriber notes, unlike production creative text",
    ),
    RejectedSource(
        "timaeus/pile-wikipedia_en",
        "*",
        "works_not_used",
        "loads; encyclopedic register is not one of the product genres",
    ),
    # ---- Q-11 search (2026-11-09): essay
    RejectedSource(
        "webis/args_me (debate.org part)",
        "essay",
        "works_not_used",
        "87% of args.me; rounds quote the opponent inline (another author's words inside the text), round "
        "boilerplate ('my opponent', 'Con:'), HTML-stripping artifacts; only the edited portals are used",
    ),
    RejectedSource(
        "notaphoenix/debateorg_w_effect_for_liberal (+ _conservative, _subset)",
        "essay",
        "works_not_used",
        "debate.org again (same problems as the args.me debate.org part)",
    ),
    RejectedSource(
        "simonycl/cmv_2017_2025 / idodah/webis-cmv-20-pairs / nailiamirzakhmedova/cmv_op_10k / "
        "JingyangYi/CMV_op",
        "essay",
        "works_not_used",
        "load; subsets of r/changemyview (pairs repeat the same post, some without dates or prompt-"
        "formatted); the full GECLM submissions dump with `created_utc` is used instead",
    ),
    RejectedSource(
        "Siddish/change-my-view-subreddit-cleaned / launch/CMV / QCRI/cmv",
        "essay",
        "works_not_used",
        "chat-formatted (### Human/### Assistant) / test-only json / gated audio corpus",
    ),
    RejectedSource(
        "mmmaurer/winning-arguments-ling-enriched / tjb001/convokit-winning-arguments",
        "essay",
        "broken",
        "feature columns only, no text (merge with the original corpus required) / empty repo",
    ),
    RejectedSource(
        "DriftLogic/Annotated_Persuasive_Essays",
        "essay",
        "unsafe",
        "synthetically generated (LLM) essays, per its card; 5-row sample",
    ),
    RejectedSource(
        "Efstathios/guardian_authorship",
        "essay",
        "broken",
        "loading script; also small (~450 Guardian opinion columns by 13 authors)",
    ),
    RejectedSource(
        "ndtran0101/writing9-ielts-essays",
        "essay",
        "works_not_used",
        "163k IELTS essays crawled 2026; only ~9k have ObjectId timestamps < 2022 (2020-21), license "
        "unknown, L2-learner register already covered by ELLIPSE",
    ),
    RejectedSource(
        "chillies/IELTS-writing-task-2-evaluation / Atharva1244/college_essays",
        "essay",
        "unsafe",
        "no date or provenance (IELTS practice essays with LLM feedback) / ~100 essays",
    ),
    RejectedSource(
        "vblagoje/cc_news (opinion/letters URLs)",
        "essay",
        "works_not_used",
        "only 5.6k opinion and 1.4k letter URLs among 708k articles; C4 realnewslike has ~10x more",
    ),
    # ---- Q-11 search: email / letter
    RejectedSource(
        "fddemarco/pushshift-reddit",
        "email_letter",
        "works_not_used",
        "all-subreddit submissions 2012-2018 incl. r/UnsentLetters (~1k posts/month), but `selftext` is "
        "lowercased with punctuation stripped -- an artifact detectors would see",
    ),
    RejectedSource(
        "Hieuman/gmane_dataset",
        "email_letter",
        "unsafe",
        "mailing-list messages grouped by anonymized author, no dates (Gmane kept archiving after 2022), "
        "multilingual spam mixed in",
    ),
    RejectedSource(
        "dwablimol/r-mailing-lists-raw",
        "email_letter",
        "unsafe",
        "R mailing lists without a date field (r-help runs to 2024); mostly code questions",
    ),
    RejectedSource(
        "HypernetworkRG/email-W3C",
        "email_letter",
        "broken",
        "74 KB hypergraph of the W3C list (sender/recipient ids), no text",
    ),
    RejectedSource(
        "talby/spamassassin",
        "email_letter",
        "broken",
        "loading script that downloads the 2002 corpus from apache.org; nothing streamable on the Hub",
    ),
    RejectedSource(
        "from-our-page/hillary-clinton-emails-wikileaks",
        "email_letter",
        "works_not_used",
        "OCR text of FOIA PDF releases split into lines (State Dept headers, redaction codes), ~9.6k "
        "files mostly forwarded news; no license",
    ),
    RejectedSource(
        "OwnedByDanes/Usenet-Corpus-1980-2013-* / wizzense/Usenet-Corpus-1980-2013",
        "email_letter",
        "works_not_used",
        "full corpus needs a commercial license (free part = samples); Usenet threads are forum "
        "discussion, not letters",
    ),
    RejectedSource(
        "devplex97/usenet-humanities",
        "email_letter",
        "unsafe",
        "Usenet text without dates or provenance (likely derived from the licensed corpus above)",
    ),
    RejectedSource(
        "tim-a-wood/fda-warning-letters / emperor-mew/sec-comment-letters",
        "email_letter",
        "works_not_used",
        "templated regulatory letters (dated, but boilerplate) / card only, data behind an API",
    ),
    RejectedSource(
        "ISTA-DASLab/Panza-emails / eagle0504/warren-buffett-annual-letters-*",
        "email_letter",
        "works_not_used",
        "< 1k emails of three personas / one author, ~45 shareholder letters",
    ),
    RejectedSource(
        "notesbymuneeb/epstein-emails (and mirrors)",
        "email_letter",
        "unsafe",
        "not inspected: correspondence from the 2025 Epstein-estate release (third parties' personal "
        "data); excluded on sensitivity grounds",
    ),
    RejectedSource(
        "Lucius-Morningstar/enron-correspondence-dedup / Hellisotherpeople/enron_emails_parsed",
        "email_letter",
        "works_not_used",
        "the same Enron corpus again",
    ),
    # ---- production re-shape (2026-11): academic full texts
    RejectedSource(
        "timaeus/pile-pubmed_central",
        "academic",
        "works_not_used",
        "Pile PMC (same pandoc format, ~89% of papers yield a window) but only the first 100k Pile "
        "documents, "
        "no ids, no per-row date or license (NC licenses included); common-pile/pubmed is used instead",
    ),
    RejectedSource(
        "timaeus/pile-arxiv / suolyer/pile_arxiv",
        "academic",
        "works_not_used",
        "LaTeX-derived: `$...$` maths in 95% of papers, YAML front matter, stripped \\cite/\\ref leave "
        "broken "
        "sentences ('as in by using Lemma [x]'); only 30% of papers yield a clean window",
    ),
    RejectedSource(
        "common-pile/arxiv_papers",
        "academic",
        "unsafe",
        "`created` is the v1 submission date but the text is the latest version fetched in 2024, so the date "
        "does not bound the text; also hard-wrapped lines and footnote numbers fused into words",
    ),
    RejectedSource(
        "ccdv/arxiv-summarization / ccdv/pubmed-summarization",
        "academic",
        "works_not_used",
        "Cohan et al. (2018) article texts: lowercased and PTB-tokenized with @xcite/@xmath placeholders",
    ),
    RejectedSource(
        "armanc/scientific_papers", "academic", "broken", "loading script only; same data as the ccdv sets"
    ),
    RejectedSource(
        "allenai/peS2o (v2) / laion/Pes2oX-fulltext",
        "academic",
        "works_not_used",
        "load (row `created`, ODC-BY), but no fields of study or per-paper license; the Common Pile copy "
        "(common-pile/peS2o) adds both / parquet repack of the same texts",
    ),
    RejectedSource(
        "claran/modular-s2orc-parquet",
        "academic",
        "works_not_used",
        "S2ORC by field and year range, but empty card, no license, and pre-2000 rows are OCR scans",
    ),
    RejectedSource(
        "timaeus/pile-philpapers",
        "academic",
        "works_not_used",
        "no paragraph breaks in 100/100 documents; 'Page 1 of 6' headers and captions run into the text",
    ),
    RejectedSource(
        "AlgorithmicResearchGroup/s2orc_full / casperhansen/pmc-oa-markdown / "
        "SIRIS-Lab/scilake-fulltext-corpus",
        "academic",
        "unsafe",
        "no date column (s2orc_full also has raw PDF front matter in the text)",
    ),
    RejectedSource(
        "pszemraj/scientific_lay_summarisation-plos-norm",
        "academic",
        "works_not_used",
        "24.8k PLOS papers with `year`, already part of PubMed Central",
    ),
    # ---- production re-shape: essay subtypes (expository, narrative, literary analysis)
    RejectedSource(
        "realbenpope/PERSUADE_manageable",
        "essay",
        "works_not_used",
        "the PERSUADE mirror used before: no prompt column, so explanatory and analysis prompts cannot be "
        "routed; replaced by nlpatunt/D_persuade_2 (same essays + `prompt_name`)",
    ),
    RejectedSource(
        "allenai/hippocorpus",
        "essay",
        "broken",
        "loading script with a manual download from a retired site; the parquet branch holds no data",
    ),
    RejectedSource(
        "HuggingFaceGECLM/REDDIT_submissions (r/tifu)",
        "essay",
        "works_not_used",
        "dated personal narratives (~150k), but adult comic Reddit stories (profanity, TL;DR, Reddit slang), "
        "not school narrative essays; left for an owner decision",
    ),
    RejectedSource(
        "dim/essayforum_raw_writing_10k",
        "essay",
        "works_not_used",
        "row dates, but 52% IELTS/TOEFL argumentative and 20% formulaic IELTS chart reports, ~1% narrative; "
        "markdown prompt headers, feedback requests and ads inside posts",
    ),
    RejectedSource(
        "dim/essayforum_writing_prompts_6k", "essay", "unsafe", "prompt/reply pairs without dates"
    ),
    RejectedSource(
        "trentmkelly/pre-chatgpt-collected-student-writings",
        "essay",
        "unsafe",
        "PDF-extracted school literary magazines (mostly poetry/fiction, several authors per document, page "
        "noise); 40% of words have no year and some issues are from 2022",
    ),
    RejectedSource(
        "trentmkelly/lots-of-essays / artfultom/human-essays / thisisHJLee/human_essay / "
        "danibor/human-essay-*"
        " / polsci/ghostbuster-essay-cleaned / ChristophSchuhmann/essays-with-instructions",
        "essay",
        "unsafe",
        "repacks of ASAP 2.0, PERSUADE, ELLIPSE and IvyPanda (no dates), some paired with AI text or "
        "LLM-written instructions",
    ),
    RejectedSource(
        "ahs10/essays-big5 / Ankonbh/aeon-essays-preprocessed",
        "essay",
        "works_not_used",
        "stream-of-consciousness writing (1997-2004), not essays / magazine essays without date or license",
    ),
    RejectedSource(
        "Xiaochr/Chinese-Student-English-Essay / UniversalCEFR/icle500_en",
        "essay",
        "works_not_used",
        "gaokao letter tasks, median 130 words, no date / 500 argumentative essays, doubtful CC0 (ICLE is "
        "commercially licensed)",
    ),
    RejectedSource(
        "bea2019st/wi_locness / 512duncanl/wi_locness_detokenized / matejklemen/clc_fce",
        "essay",
        "broken",
        "loading scripts or sentence-level tokenized grammar-correction pairs (LOCNESS part is tiny); BAWE, "
        "MICUSP and TOEFL11 are not on the Hub",
    ),
    # ---- production re-shape: application
    RejectedSource(
        "qsardor/viorra-admissions-essays",
        "application",
        "unsafe",
        "scraped 2026 from admissions blogs incl. post-2022 entries, no dates, redacted by an "
        "'AI agent team'",
    ),
    RejectedSource(
        "Ayesha490/personalstatement / tekriwar/Statement_of_Purpose / DharamArora/GAMSAT-Essays",
        "application",
        "unsafe",
        "LLM mentor instruction pairs / Llama-2 fine-tuning data / instruction-formatted marking data",
    ),
    RejectedSource(
        "kwanyick/* / lav-ml / dhruvvaidh / akhan02 / justinthelaw / jenrajaseharan / Bharath1121 "
        "cover letters",
        "application",
        "unsafe",
        "copies or derivatives of the rejected templated ShashiVish set, LLM-rewritten or synthetic letters, "
        "or a handful of templated rows",
    ),
    RejectedSource(
        "anhchanghoangsg/reddit_pushshift_dataset_cleaned (ApplyingToCollege, gradadmissions, ...)",
        "application",
        "works_not_used",
        "row dates, but only ~400 essays among advice threads, usually '>'-quoted inside a request post",
    ),
    # ---- production re-shape: business / marketing
    RejectedSource(
        "McAuley-Lab/Amazon-Reviews-2023 (metadata) and *_2023 mirrors",
        "business_marketing",
        "unsafe",
        "product pages crawled 2023; 'Date First Available' is rare and dates the listing, not the text",
    ),
    RejectedSource(
        "milistu/Amazon_*_2014 / genghaobuaa/Amazon-Review-Data-2018 / huseinzol05/translated-metadata-"
        "amazon-review",
        "business_marketing",
        "works_not_used",
        "2014 metadata overlaps the 2018 items / one category only / Python-repr strings with Malay "
        "translations",
    ),
    RejectedSource(
        "calmgoose/amazon-product-data-2020",
        "business_marketing",
        "works_not_used",
        "'About Product' is pipe-joined bullet points (0.1% usable)",
    ),
    RejectedSource(
        "c3po-ai/edgar-corpus / BEE-spoke-data/edgar-corpus / TeraflopAI/8-K-text",
        "business_marketing",
        "works_not_used",
        "the same EDGAR-CORPUS / 8-K cover pages without the EX-99 press releases",
    ),
    RejectedSource(
        "vblagoje/cc_news (wire-service domains)",
        "business_marketing",
        "works_not_used",
        "only ~3.3k press-wire rows in 708k, 1.3k without a date; C4 press-release URLs give ~10x more",
    ),
    RejectedSource(
        "victor/real-or-fake-fake-jobposting-prediction",
        "business_marketing",
        "works_not_used",
        "EMSCAD job ads (2012-14, CC0): `#URL_<hash>#` tokens and run-together sentences need repair; "
        "a possible later addition",
    ),
    RejectedSource(
        "allenai/c4 en (/products/, /shop/ URLs)",
        "business_marketing",
        "works_not_used",
        "product pages are mostly bullet lists; Amazon descriptions cover product copy",
    ),
    RejectedSource(
        "bitmorse/kickstarter_2022-2021 / james-burton/kick_starter_funding / "
        "svenschultze/inspaire-kickstarter",
        "business_marketing",
        "works_not_used",
        "campaign blurbs of <= 135 characters / instruction-formatted, no date",
    ),
    RejectedSource(
        "FronkonGames/steam-games-dataset / haritzpuerto/steam-product-descriptions / jaykin01/advertisement-"
        "copy / re-mind/synthetic-*",
        "business_marketing",
        "unsafe",
        "descriptions scraped 2023-26 (editable) / LLM-'boosted' rewrites / synthetic ad copy",
    ),
    # ---- production re-shape: homework answers
    RejectedSource(
        "vblagoje/lfqa / Pavithree/eli5 (+ askScience, askHistorians)",
        "homework_answer",
        "works_not_used",
        "ELI5/KILT answers without row dates, the same comments as the GECLM dump (AskHistorians overlaps "
        "forum_qa) / 2022 re-uploads",
    ),
    RejectedSource("defunct-datasets/eli5 / eli5_category", "homework_answer", "broken", "loading scripts"),
    RejectedSource(
        "DataDump1/{askphilosophy, AskEngineers, AskPhysics, askmath, NoStupidQuestions}_reddit",
        "homework_answer",
        "broken",
        "gated with manual approval (403), 2025 uploads without card or license",
    ),
    RejectedSource(
        "HuggingFaceGECLM/REDDIT_comments (science, space, philosophy, history)",
        "homework_answer",
        "works_not_used",
        "link-discussion subreddits, not question/answer",
    ),
    RejectedSource(
        "flax-sentence-embeddings stackexchange (skeptics, law, politics, academia)",
        "homework_answer",
        "works_not_used",
        "47-70% usable but argument or advice rather than study answers",
    ),
)


def get_source(source_id: str) -> Source:
    for s in SOURCES:
        if s.id == source_id:
            return s
    raise KeyError(source_id)


def enabled_sources(ids: list[str] | None = None) -> list[Source]:
    if ids:
        return [get_source(i) for i in ids]
    return [s for s in SOURCES if s.enabled]
