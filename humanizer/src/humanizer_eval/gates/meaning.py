"""G1 v2: meaning preservation with sentence-anchored facts, deterministic shortcuts and arbitration.

Why v2 (report 04b): v1 rejected ~15% of *unchanged* outputs. Causes: the verifier saw a quote without
context, the grounder failed to copy quotes, some extracted facts were wrong, and precision re-extracted
facts from the output (a second noisy extraction). v2:

1. **Facts at period build** (`prepare_facts`): literal atomic facts, each with the ids of the source
   sentences that state it; every fact is then checked against the source itself by both judge families
   and dropped unless both say ``supported``.
2. **Deterministic pass**: a fact whose source sentences all occur unchanged in the output is supported;
   an output sentence that occurs unchanged in the source is supported. No LLM call for those.
3. **Recall** (`recall_verdicts`): the judge sees the whole output and, per fact, the source sentence(s)
   it came from; verdict ``supported`` / ``omitted`` / ``contradicted``.
4. **Precision** (`precision_verdicts`): per output sentence against the whole source; verdict
   ``supported`` / ``embellished`` / ``added`` / ``contradicted`` (no second fact extraction).
5. **Aggregation**: two families judge every open item; where their classes differ, an arbiter decides.
6. **Thresholds as counts** (`G1Config`): omissions <= max(1, 10% of facts); contradictions and added
   specific claims: 0; content-free filler sentences <= max(2, 20% of the input's sentences).

A judge reply that is missing or malformed makes the result ``error`` (re-run), never a miner failure.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from humanizer_eval.gates.judges import _DATA_RULE, _OUTPUT_RULE, LLM, Judge, _ask, _doc, _norm_ws, _obj

PROMPT_VERSION = "g1-v2"

# --- sentences -------------------------------------------------------------------------------

_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "vs", "etc", "e.g", "i.e", "inc", "ltd", "co",
    "corp", "no", "fig", "u.s", "u.k", "a.m", "p.m", "approx", "dept", "est", "mt", "gen", "gov", "sen",
    "rep", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}  # fmt: skip
_END = re.compile(r"[.!?…]+[\"'”’)\]]*\s+")
_NEXT_OK = re.compile(r"[\"'“‘(\[]*[A-Z0-9]")


def split_sentences(text: str) -> list[str]:
    """Deterministic sentence split: paragraphs/lines first, then sentence ends before a capital/digit.

    Common abbreviations ("Dr.", "e.g.", "U.S.") and single initials do not end a sentence. The same
    splitter is used on inputs and outputs, so an unchanged sentence always matches itself.
    """
    out: list[str] = []
    for line in re.split(r"\n+", text.strip()):
        line = line.strip()
        if not line:
            continue
        start = 0
        for m in _END.finditer(line):
            end = m.end()
            if not _NEXT_OK.match(line, end):
                continue
            head = line[start : m.start() + 1].rstrip()
            word = re.search(r"([A-Za-z][A-Za-z.]*)\.$", head)
            if word and (word.group(1).lower() in _ABBREV or len(word.group(1)) == 1):
                continue
            seg = line[start:end].strip()
            if seg:
                out.append(seg)
            start = end
        tail = line[start:].strip()
        if tail:
            out.append(tail)
    return out


def _key(s: str) -> str:
    return _norm_ws(s).strip(" .,;:!?\"'()[]")


_BRACKET_NUM = re.compile(r"\[(\d+)\]")


def _occurs(sentence_key: str, keys: set[str], norm_text: str) -> bool:
    """Unchanged-sentence test: exact sentence match, or a substring match for sentences of >= 4 words
    (a short sentence such as "Yes." would match almost anywhere)."""
    if len(sentence_key) < 3:
        return False
    return sentence_key in keys or (len(sentence_key.split()) >= 4 and sentence_key in norm_text)


def _numbered(sents: Sequence[str]) -> str:
    # a sentence must not be able to fake our [n] markers
    safe = [_BRACKET_NUM.sub(r"(\g<1>)", s) for s in sents]
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(safe))


# --- facts -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Fact:
    text: str
    sents: tuple[int, ...] = ()  # ids of the source sentences that state the fact

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "sents": list(self.sents)}

    @classmethod
    def coerce(cls, f: Fact | dict | str) -> Fact:
        if isinstance(f, Fact):
            return f
        if isinstance(f, str):  # v1 period data: no anchors -> no deterministic shortcut
            return cls(f)
        return cls(str(f["text"]), tuple(int(i) for i in f.get("sents", ())))


FACTS_SCHEMA = {
    "title": "facts",
    **_obj(
        {
            "facts": {
                "type": "array",
                "items": _obj(
                    {"fact": {"type": "string"}, "sentences": {"type": "array", "items": {"type": "integer"}}}
                ),
            }
        }
    ),
}
FACTS_SYSTEM = (
    "You list the atomic content of a document so that a rewrite of it can later be checked against the "
    "list. The document's sentences are numbered [0], [1], ... For each fact give one short self-contained "
    "sentence and the numbers of the document sentences that state it. Rules: one claim per fact (split "
    "compound sentences); stay literal and use the document's own terms; keep numbers, dates, names, "
    "quantities, who did, said or believes what, conditions, causes, contrasts, examples and reasons; an "
    "opinion or argument is a fact too ('The author argues that ...', or as stated by the speaker); resolve "
    "pronouns so each fact stands alone. Do not infer, generalise, merge or add anything. Skip greetings, "
    "sign-offs and pure transitions. Roughly one or two facts per content sentence, at most 25. " + _DATA_RULE
)

VERDICTS_RECALL = ("supported", "omitted", "contradicted")
RECALL_SCHEMA = {
    "title": "recall",
    **_obj(
        {
            "verdicts": {
                "type": "array",
                "items": _obj(
                    {
                        "id": {"type": "integer"},
                        "verdict": {"type": "string", "enum": list(VERDICTS_RECALL)},
                        "evidence": {"type": "string"},
                    }
                ),
            }
        }
    ),
}
RECALL_SYSTEM = (
    "You check whether a TEXT conveys a list of FACTS. Each fact comes with the ORIGINAL sentence(s) it was "
    "taken from, for context only. For each fact answer:\n"
    "- supported: the TEXT states this information, in any wording; it may be split across sentences, "
    "merged with other content, or told in another voice or person ('The author argues X' = 'I think X'). "
    "Small shifts of emphasis are fine.\n"
    "- contradicted: the TEXT states something incompatible with the fact: another number, date, name or "
    "quantity; the opposite claim; a swapped actor, speaker or target; a changed condition, cause or "
    "attribution.\n"
    "- omitted: the TEXT does not convey the fact, or loses an essential element of it (who, what, how "
    "much, when, why).\n"
    "Judge information only, not style or quality. For supported and contradicted, copy the TEXT's evidence "
    "verbatim (short); for omitted leave evidence empty. " + _OUTPUT_RULE + _DATA_RULE
)

VERDICTS_PRECISION = ("supported", "embellished", "added", "contradicted")
PRECISION_SCHEMA = {
    "title": "precision",
    **_obj(
        {
            "verdicts": {
                "type": "array",
                "items": _obj(
                    {
                        "id": {"type": "integer"},
                        "verdict": {"type": "string", "enum": list(VERDICTS_PRECISION)},
                        "note": {"type": "string"},
                    }
                ),
            }
        }
    ),
}
PRECISION_SYSTEM = (
    "You check whether sentences of a REWRITE stay faithful to the ORIGINAL text. The rewrite's sentences "
    "are numbered [0], [1], ... Label each requested sentence:\n"
    "- supported: everything it states is stated or clearly implied by the ORIGINAL (paraphrase, merging, "
    "splitting, another voice or person are fine).\n"
    "- embellished: it adds only colour without new checkable information: emphasis, generic remarks, "
    "transitions, hedges, rhetorical questions, the writer's attitude.\n"
    "- added: it introduces specific information absent from the ORIGINAL: new facts, numbers, dates, "
    "names, places, quotes, studies, examples, causes, consequences or claims about the world.\n"
    "- contradicted: it states something incompatible with the ORIGINAL.\n"
    "Give the most serious label that applies (contradicted > added > embellished > supported) and a short "
    "note naming the added or contradicted content (empty otherwise). " + _OUTPUT_RULE + _DATA_RULE
)


def _parse_verdicts(
    out: dict[str, Any] | None, ids: Sequence[int], allowed: Sequence[str], text_key: str
) -> dict[int, tuple[str, str]] | None:
    if out is None:
        return None
    want = set(ids)
    got: dict[int, tuple[str, str]] = {}
    for v in out.get("verdicts", []):
        if not isinstance(v, dict):
            continue
        try:
            i = int(v.get("id"))
        except (TypeError, ValueError):
            continue
        if i in want and v.get("verdict") in allowed and i not in got:
            got[i] = (str(v["verdict"]), str(v.get(text_key) or "")[:300])
    return got


def recall_verdicts(
    llm: LLM,
    judge: Judge,
    facts: Sequence[Fact],
    source_sents: Sequence[str],
    text: str,
    ids: Sequence[int],
    purpose: str = "recall",
) -> tuple[dict[int, tuple[str, str]] | None, float]:
    """{fact id: (verdict, evidence)} for the facts ``ids`` against ``text``; None on judge failure."""
    if not ids:
        return {}, 0.0
    lines = []
    for i in ids:
        f = facts[i]
        ctx = " ".join(source_sents[k] for k in f.sents if 0 <= k < len(source_sents))
        lines.append(f"{i}. FACT: {f.text}" + (f"\n   ORIGINAL: {ctx}" if ctx else ""))
    user = f"{_doc('facts', chr(10).join(lines))}\n{_doc('text', text)}"
    out, cost = _ask(llm, judge, RECALL_SYSTEM, user, RECALL_SCHEMA, f"{PROMPT_VERSION}:{purpose}")
    return _parse_verdicts(out, ids, VERDICTS_RECALL, "evidence"), cost


def precision_verdicts(
    llm: LLM,
    judge: Judge,
    source: str,
    out_sents: Sequence[str],
    ids: Sequence[int],
    purpose: str = "precision",
) -> tuple[dict[int, tuple[str, str]] | None, float]:
    """{output sentence id: (verdict, note)} for the sentences ``ids``; None on judge failure."""
    if not ids:
        return {}, 0.0
    user = (
        f"{_doc('original', source)}\n{_doc('rewrite', _numbered(out_sents))}\n"
        f"Label these rewrite sentences: {', '.join(str(i) for i in ids)}"
    )
    out, cost = _ask(llm, judge, PRECISION_SYSTEM, user, PRECISION_SCHEMA, f"{PROMPT_VERSION}:{purpose}")
    return _parse_verdicts(out, ids, VERDICTS_PRECISION, "note"), cost


# --- aggregation -----------------------------------------------------------------------------

#: precision labels that are acceptable collapse to one class, so supported-vs-embellished needs no arbiter
_CLASS = {"supported": "ok", "embellished": "ok", "omitted": "omitted", "added": "added"}
_CLASS["contradicted"] = "contradicted"


@dataclass
class Verdict:
    final: str  # class: ok | omitted | added | contradicted
    by: dict[str, str]  # model -> raw label
    note: str = ""
    arbitrated: bool = False
    deterministic: bool = False


def _two_family(
    ask, judges: tuple[Judge, Judge], arbiter: Judge | None, ids: Sequence[int]
) -> tuple[dict[int, Verdict] | None, float, list[str]]:
    """Ask both judges about ``ids`` in parallel; disputed ids go to the arbiter."""
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        (va, ca), (vb, cb) = ex.map(lambda j: ask(j, ids), judges)
    cost = ca + cb
    if va is None or vb is None:
        return None, cost, ["judge call failed"]
    res: dict[int, Verdict] = {}
    disputed: list[int] = []
    for i in ids:
        a, b = va.get(i), vb.get(i)
        by = {judges[0].model: a[0] if a else "missing", judges[1].model: b[0] if b else "missing"}
        if a and b and _CLASS[a[0]] == _CLASS[b[0]]:
            note = a[1] if _CLASS[a[0]] != "ok" else ""
            res[i] = Verdict(_CLASS[a[0]], by, note or (b[1] if _CLASS[b[0]] != "ok" else ""))
        else:
            disputed.append(i)
            res[i] = Verdict("?", by)
    if disputed:
        if arbiter is None:
            return None, cost, [f"{len(disputed)} disputed verdicts and no arbiter"]
        vc, cc = ask(arbiter, disputed)
        cost += cc
        if vc is None:
            return None, cost, ["arbiter call failed"]
        for i in disputed:
            c = vc.get(i)
            if c is None:
                errors.append(f"arbiter gave no verdict for {i}")
                continue
            res[i].by[arbiter.model] = c[0]
            res[i].final = _CLASS[c[0]]
            res[i].note = c[1] if _CLASS[c[0]] != "ok" else ""
            res[i].arbitrated = True
    if errors:
        return None, cost, errors
    return res, cost, []


def _row(v: Verdict) -> dict[str, Any]:
    return {
        "verdict": "supported" if v.final == "ok" else v.final,
        "by": v.by,
        "note": v.note,
        "arbitrated": v.arbitrated,
        "deterministic": v.deterministic,
    }


# --- G1 check --------------------------------------------------------------------------------


@dataclass(frozen=True)
class G1Config:
    max_omitted_frac: float = 0.10
    min_omitted_allowed: int = 1
    max_contradicted: int = 0  # recall + precision
    max_added: int = 0
    #: padding guard: sentences BOTH judges call "embellished" <= max(min_embellished_allowed,
    #: frac * number of SOURCE sentences) -- relative to the input, so padding cannot raise its own cap
    max_embellished_frac: float = 0.20
    min_embellished_allowed: int = 2
    #: quality levels (D-22): a failed output is "minor" (level 0.5) if it has at most this many
    #: contradictions + additions and at most this many omissions over the allowance; else level 0
    minor_max_distortions: int = 1
    minor_extra_omissions: int = 1

    def omitted_allowed(self, n_facts: int) -> int:
        return max(self.min_omitted_allowed, math.floor(self.max_omitted_frac * n_facts + 1e-9))

    def embellished_allowed(self, n_sentences: int) -> int:
        return max(self.min_embellished_allowed, math.ceil(self.max_embellished_frac * n_sentences - 1e-9))


@dataclass
class G1Result:
    status: str  # pass | fail | error
    level: float = 1.0  # 1 pass, 0.5 minor violation, 0 severe (D-22); meaningless on error
    counts: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    sentences: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == "pass"


@dataclass
class G1Plan:
    """What G1 must ask the judges about one (source, output) pair after the deterministic pass."""

    facts: list[Fact]
    src_sents: list[str]
    out_sents: list[str]
    open_facts: list[int]  # fact ids that need judges
    open_sents: list[int]  # output sentence ids that need judges


def plan_g1(source: str, output: str, facts: Sequence[Fact | dict | str]) -> G1Plan:
    facts = [Fact.coerce(f) for f in facts]
    src_sents, out_sents = split_sentences(source), split_sentences(output)
    out_norm, src_norm = _norm_ws(output), _norm_ws(source)
    out_keys, src_keys = {_key(x) for x in out_sents}, {_key(x) for x in src_sents}
    open_f = [
        i
        for i, f in enumerate(facts)
        if not (
            f.sents
            and all(0 <= k < len(src_sents) for k in f.sents)
            and all(_occurs(_key(src_sents[k]), out_keys, out_norm) for k in f.sents)
        )
    ]
    open_s = [i for i, x in enumerate(out_sents) if not _occurs(_key(x), src_keys, src_norm)]
    return G1Plan(facts, src_sents, out_sents, open_f, open_s)


def decide_g1(
    plan: G1Plan, rv: dict[int, Verdict], pv: dict[int, Verdict], config: G1Config | None = None
) -> G1Result:
    """Apply the count thresholds to judge verdicts for the open ids (the rest are deterministic)."""
    cfg = config or G1Config()
    rv, pv = dict(rv), dict(pv)
    for i in range(len(plan.facts)):
        rv.setdefault(i, Verdict("ok", {}, deterministic=True))
    for i in range(len(plan.out_sents)):
        pv.setdefault(i, Verdict("ok", {}, deterministic=True))
    res = G1Result("pass")
    res.facts = [{"id": i, "fact": f.text, **_row(rv[i])} for i, f in enumerate(plan.facts)]
    res.sentences = [{"id": i, "sentence": t, **_row(pv[i])} for i, t in enumerate(plan.out_sents)]
    omitted = sum(v.final == "omitted" for v in rv.values())
    contra = sum(v.final == "contradicted" for v in rv.values()) + sum(
        v.final == "contradicted" for v in pv.values()
    )
    added = sum(v.final == "added" for v in pv.values())
    padded = sum(
        v.final == "ok" and len(v.by) >= 2 and all(lab == "embellished" for lab in v.by.values())
        for v in pv.values()
    )
    pad_allowed = cfg.embellished_allowed(len(plan.src_sents))
    allowed = cfg.omitted_allowed(len(plan.facts))
    res.counts = {
        "facts": len(plan.facts),
        "omitted": omitted,
        "omitted_allowed": allowed,
        "contradicted": contra,
        "added": added,
        "embellished": padded,
        "embellished_allowed": pad_allowed,
        "sentences": len(plan.out_sents),
        "llm_facts": len(plan.open_facts),
        "llm_sentences": len(plan.open_sents),
        "arbitrated": sum(v.arbitrated for v in list(rv.values()) + list(pv.values())),
    }
    if omitted > allowed:
        res.reasons.append(f"omitted {omitted} of {len(plan.facts)} facts (allowed {allowed})")
    if contra > cfg.max_contradicted:
        res.reasons.append(f"{contra} contradicted")
    if added > cfg.max_added:
        res.reasons.append(f"{added} sentences add new specific content")
    if padded > pad_allowed:
        res.reasons.append(f"{padded} sentences are filler without content (allowed {pad_allowed})")
    res.status = "fail" if res.reasons else "pass"
    severe = contra + added > cfg.minor_max_distortions or omitted > allowed + cfg.minor_extra_omissions
    res.level = 1.0 if res.status == "pass" else 0.0 if severe else 0.5
    return res


def run_two_family(
    plan: G1Plan, ask_recall, ask_precision, judges: tuple[Judge, Judge], arbiter: Judge | None
) -> tuple[dict[int, Verdict] | None, dict[int, Verdict] | None, float, list[str]]:
    with ThreadPoolExecutor(max_workers=2) as ex:
        fr = ex.submit(_two_family, ask_recall, judges, arbiter, plan.open_facts) if plan.open_facts else None
        fp = (
            ex.submit(_two_family, ask_precision, judges, arbiter, plan.open_sents)
            if plan.open_sents
            else None
        )
        rv, rc, rerr = fr.result() if fr else ({}, 0.0, [])
        pv, pc, perr = fp.result() if fp else ({}, 0.0, [])
    return rv, pv, rc + pc, rerr + perr


def check_g1(
    llm: LLM,
    judges: tuple[Judge, Judge],
    arbiter: Judge | None,
    source: str,
    output: str,
    facts: Sequence[Fact | dict | str],
    config: G1Config | None = None,
) -> G1Result:
    plan = plan_g1(source, output, facts)

    def ask_recall(j: Judge, ids):
        return recall_verdicts(llm, j, plan.facts, plan.src_sents, output, ids)

    def ask_precision(j: Judge, ids):
        return precision_verdicts(llm, j, source, plan.out_sents, ids)

    rv, pv, cost, errors = run_two_family(plan, ask_recall, ask_precision, judges, arbiter)
    if rv is None or pv is None:
        return G1Result("error", reasons=errors, cost_usd=cost)
    res = decide_g1(plan, rv, pv, config)
    res.cost_usd = cost
    return res


# --- facts at period build ---------------------------------------------------------------------


@dataclass
class FactsReport:
    extracted: int = 0
    kept: int = 0
    dropped: list[dict[str, Any]] = field(default_factory=list)
    uncovered_sentences: list[int] = field(default_factory=list)
    cost_usd: float = 0.0
    error: str | None = None


def extract_facts_v2(llm: LLM, judge: Judge, text: str) -> tuple[list[Fact] | None, float]:
    sents = split_sentences(text)
    out, cost = _ask(
        llm, judge, FACTS_SYSTEM, _doc("source", _numbered(sents)), FACTS_SCHEMA, f"{PROMPT_VERSION}:facts"
    )
    if out is None:
        return None, cost
    facts = []
    for f in out.get("facts", []):
        if not isinstance(f, dict) or not isinstance(f.get("fact"), str) or not f["fact"].strip():
            continue
        ids = tuple(
            sorted({int(k) for k in f.get("sentences", []) if isinstance(k, int) and 0 <= k < len(sents)})
        )
        facts.append(Fact(f["fact"].strip(), ids))
    return facts[:25], cost


def prepare_facts(
    llm: LLM, extractor: Judge, validators: tuple[Judge, Judge], text: str
) -> tuple[list[Fact] | None, FactsReport]:
    """Extract anchored facts and keep only those both validator families find supported by ``text``."""
    rep = FactsReport()
    facts, c = extract_facts_v2(llm, extractor, text)
    rep.cost_usd += c
    if facts is None:
        rep.error = "extraction failed"
        return None, rep
    rep.extracted = len(facts)
    sents = split_sentences(text)
    ids = list(range(len(facts)))
    with ThreadPoolExecutor(max_workers=2) as ex:
        results = list(
            ex.map(lambda j: recall_verdicts(llm, j, facts, sents, text, ids, "validate"), validators)
        )
    rep.cost_usd += sum(c for _, c in results)
    if any(v is None for v, _ in results):
        rep.error = "validation failed"
        return None, rep
    kept = []
    for i, f in enumerate(facts):
        labels = [v.get(i, ("missing", ""))[0] for v, _ in results]
        if all(lab == "supported" for lab in labels):
            kept.append(f)
        else:
            rep.dropped.append({"fact": f.text, "labels": labels})
    rep.kept = len(kept)
    covered = {k for f in kept for k in f.sents}
    rep.uncovered_sentences = [k for k, s in enumerate(sents) if k not in covered and len(s.split()) >= 4]
    return kept, rep
