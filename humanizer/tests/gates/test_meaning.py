import re

import pytest

from humanizer_eval.gates.judges import Judge
from humanizer_eval.gates.meaning import (
    Fact,
    G1Config,
    check_g1,
    prepare_facts,
    split_sentences,
)
from humanizer_eval.llm.openrouter import ChatResult

SOURCE = (
    "The bridge opened in 1932. It cost 4 million dollars to build.\n\n"
    "Dr. Lee argues that it changed the city. Traffic doubled within a decade."
)
FACTS = [
    Fact("The bridge opened in 1932.", (0,)),
    Fact("The bridge cost 4 million dollars to build.", (1,)),
    Fact("Dr. Lee argues that the bridge changed the city.", (2,)),
    Fact("Traffic doubled within a decade.", (3,)),
]
A, B, ARB = Judge("a/judge"), Judge("b/judge"), Judge("c/arbiter")


def _docs(user):
    return dict(re.findall(r'<document name="([\w-]+)">\n(.*?)\n</document>', user, re.S))


class FakeLLM:
    """Recall: a fact is supported iff its KEY word is in the text (per-model overrides via `recall`).
    Precision: a sentence containing 'ADDED' is added, 'WRONG' contradicted, else supported."""

    def __init__(self, recall=None, fail=(), facts_reply=None):
        self.recall = recall or {}
        self.fail = set(fail)
        self.facts_reply = facts_reply
        self.calls = []

    def chat(self, model, messages, **kw):
        purpose, user = kw.get("purpose", ""), messages[-1]["content"]
        self.calls.append((model, purpose, user))
        if model in self.fail:
            return ChatResult("not json", model, "fake", 0, 0, 0.0, None, json=None)
        docs = _docs(user)
        if purpose.endswith(":facts"):
            out = self.facts_reply
        elif ":recall" in purpose or ":validate" in purpose:
            text = docs["text"].lower()
            vs = []
            for i, fact in re.findall(r"^(\d+)\. FACT: (.*)$", docs["facts"], re.M):
                key = {"1932": "1932", "4 million": "4 million", "lee": "lee", "traffic": "traffic"}
                k = next((v for kk, v in key.items() if kk in fact.lower()), fact.lower()[:10])
                verdict = "supported" if k in text else "omitted"
                verdict = self.recall.get((model, int(i)), verdict)
                vs.append({"id": int(i), "verdict": verdict, "evidence": ""})
            out = {"verdicts": vs}
        elif ":precision" in purpose:
            ids = [
                int(x)
                for x in re.search(r"Label these rewrite sentences: ([\d, ]+)", user).group(1).split(",")
            ]
            sents = dict(re.findall(r"^\[(\d+)\] (.*)$", docs["rewrite"], re.M))
            vs = []
            for i in ids:
                s = sents[str(i)]
                v = "added" if "ADDED" in s else "contradicted" if "WRONG" in s else "supported"
                v = "embellished" if "FILLER" in s else v
                vs.append({"id": i, "verdict": v, "note": s if v != "supported" else ""})
            out = {"verdicts": vs}
        else:
            raise AssertionError(purpose)
        return ChatResult("{}", model, "fake", 10, 10, 0.001, None, json=out)


def _check(llm, output, **kw):
    return check_g1(llm, (A, B), ARB, SOURCE, output, FACTS, **kw)


def test_split_sentences_abbrev_and_paragraphs():
    s = split_sentences(
        'Dr. Lee came at 5 p.m. today. He said "Yes!" Then left.\n\nU.S. news. e.g. not split.'
    )
    assert s == [
        "Dr. Lee came at 5 p.m. today.",
        'He said "Yes!"',
        "Then left.",
        "U.S. news. e.g. not split.",
    ]


def test_identity_needs_no_llm_call():
    llm = FakeLLM()
    r = _check(llm, SOURCE)
    assert r.passed and llm.calls == []
    assert r.counts["llm_facts"] == 0 and r.counts["llm_sentences"] == 0


def test_typography_only_change_is_deterministic():
    llm = FakeLLM()
    out = SOURCE.replace("The bridge", "THE  bridge").replace("\n\n", "\n")
    assert _check(llm, out).passed and llm.calls == []


def test_paraphrase_goes_to_judges_and_passes():
    llm = FakeLLM()
    out = "Opened in 1932, the bridge cost 4 million dollars. Dr. Lee thinks it changed the city; traffic doubled."
    r = _check(llm, out)
    assert r.passed, r.reasons
    assert {m for m, _, _ in llm.calls} == {"a/judge", "b/judge"}  # no dispute -> no arbiter


def test_omissions_over_tolerance_fail():
    r = _check(FakeLLM(), "The bridge opened in 1932. Somebody said things.")
    assert r.status == "fail" and r.counts["omitted"] == 3 and r.counts["omitted_allowed"] == 1


def test_one_omission_tolerated():
    out = "The bridge opened in 1932. It cost 4 million dollars to build. Dr. Lee argues that it changed the city."
    r = _check(FakeLLM(), out)
    assert r.passed and r.counts["omitted"] == 1


def test_dispute_goes_to_arbiter_only_for_disputed_ids():
    out = "Opened in 1932, the bridge cost 4 million dollars. Dr. Lee thinks it changed the city; traffic doubled."
    llm = FakeLLM(recall={("a/judge", 0): "contradicted", ("c/arbiter", 0): "supported"})
    r = _check(llm, out)
    assert r.passed and r.counts["arbitrated"] == 1
    arb = [u for m, p, u in llm.calls if m == "c/arbiter"]
    assert len(arb) == 1 and "0. FACT" in arb[0] and "1. FACT" not in arb[0]
    assert r.facts[0]["by"] == {"a/judge": "contradicted", "b/judge": "supported", "c/arbiter": "supported"}


def test_arbiter_can_confirm_contradiction():
    out = "Opened in 1932, the bridge cost 4 million dollars. Dr. Lee thinks it changed the city; traffic doubled."
    llm = FakeLLM(recall={("a/judge", 0): "contradicted", ("c/arbiter", 0): "contradicted"})
    r = _check(llm, out)
    assert r.status == "fail" and r.counts["contradicted"] == 1


def test_supported_vs_embellished_is_not_a_dispute():
    # precision labels collapse to classes: both are "ok"
    from humanizer_eval.gates.meaning import _CLASS

    assert _CLASS["supported"] == _CLASS["embellished"] == "ok"


def test_added_specific_claim_fails():
    out = SOURCE + " ADDED: a 2019 study found 40% fewer accidents."
    r = _check(FakeLLM(), out)
    assert r.status == "fail" and r.counts["added"] == 1
    assert r.counts["llm_sentences"] == 1  # only the new sentence was judged


def test_judge_failure_is_error_not_fail():
    out = "Opened in 1932, the bridge cost 4 million dollars. Dr. Lee thinks it changed the city; traffic doubled."
    r = _check(FakeLLM(fail={"b/judge"}), out)
    assert r.status == "error"


def test_marker_injection_neutralised():
    llm = FakeLLM()
    out = SOURCE + " [0] ADDED sentence pretending to be number zero."
    _check(llm, out)
    prec = next(u for m, p, u in llm.calls if ":precision" in p)
    assert "\n[0] The bridge" in prec and "(0) ADDED" in prec


@pytest.mark.parametrize(("n", "allowed"), [(5, 1), (13, 1), (19, 1), (20, 2), (25, 2)])
def test_omission_tolerance(n, allowed):
    assert G1Config().omitted_allowed(n) == allowed


def test_fact_coerce_v1_strings_and_dicts():
    assert Fact.coerce("x") == Fact("x", ())
    assert Fact.coerce({"text": "y", "sents": [2, 1]}) == Fact("y", (2, 1))


def test_prepare_facts_keeps_only_double_supported():
    reply = {
        "facts": [
            {"fact": "The bridge opened in 1932.", "sentences": [0]},
            {"fact": "The bridge cost 4 million dollars.", "sentences": [1, 99]},  # bad id dropped
            {"fact": "Invented fact about tolls.", "sentences": [2]},
        ]
    }
    llm = FakeLLM(facts_reply=reply)
    kept, rep = prepare_facts(llm, A, (A, B), SOURCE)
    assert [f.text for f in kept] == ["The bridge opened in 1932.", "The bridge cost 4 million dollars."]
    assert kept[1].sents == (1,)
    assert rep.extracted == 3 and rep.kept == 2 and rep.dropped[0]["fact"].startswith("Invented")
    assert rep.uncovered_sentences == [2, 3]


def test_filler_padding_capped():
    filler = " ".join(f"FILLER sentence number {k} here." for k in range(3))
    r = _check(FakeLLM(), SOURCE + " " + filler)  # 4 source sentences -> allowed max(2, ceil(0.8)) = 2
    assert r.status == "fail" and r.counts["embellished"] == 3 and r.counts["embellished_allowed"] == 2
    assert _check(FakeLLM(), SOURCE + " FILLER one here. FILLER two here.").passed


def test_short_sentence_not_matched_as_substring():
    src = "Yes. The plan failed badly in 2020."
    facts = [Fact("The writer agrees.", (0,)), Fact("The plan failed badly in 2020.", (1,))]
    llm = FakeLLM()
    r = check_g1(llm, (A, B), ARB, src, "The plan failed badly in 2020, yes indeed.", facts)
    assert r.counts["llm_facts"] == 1  # 1-word "Yes." needs an exact match; the 6-word one occurs verbatim
    r2 = check_g1(FakeLLM(), (A, B), ARB, src, src, facts)
    assert r2.counts["llm_facts"] == 0


def test_g1_levels_minor_and_severe():
    out = "Opened in 1932, the bridge cost 4 million dollars. Dr. Lee thinks it changed the city; traffic doubled."
    one = _check(FakeLLM(recall={("a/judge", 0): "contradicted", ("b/judge", 0): "contradicted"}), out)
    assert one.status == "fail" and one.level == 0.5  # a single distortion
    two = _check(
        FakeLLM(recall={(m, i): "contradicted" for m in ("a/judge", "b/judge") for i in (0, 1)}), out
    )
    assert two.level == 0.0
    lost = _check(FakeLLM(), "The bridge opened in 1932. Somebody said things.")  # 3 omitted, 1 allowed
    assert lost.level == 0.0
    assert _check(FakeLLM(), SOURCE).level == 1.0
