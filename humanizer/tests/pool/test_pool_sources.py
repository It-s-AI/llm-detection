import os
from datetime import UTC, datetime

import pytest

from humanizer_eval.pool import sources as S
from humanizer_eval.pool.sources import (
    CUTOFF,
    GENRES,
    REJECTED,
    SOURCES,
    arxiv_id_date,
    enabled_sources,
    get_source,
    parse_date,
)


# --------------------------------------------------------------------------- registry integrity
def test_registry_is_well_formed():
    ids = [s.id for s in SOURCES]
    assert len(ids) == len(set(ids))
    for s in SOURCES:
        assert s.genre in GENRES, s.id
        assert s.license.strip() and s.date_evidence.strip(), s.id
        assert s.weight > 0 and s.stride >= 1 and s.oversample >= 1
        assert "/" in s.hf_id
    for r in REJECTED:
        assert r.status in {"works_not_used", "broken", "unsafe"} and r.reason.strip()


def test_every_genre_has_an_enabled_source():
    assert {s.genre for s in enabled_sources()} == set(GENRES)


def test_get_source_and_subset():
    assert get_source("enron").genre == "email_letter"
    assert [s.id for s in enabled_sources(["imdb", "yelp"])] == ["imdb", "yelp"]
    with pytest.raises(KeyError):
        get_source("nope")


# --------------------------------------------------------------------------- date evidence
class _FakeTimestamp:
    def to_pydatetime(self):
        return datetime(2018, 7, 7, 22, 9, 11)


@pytest.mark.parametrize(
    "value,expected",
    [
        (datetime(2001, 5, 14, 23, 39, tzinfo=UTC), datetime(2001, 5, 14, 23, 39, tzinfo=UTC)),
        ("1314593142", datetime(2011, 8, 29, 4, 45, 42, tzinfo=UTC)),
        (1314593142, datetime(2011, 8, 29, 4, 45, 42, tzinfo=UTC)),
        ("2020-12-26 03:38:10.479000+00:00", datetime(2020, 12, 26, 3, 38, 10, 479000, tzinfo=UTC)),
        ("2017-12-11 20:19:05", datetime(2017, 12, 11, 20, 19, 5, tzinfo=UTC)),
        ("14,May,2004", datetime(2004, 5, 14, tzinfo=UTC)),
        (_FakeTimestamp(), datetime(2018, 7, 7, 22, 9, 11, tzinfo=UTC)),
    ],
)
def test_parse_date(value, expected):
    assert parse_date(value) == expected


@pytest.mark.parametrize("value", [None, "", "not a date", "14,mai,2004"])
def test_parse_date_unparseable(value):
    assert parse_date(value) is None


def test_arxiv_ids_encode_month():
    assert arxiv_id_date("0704.0001") == datetime(2007, 4, 1, tzinfo=UTC)
    assert arxiv_id_date("hep-th/9901001") == datetime(1999, 1, 1, tzinfo=UTC)
    assert arxiv_id_date("2112.01234") == datetime(2021, 12, 1, tzinfo=UTC)
    assert arxiv_id_date("2201.00001") >= CUTOFF
    assert arxiv_id_date("garbage") is None


def test_row_level_date_filter_is_strict():
    enron = get_source("enron")
    assert enron.check_date({"date": datetime(2001, 1, 1, tzinfo=UTC)}) == (True, "row")
    assert enron.check_date({"date": datetime(2021, 12, 31, 23, 59, tzinfo=UTC)}) == (True, "row")
    assert enron.check_date({"date": datetime(2022, 1, 1, tzinfo=UTC)}) == (False, "date_too_late")
    assert enron.check_date({"date": None}) == (False, "date_missing")
    assert enron.check_date({}) == (False, "date_missing")


def test_dataset_level_evidence_passes():
    assert get_source("persuade2").check_date({"full_text": "x"}) == (True, "dataset")


# --------------------------------------------------------------------------- per-source cleaning
def test_stackexchange_thread_is_split_into_posts():
    row = {"text": "Q:\n\nHow do I do X?\n\nA:\n\nLike this.\n\nA:\n\nOr like that."}
    assert S.stackexchange_posts(row) == ["How do I do X?", "Like this.", "Or like that."]


def test_enron_keeps_only_new_text():
    body = (
        "Thanks, I will review it tomorrow.\n\n"
        "-----Original Message-----\nFrom: Someone\nSent: Monday\nTo: Me\nSubject: plan\n\nOld text here."
    )
    assert S.clean_enron(body).strip() == "Thanks, I will review it tomorrow."
    fwd = "See below.\n---------------------- Forwarded by Ben Rogers/HOU/ECT on 10/19/2000 03:13 PM ---\nold"
    assert S.clean_enron(fwd).strip() == "See below."
    notes = "My comments.\n\nPaul Radous@ENRON\nDplflan@aol.com on 10/18/2000 06:18:51 PM\nTo: Ben\nold"
    assert "06:18:51" not in S.clean_enron(notes)
    sep = "Here is the latest from Joe.\n//////////////////////\nSAINTS EXPECT WILLIAMS TO START"
    assert S.clean_enron(sep).strip() == "Here is the latest from Joe."
    assert S.enron_internal_sender({"from": "Phillip.Allen@enron.com"}) is None
    assert S.enron_internal_sender({"from": "news@ft.com"}) == "external_sender"


def test_reddit_markdown_and_quotes():
    body = "&gt; quoted text\nI **really** think [this link](http://x.com/a) is fine &amp; good."
    assert S.clean_reddit(body) == "I really think this link is fine & good."
    assert S.reddit_documents({"body": "[deleted]"}) == []
    assert S.reddit_documents({"body": "hi", "author": "AutoModerator"}) == []


def test_writingprompts_takes_top_level_stories_only():
    story = {"body": "Once upon a time.", "parent_id": "t3_abc", "author": "a"}
    reply = {"body": "Nice story!", "parent_id": "t1_def", "author": "b"}
    assert S.writingprompts_story(story) == ["Once upon a time."]
    assert S.writingprompts_story(reply) == []


def test_hacker_news_html_and_quotes():
    text = "&gt; their words<p>My <i>own</i> reply &amp; more."
    assert S.clean_html_comment(text).strip() == "My own reply & more."


def test_misc_precleans():
    assert S.clean_imdb("Good.<br /><br />Bad.") == "Good.\n\nBad."
    assert "urlLink" not in S.clean_blog("see urlLink this")
    assert S.unwrap_lines("  line one\n  line two\n") == "line one line two"
    assert S.drop_first_line("Title of paper.\nAbstract text.") == "Abstract text."
    assert S.clean_cnn_dm("By . Daily Mail Reporter . PUBLISHED: . 12:00 EST") == ""
    assert S.clean_cnn_dm("LONDON (CNN) -- Text. Watch the video » . More text.") == "Text. More text."
    assert S.medium_documents({"title": "My Title", "text": "My Title\n\nBody."}) == ["Body."]


# --------------------------------------------------------------------------- Q-11: essay / letter sources
def test_essay_and_letters_have_several_sources():
    by_genre = {g: [s.id for s in enabled_sources() if s.genre == g] for g in ("essay", "email_letter")}
    assert {"cmv", "c4_opinion", "argsme", "persuade2", "ellipse"} <= set(by_genre["essay"])
    assert {"enron", "c4_letters"} <= set(by_genre["email_letter"])
    essay = [s for s in enabled_sources() if s.genre == "essay"]
    school = sum(s.weight for s in essay if s.id in {"persuade2", "ellipse"})
    assert school / sum(s.weight for s in essay) < 0.5  # school essays are no longer the whole genre


def test_cmv_cuts_edits_and_moderator_footnote():
    body = (
        "Taxes on sugar work because prices change habits.\n\nPeople buy less soda when it costs more.\n\n"
        "EDIT: Thanks everyone, delta awarded to the first comment.\n\nMore edit text.\n\n_____\n\n"
        "> *Hello, users of CMV! This is a footnote from your moderators.*"
    )
    out = S.clean_cmv(body)
    assert out.strip().endswith("People buy less soda when it costs more.")
    assert "EDIT" not in out and "footnote" not in out
    top = "Edit: fixed the title.\n\nI think cities should ban cars downtown.\n\nEdit 2: typo"
    assert S.clean_cmv(top).strip() == "I think cities should ban cars downtown."
    assert S.clean_cmv("I **really** think [this](http://x.com) matters.") == "I really think this matters."
    assert S.cmv_documents({"selftext": "[removed]", "author": "a"}) == []
    assert S.cmv_documents({"selftext": "My view.", "author": "a"}) == ["My view."]


def test_cmv_uses_row_level_reddit_timestamps():
    cmv = get_source("cmv")
    assert cmv.check_date({"created_utc": "1483162169"}) == (True, "row")  # 2016-12-31
    assert cmv.check_date({"created_utc": "1641000000"}) == (False, "date_too_late")  # 2022-01-01
    assert cmv.check_date({"created_utc": None}) == (False, "date_missing")


def test_argsme_keeps_edited_portals_and_dates_by_crawl_time():
    def row(url, when=datetime(2019, 4, 18, 13, 32, 5)):
        return {
            "premises": [{"text": "An argument.", "stance": "CON"}],
            "context": {"sourceUrl": url, "acquisitionTime": when},
        }

    assert S.argsme_curated_portal(row("https://www.debate.org/debates/X/1/")) == "debate_org_round"
    assert S.argsme_curated_portal(row("http://www.idebate.org/debatabase/x")) is None
    assert S.argsme_curated_portal(row("https://debatewise.org/123-x/")) is None
    assert S.argsme_curated_portal(row("")) == "unknown_portal"
    assert S.argsme_documents(row("x")) == ["An argument."]
    assert S.argsme_documents({"premises": []}) == []
    src = get_source("argsme")
    assert src.builder == "json" and src.check_date(row("x")) == (True, "row")
    assert src.check_date(row("x", when=datetime(2022, 2, 1))) == (False, "date_too_late")
    assert src.check_date({"context": None}) == (False, "date_missing")


def test_argsme_cleaning_repairs_quotes_joins_and_drops_references():
    text = 'Jefferson"s view was clear.It don"t matter [1] much.\n[1] Smith, J. A book, 1998.'
    out = S.clean_argsme(text)
    assert "Jefferson's" in out and "don't" in out
    assert "clear.\n\nIt" in out
    assert "[1]" not in out and "Smith" not in out
    assert S.clean_argsme("U.S.A is fine. e.g.It stays.") == "U.S.A is fine. e.g.It stays."


def test_c4_letter_and_opinion_url_selection():
    letter = "word " * 200
    ok = {"url": "https://www.bnd.com/opinion/letters-to-the-editor/article153428314.html", "text": letter}
    assert S.c4_letter_page(ok) is None
    assert S.c4_opinion_page(ok) == "letter_url"  # letters belong to email_letter, not essay
    assert S.c4_letter_page({**ok, "text": "word " * 400}) == "page_too_long"  # likely several letters
    roundups = [
        "https://www.whittierdailynews.com/2009/10/05/tuesday-letters-to-the-editor-2/",
        "https://www.ttnews.com/categories/letter-0?page=2",
        "http://www.stcatharinesstandard.ca/2017/12/11/letters-to-the-editor",
        "http://thenews.org/2017/02/14/letter-to-the-editor-2-16-17/",
    ]
    for url in roundups:
        assert S.c4_letter_page({"url": url, "text": letter}) == "listing_url", url
    assert S.c4_letter_page({"url": "https://example.com/news/city-council", "text": letter}) == (
        "not_letter_url"
    )
    oped = {"url": "https://www.nytimes.com/2018/02/02/opinion/patriots-fan.html", "text": letter}
    assert S.c4_opinion_page(oped) is None and S.c4_letter_page(oped) == "not_letter_url"
    assert S.c4_opinion_page({"url": "https://x.com/sports/game-recap", "text": letter}) == "not_opinion_url"
    assert get_source("c4_letters").check_date({"timestamp": "2019-04-26T14:32:22Z"}) == (True, "row")


def test_c4_cleaning_drops_widgets_standfirst_and_signatures():
    body = " ".join(["The council should fund the library before anything else."] * 20)
    page = "\n".join(
        [
            "Morning high of 63F with temps falling to near 50. Winds W at 10 to 20 mph. Chance of rain 80%.",
            "Writer feels that the library deserves money.",
            body,
            "Second paragraph of the argument, which matters too.",
            "Jane Doe is a retired teacher who lives in Springfield.",
        ]
    )
    out = S.clean_c4_news(page)
    assert out.startswith("The council") and out.endswith("matters too.")
    assert "Chance of rain" not in out and "Writer feels" not in out
    # a short first paragraph that is not a desk summary stays: the excerpt must start at the start
    opening = "It's no easy job, but somebody has to do it."
    assert S.clean_c4_news(opening + "\n" + body).startswith(opening)
    letter = "\n".join(
        ["Letter: Are we truly free?", "I disagree with the editorial.", body, "John Smith, Springfield."]
    )
    out = S.clean_c4_news(letter)
    assert out.startswith("I disagree") and "John Smith" not in out and "Letter:" not in out
    assert S.clean_c4_news("By then it was over.\nThe writer is a nurse in Ohio.") == "By then it was over."
    credit = "A version of this article appeared in the print edition of The Daily Star on page 7."
    assert S.clean_c4_news(credit + "\nYou&apos;re right, it&apos;s late.") == "You're right, it's late."
    interview = {"url": "https://www.emirates247.com/eb247/opinion/q-a/i-have-regrets", "text": "x"}
    assert S.c4_opinion_page(interview) == "interview_url"  # two voices, not one author's essay


# --------------------------------------------------------------------------- 2026-11 production re-shape
def _para(n: int, offset: int = 0) -> str:
    from pool_helpers import prose

    return prose(n, offset)


def test_new_genres_have_sources_and_the_mix_is_declared():
    by_genre = {g: [s for s in enabled_sources() if s.genre == g] for g in GENRES}
    assert {"application", "business_marketing", "homework_answer"} <= set(GENRES)
    assert [s.id for s in by_genre["application"]] == ["essayforum"]
    for g in ("essay", "academic", "application", "business_marketing", "homework_answer"):
        assert all(s.subtype for s in by_genre[g]), g  # the within-genre mix is reported by subtype
    academic = by_genre["academic"]
    abstracts = sum(s.weight for s in academic if s.subtype == "abstract") / sum(s.weight for s in academic)
    assert abstracts == pytest.approx(0.12)  # production: ~12% abstracts, ~88% paper sections
    essay_subtypes = {s.subtype for s in by_genre["essay"]}
    assert essay_subtypes == {"expository", "argumentative", "narrative", "literary"}
    ids = [s.id for s in by_genre["essay"]]
    assert ids.index("persuade2_expository") < ids.index("cmv")  # scarce subtypes have dedupe priority


def test_json_line_rows():
    pmc = get_source("pmc_papers")
    row = pmc.parse_row({"text": '{"id": "PMC1", "created": "2021-8-13"}'})
    assert pmc.builder == "text" and row == {"id": "PMC1", "created": "2021-8-13"}
    assert pmc.parse_row({"text": "[1, 2]"}) is None and pmc.parse_row({"text": "{broken"}) is None
    assert get_source("imdb").parse_row({"text": "x"}) == {"text": "x"}


def test_parse_date_handles_unpadded_iso_dates():
    assert parse_date("2003-8-18") == datetime(2003, 8, 18, tzinfo=UTC)
    assert parse_date("2021-11-03 00:00:00") == datetime(2021, 11, 3, tzinfo=UTC)
    assert get_source("pmc_papers").check_date({"created": "2003-8-18"}) == (True, "row")
    assert get_source("pmc_papers").check_date({"created": "2022-1-3"}) == (False, "date_too_late")


# ---- academic: paper paragraphs
def test_paper_paragraph_rules():
    assert S.paper_paragraph_ok(_para(120))
    assert not S.paper_paragraph_ok("Introduction")  # heading
    assert not S.paper_paragraph_ok("Far too short. To be a paragraph.")
    assert not S.paper_paragraph_ok(_para(400))  # would have to be cut mid-paragraph
    assert not S.paper_paragraph_ok("Figure 2. " + _para(60))  # caption
    assert not S.paper_paragraph_ok("2. " + _para(60))  # numbered list item
    assert not S.paper_paragraph_ok(_para(60) + " where $x_i$ is the mean.")  # maths
    assert not S.paper_paragraph_ok(_para(60) + " and it continues without an end")
    assert not S.paper_paragraph_ok(_para(60) + " See Smith [12] for details.")  # leftover markup
    stats = " ".join(f"Group {i} had 12.{i} and 3{i}% of 4{i} (p < 0.0{i})." for i in range(1, 9))
    assert not S.paper_paragraph_ok(stats)  # statistics dump
    assert not S.paper_paragraph_ok(_para(60).replace(". ", ".The ", 1))  # PDF sentence join


def test_pandoc_inline_citations():
    t = r"Myelin is produced by glial cells \[[@B1-x],[@B2-x]\]. It is abundant ([@B3]) in CNS."
    assert S.clean_pandoc_inline(t) == "Myelin is produced by glial cells. It is abundant in CNS."
    fig = 'As shown in Figure [1](#fig1){ref-type="fig"}, CO~2~ rose by *x* units.'
    assert S.clean_pandoc_inline(fig) == "As shown in Figure 1, CO2 rose by x units."
    assert S.clean_pandoc_inline("This was shown in [@B4] for mice.") is None  # citation used as a noun
    assert S.clean_pandoc_inline("As Smith et al. [@B5] found, it works.") is None  # year lost upstream


def test_grobid_inline_citations():
    assert S.clean_grobid_inline("Obesity is linked to cancer [2][3][4]. It is common [6, 7].") == (
        "Obesity is linked to cancer. It is common."
    )
    assert (
        S.clean_grobid_inline("It was presented.The results hold.") == "It was presented. The results hold."
    )
    assert S.clean_grobid_inline("This was reported in [12] before.") is None
    assert S.clean_grobid_inline("We measured the magnetization 21. It rose.") is None  # superscript cite


def test_pandoc_blocks_keep_only_body_sections():
    intro = [_para(110, k) for k in range(3)]
    doc = "\n\n".join(
        [
            "pmc",
            "Data availability: on request.",  # before the first heading: front matter
            "1. Introduction {#sec1}\n===============",
            *intro,
            "Results {#sec2}\n=======",
            _para(110, 5),  # results are skipped
            "Methods\n=======",
            "Participants\n------------",
            _para(110, 6),
            "######",  # table title holder: the next block is the table/figure
            _para(110, 7),
            "References\n==========",
            _para(110, 8),
        ]
    )
    blocks = S.pandoc_blocks(doc)
    body = [b for b in blocks if b is not None]
    assert body == [*intro, _para(110, 6)]
    i = blocks.index(intro[2])
    assert blocks[i + 1] is None  # the Results heading breaks the run
    wins = S.paragraph_windows(blocks, "paper")
    assert wins and all(len(w.split("\n\n")) <= 3 for w in wins)
    assert not any(_para(110, 6) in w and intro[2] in w for w in wins)


def test_grobid_blocks_skip_title_abstract_and_back_matter():
    doc = (
        "A Title Of The Paper\n\n"
        + _para(90, 1)  # abstract
        + "\n\n\nIntroduction\n"
        + _para(100, 2)
        + "\n"
        + _para(100, 3)
        + "\n\nResults\n"
        + _para(100, 4)
        + "\n\nDiscussion\n"
        + _para(100, 5)
        + "\n\nConflicts of interest\nThe authors declare that they have no competing interests at all here."
    )
    body = [b for b in S.grobid_blocks(doc) if b is not None]
    assert body == [_para(100, 2), _para(100, 3), _para(100, 5)]


def test_paragraph_windows_are_consecutive_body_paragraphs():
    paras = [_para(120, k) for k in range(6)]
    blocks = [*paras[:3], None, *paras[3:]]  # a heading/caption between paragraphs 3 and 4
    wins = S.paragraph_windows(blocks, "paper-1")
    assert 1 <= len(wins) <= S.PAPER_WINDOWS_PER_DOC
    assert wins == S.paragraph_windows(blocks, "paper-1")  # seeded
    for w in wins:
        parts = w.split("\n\n")
        assert 1 <= len(parts) <= 3 and 150 <= len(w.split()) <= 500
        i = paras.index(parts[0])
        assert parts == paras[i : i + len(parts)]  # consecutive in the paper
        assert not (paras[2] in parts and paras[3] in parts)  # never across the break
    starts = {w.split("\n\n")[0] for k in range(30) for w in S.paragraph_windows(blocks, f"p{k}", n=1)}
    assert len(starts) >= 3  # the start position varies between papers
    assert S.paragraph_windows([_para(60), None, _para(60)], "short") == []  # no run reaches 150 words


def test_pes2o_field_and_year_filter():
    stem = S.pes2o_filter(S.PES2O_STEM_FIELDS)

    def row(fields, year=2019):
        return {"metadata": {"s2fieldsofstudy": fields, "year": year}}

    assert stem(row(["Physics"])) is None
    assert stem(row(["Psychology"])) == "other_field"
    assert stem(row(["Chemistry", "Medicine"])) == "biomedical_field"
    assert stem(row(["Physics"], 2022)) == "date_too_late"
    assert stem(row(["Physics"], None)) == "year_missing"
    assert S.pes2o_filter(S.PES2O_SOCIAL_FIELDS)(row(["Education"])) is None
    src = get_source("pes2o_social")
    assert src.check_date({"created": "2019-07-03T00:00:00.000"}) == (True, "row")
    assert src.check_date({"created": "0001-01-01T00:00:00.000"}) == (False, "date_missing")  # placeholder
    assert src.check_date({"created": "1970-01-01T00:00:00.000"}) == (False, "date_missing")


# ---- essay subtypes
def test_persuade_prompt_routing_and_title_line():
    expo, analysis, arg = (get_source(i) for i in ("persuade2_expository", "persuade2_analysis", "persuade2"))
    for name, ok in [("Car-free cities", expo), ("Exploring Venus", analysis), ("Phones and driving", arg)]:
        for src in (expo, analysis, arg):
            assert (src.row_filter({"prompt_name": name}) is None) == (src is ok), (name, src.id)
    assert (
        S.drop_title_line("The Debate of the Faces\nA new technology is here.") == "A new technology is here."
    )
    assert S.drop_title_line("I think so.\nMore text.") == "I think so.\nMore text."


def test_asap_sets_and_cp1252_quotes():
    lit = get_source("asap_literary")
    assert lit.row_filter({"essay_set": "4"}) is None and lit.row_filter({"essay_set": 6}) == "other_set"
    assert get_source("asap_narrative").row_filter({"essay_set": "8"}) is None
    assert dict(lit.builder_kwargs) == {"sep": "\t", "encoding": "latin-1"} and lit.interleave_files
    assert S.fix_cp1252("the men\x92s help, \x93town\x94") == "the men’s help, “town”"


# ---- application
def test_essayforum_cleaning():
    essay = "I was twelve when we moved. The house was small. My sister cried for a week."
    raw = (
        "Please give feedback, about anything about the essay. Thank you very much. Essay starts after this "
        "line. " + essay + " Any suggestions would be greatly appreciated."
    )
    assert S.clean_essayforum(raw) == essay
    prompt = "Tell us about a personal quality that is important to you. " + essay
    assert S.clean_essayforum(prompt) == essay
    sep = "Any help would be appreciated.------------ ----I ascended the knoll.It landed right."
    assert S.clean_essayforum(sep) == "I ascended the knoll. It landed right."
    assert S.essayforum_row({"Cleaned Essay": "Any suggestions andor advice"}) == "slash_merged"
    assert S.essayforum_row({"Cleaned Essay": essay}) is None


# ---- business / marketing
def test_amazon_description_cleaning():
    assert S.amazon_description({"description": ["Para one.", "Para two."]}) == ["Para one.\n\nPara two."]
    assert S.amazon_description({"description": ""}) == []
    raw = "Great &amp; light.<br><br>Choose a side:*The Healers fight.*The Lords attack."
    assert S.clean_amazon(raw) == "Great & light.\n\nChoose a side:\nThe Healers fight.\nThe Lords attack."


def test_press_release_selection_and_cleaning():
    assert S.c4_press_page({"url": "https://www.prnewswire.com/news-releases/x-300.html"}) is None
    assert S.c4_press_page({"url": "https://example.com/news/x"}) == "not_press_url"
    for gov in (
        "https://www.fbi.gov/news/press-releases/x",
        "https://www.un.org/press/en/2018/bio5153.doc.htm",
    ):
        assert S.c4_press_page({"url": gov}) == "public_sector"
    assert S.c4_press_page({"url": "https://www.amnestyusa.org/press-releases/x"}) is None
    body = "Acme Corp. today announced a new widget for small farms."
    wire = "EL DORADO, Ark., April 15, 2019 /PRNewswire/ -- " + body + "\nAbout Acme\nAcme makes widgets."
    assert S.clean_press(wire) == body
    plain = "MONTREAL (April, 2018) – " + body + "\nMedia contact: Jane"
    assert S.clean_press(plain) == body
    comtex = "Vancouver, British Columbia, Sep 04, 2018 (FSCWire via Comtex) - Vancouver (FSCwire) - " + body
    assert S.clean_press(comtex) == body
    assert S.clean_press("CALGARY, AB--(Marketwired - February 15, 2017) - " + body) == body
    prose_dash = "The company (the wire maker) - said today it had grown in every one of its markets."
    assert S.clean_press(prose_dash) == prose_dash
    assert S.c4_about_page({"url": "http://www.herefordquarries.co.uk/about-us/"}) is None
    assert S.c4_about_page({"url": "https://x.com/services/data-migration/"}) is None
    assert S.c4_about_page({"url": "https://x.com/blog/post-1"}) == "not_about_url"


def test_edgar_cleaning_and_year():
    text = "\n".join(
        [
            "ITEM 1. BUSINESS",
            "General",
            "MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS.",
            "First Capital Bancorp, Inc. is a bank holding company that was incorporated under Virginia law.",
            "Our principal executive offices are located in Glen Allen, and our telephone number is listed.",
            "This discussion should be read in conjunction with our consolidated financial statements.",
            "Net revenue 2010 2009 2008 12,000 11,000 10,500 9,000 8,500 7,000.",
            "The bank offers a full range of banking services to small and medium-sized businesses.",
        ]
    )
    assert S.clean_edgar(text) == (
        "First Capital Bancorp, Inc. is a bank holding company that was incorporated under Virginia law.\n\n"
        "The bank offers a full range of banking services to small and medium-sized businesses."
    )
    assert S.year_date("2008") == datetime(2008, 1, 1, tzinfo=UTC)
    assert S.year_date("n/a") is None
    assert get_source("edgar_business").check_date({"year": "2020"}) == (True, "row")


# ---- homework answers
def test_reddit_answer_filter():
    f = S.reddit_answer(min_score=3)
    ok = {"parent_id": "t3_abc", "author": "someone", "body": "Because heat rises.", "score": "5"}
    assert f(ok) is None
    assert f({**ok, "parent_id": "t1_xyz"}) == "not_top_level"
    assert f({**ok, "author": "AskScienceModerator"}) == "bot"
    assert f({**ok, "body": "Your submission has been removed for the following reason(s)"}) == "mod_template"
    assert f({**ok, "score": "2"}) == "low_score"
    assert get_source("eli5").data_files[0].startswith("data/explainlikeimfive-00000-of-00017-")


def test_stackexchange_answer_pairs():
    import json

    row = {
        "text": json.dumps(
            ["How do vaccines work?", "They train &quot;memory&quot; cells.\n    quoted code\nDone."]
        )
    }
    assert S.se_answer(row) == ['They train "memory" cells.\nDone.']
    assert S.se_answer({"text": "not json"}) == []
    assert get_source("se_science").builder == "text"


# --------------------------------------------------------------------------- live (network) checks
@pytest.mark.live
@pytest.mark.parametrize("source", [s.id for s in enabled_sources()])
def test_live_source_streams(source):
    """Stream a few rows from the Hub and check the text field and date evidence (needs network)."""
    from itertools import islice

    os.environ.setdefault("HF_HOME", "var/hf-cache")
    from humanizer_eval.config import get_settings
    from humanizer_eval.pool.build import iter_rows

    src = get_source(source)
    raw = islice(iter_rows(src, get_settings().hf_api_key or None), 20)
    rows = [r for r in map(src.parse_row, raw) if r]
    assert rows, source
    assert any(src.row_documents(r) for r in rows)
    if src.date_field:
        assert any(src.row_date(r) is not None for r in rows)
