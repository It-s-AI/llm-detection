# 02. Human pool (dev scale) — report

Design: [`docs/design/02-pool.md`](../design/02-pool.md). Code: `src/humanizer_eval/pool/`, build:
`scripts/build_pool_dev.py`, tests: `tests/pool/`. Not a single paid API was called (only free
downloads from Hugging Face and local embeddings); the spending log was not touched.

## What was done (in plain language)

1. **Corpus registry** (`pool/sources.py`): 16 working sources (after Q-11 — 20, see the last section) on Hugging Face for all 8 product genres,
   each with a license and **proof that the texts were written before 2022** (a per-row date where one
   exists: 8 of 16 sources). It also holds 21 checked and rejected candidates with reasons.
2. **Cleaning** (`pool/clean.py`): English only, no code/markup/LaTeX, no anonymization placeholders,
   no PII (e-mails, phone numbers, URLs, nicknames — such texts are **discarded**, not masked), no boilerplate
   site lines; exact and near-duplicates are removed across the whole pool; long documents are cut into contiguous
   excerpts along paragraph boundaries, 150–350 words (~230 on average).
3. **Metadata** (`pool/metadata.py`): a stable `text_id` (hash of the text), genre, source, length bin,
   formal/informal register (a simple deterministic heuristic; the formula is in the code and below).
4. **Clusters** (`pool/cluster.py`): within each genre — topics via local TF-IDF+SVD embeddings and
   k-means, each cluster no smaller than a threshold; the cluster label is 3 common words with no names, titles or digits;
   automatic check of sizes and labels.
5. **Storage and sampling** (`pool/store.py`): the pool in parquet; `sample_slots` returns **only the metadata**
   of slots (D-51), `sample_reference` returns human texts with the same counts per stratum (D-54);
   both functions are deterministic by seed and verifiable without numpy (sha256 ranking).

## Mapping to decisions

| Decision | Where in code | Verified by | Status |
|---|---|---|---|
| D-50 pool before 2022, guaranteed human | `sources.py` (`Source.check_date`, `CUTOFF`, `REJECTED`) | `test_pool_sources.py` (strict date filter, date parsers, arXiv id), 16 (+4 in Q-11) live download tests | ✅ dev 24 000; **full Q-11 build: 239 431 texts** (not millions) |
| D-51 AI inputs depend on the pool only through metadata; cluster ≥ 300 | `store.sample_slots` → `Slot(genre, cluster_id, cluster_label, length_bin, register)`; `cluster.cluster_genre(min_size=…)`, `check_clusters` | `test_sample_slots_never_returns_text_or_ids`, `test_clusters_respect_min_size…`, `check_label` | ✅ (threshold 50 in dev; **full build: 300, 259 clusters, min. 305**) |
| D-54 human reference set H with the same counts per group | `store.sample_reference` (+ `exclude_ids` for the cache of past periods) | `test_sample_reference_matches_stratum_counts_exactly`, `…relaxes_when_needed`, `…exact_matches_are_served_first` | ✅ |
| D-55 length 150–300, ~220 on average | `clean.excerpt`, `FilterConfig(min_words, max_words, target_mean_words)` | `test_excerpts_are_contiguous…`, `test_excerpt_determinism_and_target_mean` | 🟡 done as 150–350 / ~230 per the task, a parameter |
| D-59 English only | `clean.is_english` (langid, p ≥ 0.9) | `test_language_filter` | ✅ |
| D-57 determinism by seed | `store.rank_key`, `allocate` | `test_sample_slots_is_deterministic…`, `test_sample_reference_is_deterministic…` | ✅ |
| D-90 embeddings via OpenRouter | protocol `cluster.Embedder`; in dev — local `TfidfSvdEmbedder` | `test_custom_embedder_can_be_plugged_in` | 🟡 remote embedder — at full scale |

## Deviations from the specification and why

- **Length 150–350 instead of 150–300 (D-55).** That is what the phase task said (150–350, ~230 on average); the specification
  says 150–300 / ~220. It is a single parameter (`FilterConfig.min_words/max_words/target_mean_words`) plus the bins
  150–199 / 200–249 / 250–299 / 300–350. **Your decision is needed** on which range to fix.
- **Cluster threshold 50 instead of 300 (D-51).** With 3000 texts per genre, a threshold of 300 would give ~3 clusters; for dev
  the threshold is configurable (`--min-cluster-size`); at full scale it is set to 300. Labels and checks are the same.
- **Embeddings are local (D-90).** Per the phase conditions — no paid APIs: TF-IDF+SVD behind the `Embedder` protocol.
- **PII is discarded, not masked.** An `[EMAIL]` mask by itself would distinguish the human reference set H from AI texts,
  i.e. it would give the detector/humanizer an extra signal. A mask mode exists (`pii_policy="mask"`) for experiments.
- **The Pile via per-subset mirrors** (`timaeus/pile-*`), not via `monology/pile-uncopyrighted`
  (the same text, verified: the first document matches; monolithic ~10 GB shards are not needed for dev).
- **WritingPrompts — raw Reddit instead of the Fan et al. dataset.** The canonical version has PTB tokenization
  (`` `` quotes '' ``, "did n't"), an artifact visible to a detector; we took raw r/WritingPrompts comments
  with a per-row date (`sileod/reddit-WritingPrompts`), top-level only (the stories themselves).
- **Within a genre, slots replicate the strata frequencies of the pool**, not of production: production distributions by
  length/register/topic are not available yet (It's AI logs, D-52/D-55).

## Sources

Statuses: ✅ used; ⚪ works but not used; ❌ does not load; ⛔ does not guarantee a human author.
"Date" — how "before 2022" is proven: **row** = per-row timestamp, filter `< 2022-01-01` (rows without a date
are discarded); **corpus** = release/collection date of the whole corpus.

| Genre | Source (HF) | | Date | License (as stated) | Notes |
|---|---|---|---|---|---|
| essay | `HuggingFaceGECLM/REDDIT_submissions` (r/changemyview) — **Q-11** | ✅ | row `created_utc` | Reddit/Pushshift, no license on the card | adult opinion posts; EDITs/moderator footnotes are cut out |
| essay | `allenai/c4` realnewslike, URL `/opinion/`, `/op-ed/`, `/commentary/`, `/editorial/` — **Q-11** | ✅ | row `timestamp` (crawl Apr. 2019) | ODC-BY 1.0; © publishers | columns, op-eds, editorials |
| essay | `webis/args_me` (idebate, debatewise, debatepedia) — **Q-11** | ✅ | row `context.acquisitionTime` (Apr. 2019) | CC BY 4.0 | jsonl via the json builder (the repo has a loading script) |
| essay | `realbenpope/PERSUADE_manageable` (PERSUADE 2.0) | ✅ | corpus: school essays 2010–2020, Kaggle Feedback Prize (Dec. 2021) | upstream CC BY-NC-SA 4.0 (to verify); mirror MIT | US grades 6–12, ~15 prompt topics |
| essay | `tcapelle/feedback-prize-english-language-learning-fluency` (ELLIPSE) | ✅ | corpus: essays by English learners, before 2021, released Aug. 2022 (before ChatGPT) | upstream CC BY-NC-SA 4.0 (to verify) | the mirror has LLM columns, we take only `full_text` |
| email_letter | `allenai/c4` realnewslike, letters to the editor — **Q-11** | ✅ | row `timestamp` (Apr. 2019) | ODC-BY 1.0; © authors/publications | one page = one letter ≤ 350 words |
| article_blog | `fabiochiu/medium-articles` | ✅ | row `timestamp` | mirror MIT; texts © authors | |
| article_blog | `tasksource/blog_authorship_corpus` | ✅ | corpus: blogger.com, Aug. 2004 | mirror Apache-2.0; upstream non-commercial | |
| academic | `timaeus/pile-pubmed_abstracts` | ✅ | corpus: The Pile (2020-12-31) | publishers' terms (PubMed) | |
| academic | `gfissore/arxiv-abstracts-2021` | ✅ | row: month in the arXiv id | CC0 (arXiv metadata) | LaTeX abstracts are discarded |
| academic | `timaeus/pile-nih_exporter` | ✅ | corpus: The Pile (2020) | public domain (US gov) | NIH grant abstracts |
| news | `vblagoje/cc_news` | ✅ | row `date` (2017–2018) | unknown; © publishers | |
| news | `abisee/cnn_dailymail` 3.0.0 | ✅ | corpus: 2007–2015 | dataset Apache-2.0; © CNN | Daily Mail discarded (headers) |
| email_letter | `corbt/enron-emails` | ✅ | row `date` (1999–2002) | public FERC archive (2003) | only `@enron.com` senders |
| review | `yashraizad/yelp-open-dataset-reviews` | ✅ | row `date` | Yelp Dataset terms (non-commercial) | |
| review | `stanfordnlp/imdb` | ✅ | corpus: Maas et al. 2011 | other (IMDb) | |
| forum_qa | `HuggingFaceGECLM/REDDIT_comments` | ✅ | row `created_utc` | Reddit/Pushshift, no license on the card | AskHistorians, DIY, Fitness, LifeProTips, EatCheapAndHealthy, Games |
| forum_qa | `timaeus/pile-stackexchange` | ✅ | corpus: The Pile (2020) | CC BY-SA | threads are split into individual posts |
| forum_qa | `jkeisling/hacker-news-corpus-2007-2022` | ✅ | row `timestamp` (corpus up to Nov. 2022 → filter) | mirror MIT | `>` quotes are cut out |
| creative | `sileod/reddit-WritingPrompts` (comments) | ✅ | row `created_utc` | Reddit/Pushshift, no license | top-level comments only |
| * | `monology/pile-uncopyrighted` | ⚪ | corpus: The Pile | other | works with `zstandard`; for full scale |
| essay | `nlpatunt/D_ASAP-AES` (ASAP 2012) | ⚪ | corpus 2012 | not stated | 64% with `@CAPS1` placeholders, the rest is short |
| essay | `qwedsacf/ivypanda-essays` | ⛔ | no date; crawled Jan. 2023 | not stated | may contain post-ChatGPT texts |
| essay | `knarasi1/student_and_llm_essays` | ⛔ | — | — | a mix of human and LLM |
| essay | `baber/paul_graham_essays`, `srky/ICNALE_writing_score` | ⚪ | | | a single author / a few hundred texts |
| essay | `introvoyz041/PERSUADE_corpus_2.0`, `mariolinov/Ellipse` | ❌ | | | PDF only / wrong dataset |
| email_letter | `Yale-LILY/aeslc`, `timaeus/pile-enron_emails` | ⚪ | | | the same Enron (duplicates) |
| email_letter | `ShashiVish/cover-letter-dataset` | ⛔ | | | template/synthetic letters |
| review | `McAuley-Lab/Amazon-Reviews-2023` | ❌ | | | loading script; the raw jsonl can be read via `data_files` + a date filter |
| forum_qa | `webis/tldr-17` | ❌ | | CC BY 4.0 | loading script |
| forum_qa | `sentence-transformers/eli5` | ⚪ | | | answers are short (7% ≥ 150 words), no date |
| article_blog | `barilan/blog_authorship_corpus` | ❌ | | | loading script (we took the tasksource mirror) |
| article_blog | `timaeus/pile-pile-cc`, `allenai/c4`, `Skylion007/openwebtext` | ⚪ | corpora 2019–2020 | | general web, not a clean genre — reserve |
| news | `stanford-oval/ccnews` | ⚪ | row `published_date` | | multilingual 2016–2024: reserve for full scale |
| creative | `euclaise/writingprompts` | ⚪ | corpus 2018 | MIT | PTB tokenization (artifact) |
| creative | `euclaise/WritingPromptsX` | ⛔ | no date, collected Sep. 2023 | CC0 | |
| creative | `timaeus/pile-gutenberg_pg-19` | ⚪ | books before 1919 | | archaic register |
| * | `timaeus/pile-wikipedia_en` | ⚪ | Pile 2020 | | encyclopedic genre, not on the list |

## Dev build statistics

Build of 2026-11-08: `--per-genre 3000 --min-cluster-size 50`, 16 sources in parallel, ~16 min
(network: streaming reads from HF; 68 MB on disk in `var/pool/`). Re-clustering from the cache
(`--reuse-candidates`) — ~70 s. Pool `content_sha256`: `18f274fd…861e09` (in `manifest.json`).

**By genre** (length — bins by word count; register — heuristic; k — number of clusters):

| Genre | Texts | 150–199 | 200–249 | 250–299 | 300–350 | formal | informal | Mean words | k | Min./max. cluster |
|---|---|---|---|---|---|---|---|---|---|---|
| essay | 3000 | 629 | 928 | 844 | 599 | 1710 | 1290 | 248 | 19 | 50 / 310 |
| article_blog | 3000 | 884 | 1025 | 687 | 404 | 1316 | 1684 | 234 | 20 | 70 / 300 |
| academic | 3000 | 1260 | 969 | 561 | 210 | 3000 | 0 | 219 | 20 | 80 / 219 |
| news | 3000 | 865 | 932 | 757 | 446 | 2610 | 390 | 237 | 20 | 74 / 253 |
| email_letter | 3000 | 1130 | 835 | 614 | 421 | 2447 | 553 | 229 | 18 | 57 / 286 |
| review | 3000 | 1339 | 803 | 523 | 335 | 606 | 2394 | 220 | 20 | 61 / 273 |
| forum_qa | 3000 | 1436 | 811 | 471 | 282 | 1333 | 1667 | 216 | 20 | 79 / 402 |
| creative | 3000 | 786 | 960 | 769 | 485 | 538 | 2462 | 240 | 20 | 67 / 217 |
| **total** | **24 000** | | | | | | | **230** | **157** | all ≥ 50 ✅ |

**By source** (rows scanned → candidates that passed the filters → taken after deduplication and quotas):

| Source | Genre | Rows | Candidates | In pool | Dates in pool | Main rejection reasons |
|---|---|---|---|---|---|---|
| persuade2 | essay | 3 468 | 2 925 | 2 250 | corpus | placeholder 523 |
| ellipse | essay | 1 187 | 975 | 750 | corpus | placeholder 166, repetitive 31 |
| medium | article_blog | 2 952 | 2 340 | 1 800 | 2015-05 … 2021-01 | too_short 380, code 172, list_like 162, markup 129 |
| blog_authorship | article_blog | 4 164 | 1 560 | 1 200 | corpus (2004) | too_short 2 441, pii_url 93 |
| pubmed_abstracts | academic | 2 807 | 1 755 | 1 350 | corpus (Pile) | too_short 1 020 |
| arxiv_abstracts | academic | 4 993 | 1 365 | 1 050 | 2007-04 … 2019-12 | too_short 2 908, markup (LaTeX) 674 |
| nih_exporter | academic | 916 | 780 | 600 | corpus (Pile) | too_short 131 |
| cc_news | news | 4 616 | 2 730 | 2 100 | 2017-01 … 2018-07 | too_short 1 166, date_missing 403, pii_url 215 |
| cnn | news | 1 240 | 1 170 | 900 | corpus (2007–15) | pii_url 98 |
| enron | email_letter | 75 029 | 6 630 | 3 000 | 1980¹ … 2002-07 | too_short 45 277, external_sender 13 845, empty 7 668, exact_dup 1 689 |
| yelp | review | 13 429 | 2 535 | 1 950 | 2006-04 … 2018-10 | too_short 10 859 |
| imdb | review | 2 253 | 1 365 | 1 050 | corpus (2011) | too_short 860 |
| reddit | forum_qa | 49 514 | 2 340 | 1 800 | 2008-10 … 2015-05 | too_short 43 565, empty 3 452 |
| stackexchange | forum_qa | 6 501 | 975 | 750 | corpus (Pile) | too_short 9 516, markup 2 007, code 1 174 |
| hackernews | forum_qa | 10 322 | 585 | 450 | 2007-03 … 2021-12 | too_short 8 328, date_too_late 1 128 |
| writingprompts | creative | 24 770 | 3 900 | 3 000 | 2015-02 … 2020-03 | empty² 17 009, too_short 3 746 |

¹ A few Enron e-mails with a broken date of 1980-01-01 (a corpus artifact; still before 2022).
² "empty" for WritingPrompts means non-stories (replies in discussions, AutoModerator); for Enron, e-mails where nothing was left after
cutting off the forward/quote.

**Filter rejections, summed over all sources:** too_short 130 245 · empty 28 398 · external_sender 13 845 ·
markup 3 128 · **exact_dup 1 708** · code 1 624 · list_like 1 234 · date_too_late 1 128 · pii_url 945 ·
placeholder 710 · repetitive 446 · date_missing 403 · **near_dup 395** · pii_phone 314 · pii_email 178 ·
symbols 108 · pii_handle 101 · caps 82 · not_english 62 · pii_ssn 1. (Counts are per document/excerpt;
too_short dominates because reviews, comments and e-mails are mostly shorter than 150 words.)

**Sample (period demo):** `sample_slots(750, Q-2, seed="demo-period")` → essay 225, article_blog 113,
academic/news/email_letter/review/forum_qa 75 each, creative 37; 436 distinct strata out of 1 092 in the pool;
`sample_reference` for the same slots — **750/750 exact stratum matches**, a repeated call gives the same
result; time 0.08 s + 0.13 s.

## Examples of cluster labels

All 157 labels passed `check_label` (no digits, capitals, stop words, names). The "names" dictionary is built over the whole
pool (35 698 words with a capitalized share ≥ 0.3); it includes, for example, *obama, trump, enron, phillip, google,
facebook, chicago, texas, london, january, american, covid* — such words do not get into labels.

| Genre | Example labels (size) |
|---|---|
| essay | car, cars, usage (272) · school, students, career (223) · driving, phone, phones (169) · face, picture, landform (117) |
| article_blog | product, team, business (210) · blockchain, crypto, cryptocurrency (132) · election, vote, party (76) · haha, kinda, gonna (73) |
| academic | patients, treatment, patient (219) · magnetic, field, spin (201) · galaxies, emission, dust (104) · neurons, brain, neural (121) |
| news | police, officers, arrested (226) · judge, trial, charges (138) · market, percent, growth (124) · plane, airport, flight (81) |
| email_letter | trading, risk, market (232) · employees, employee, performance (211) · permit, emissions, air (122) · program, recruiting, candidates (93) |
| review | restaurant, food, menu (273) · sushi, food, roll (164) · hotel, room, rooms (107) · hair, salon, nails (61) |
| forum_qa | century, ancient, middle (167) · gym, weight, body (121) · add, pepper, salt (98) · server, client, connection (88) |
| creative | door, open, room (163) · simulation, universe, dimension (131) · gods, evil, sword (149) · sorry, love, miss (102) |

The full list is in `var/pool/clusters.json` and the tables in `var/pool/samples.html`.

## Live example (input → output)

```python
store = PoolStore.load("var/pool")
slots = store.sample_slots(750, DEFAULT_GENRE_PROPORTIONS, seed="demo-period")
slots[0]
# Slot(genre='academic', cluster_id='academic.c00', cluster_label='patients, treatment, patient',
#      length_bin='150-199', register='formal')
refs = store.sample_reference(slots, seed="demo-period")
```

Slot `('email_letter', 'email_letter.c00', 'meeting, office, issues', '150-199', 'formal')` → reference text
`e0754c3147a336b3817a` (enron, match=exact): *"Jeff, By the time you return from vacation, I will be on
vacation and only returning on Sept. 5th. As a result, I thought it would be best to e-mail this request to
you. Rick will be moderating a panel as part of an electricity roundtable…"*. The AI input generator will receive
only the slot tuple (genre, topic label, length, register) — this text itself is not available to it (D-51).

## Test results

| Acceptance criterion (design note) | Result |
|---|---|
| Offline tests green, fast | **115 passed** in ~11 s (107 of them in `tests/pool/`), 16 live tests excluded by default |
| Live download of each source | **16/16 passed** (`pytest -m live tests/pool`) |
| ≥ 2000 texts in each available genre | 3000 in each of the 8 genres ✅ |
| All clusters ≥ min_size | min 50 at threshold 50 ✅ (at threshold 300 — 154 of 157 are smaller, expected at dev scale) |
| All labels pass the check | 0 violations ✅ |
| `sample_slots` does not return text/IDs | test + type check ✅ |
| Exact counts per stratum, determinism | tests + demo 750/750 ✅ |
| `ruff check` / `ruff format --check` | clean ✅ |

What the tests cover: language, exact and near-duplicates, length boundaries, PII (e-mail, Lotus Notes, URLs, phone numbers in
various formats, SSNs, nicknames), money ≠ LaTeX, chemistry/genes ≠ code, placeholders, boilerplate lines and signatures,
excerpts (contiguity, alignment to paragraphs, splitting a long paragraph by sentences, mean
length ~230), date parsers and the strict `< 2022` filter, cleaning rules of each source, length bins, register,
clusters (size, purity on synthetic data, id stability, pluggable embedder), label and
cluster checks, distribution by genre, stratum relaxation, storage and loading, HTML escaping in the viewer,
the candidate cache with U+2028 characters.

## How to check it yourself

```bash
export UV_CACHE_DIR=/workspace/sn32_owner/.uv-cache HF_HOME=/workspace/sn32_owner/.hf-cache
export HUMEVAL_ENV_FILE=/workspace/sn32_owner/.env
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev,pool]"
.venv/bin/pytest                                   # offline, ~11 s
.venv/bin/pytest -m live tests/pool                # 20 sources over the network (after Q-11), a few minutes
.venv/bin/python scripts/build_pool_dev.py         # dev -> var/pool/, prints statistics
.venv/bin/python scripts/build_pool_dev.py --reuse-candidates   # rebuild clusters without network, ~70 s
# full scale (Q-11 / D-51), see the last section:
.venv/bin/python scripts/build_pool_dev.py --out var/pool-full --per-genre 30000 \
    --min-cluster-size 300 --stride-divisor 10 --workers 24
```

## What for you to look at (checklist)

- [ ] Open **`var/pool/samples.html`** (20 random texts per genre + a cluster table): do the texts
      look like real human texts of their genre? Is there any junk (site menus, newsletters, code)?
- [ ] Are the cluster labels general enough for the topic generator (D-51/D-52) and free of specifics?
- [ ] Decide the length range: **150–300** (spec D-55) or **150–350** (phase task) — a single parameter.
- [x] ~~Accept the composition of the "essay" genre (school PERSUADE/ELLIPSE only) and "letters" (business Enron only) for v1
      or look for a replacement~~ — decided in Q-11 (expand), done: see the last section. Open: whether
      letters to the editor are acceptable as "letters" in the absence of personal correspondence.
- [ ] Licenses: PERSUADE/ELLIPSE (probably CC BY-NC-SA), Yelp (non-commercial), Reddit/HN without a license —
      a legal review is needed before production.
- [ ] Register heuristic: academic 100% formal, review/creative ~80% informal, essay/email/forum — a mix;
      acceptable as a coarse stratum?

## Risks, open questions, what is needed next

- **Essays are narrow** *(dev build; closed in Q-11, see the last section)*. Only school essays (US grades 6–12 + English learners), ~15 PERSUADE prompt topics;
  the clusters almost coincide with the prompts ("car, cars, usage", "face, picture, landform"). University essays, opinion essays,
  admission essays with a provable pre-2022 date were not found (IvyPanda — no dates, collected after ChatGPT). Options:
  argumentative Reddit r/changemyview posts before 2022 (as "opinion essays"), ASAP with placeholder replacement,
  licensed corpora (ICLE, LOCNESS).
- **Letters are narrow** *(dev build; partially closed in Q-11, see the last section)*. Only Enron (an energy company, 2000–2001, business correspondence). There are no personal
  letters. Options: public mailing-list archives (Apache, W3C, LKML) before 2022, Usenet; the paid Avocado (LDC).
- **forum_qa** is partly technical (StackExchange even after the code filters); Reddit comments only cover
  2008–2015 (early shards). At full scale — more shards and subreddits.
- **Reviews** are only Yelp + IMDB; Amazon-2023 requires reading the raw jsonl with a date filter — to be done at full scale.
- **Detectors may have been trained on these corpora** (PERSUADE is the human class of Kaggle "LLM Detect AI"); then
  the human reference set H is "too easily human". This affects D-31/D-20; worth checking on canaries (phase 3).
- **Register** is a coarse heuristic (formula in `metadata.py`); `register_score` is stored, so the threshold can be changed without
  a rebuild. For production — calibrate on annotations or with a judge.
- **Full scale (D-50: millions):** the same code; sources `monology/pile-uncopyrighted`, `stanford-oval/ccnews`,
  C4/OpenWebText as an article_blog reserve; `--min-cluster-size 300`; an embedder via OpenRouter (D-90) behind the same
  `Embedder` protocol — with a budget in the spending log.
- **Order within corpora.** Many corpora are sorted (Enron by mailbox, cc_news by domain, IMDB by
  class, blogs by author), so sources have a `stride` (we take every n-th row) — otherwise topics
  are skewed. For full scale, an honest random sample across shards is better.
- **`datasets` crash at process exit** (streaming threads, `PyGILState_Release`): the build script and
  live tests exit via `os._exit` after writing results; this does not affect the data.

## Q-11: expanding "essays" and "letters", and the full build (2026-11-09)

Owner's decision Q-11: try to expand the narrow genres, and if that fails, remove the genre from the mix.
**Outcome: both genres were expanded; there is no need to remove them.** Essays are now mostly adult opinion texts (school
PERSUADE/ELLIPSE — 35% of the weight instead of 100%). Letters — Enron plus published letters to the editor; **personal
correspondence (private letters, personal e-mails) with a provable pre-2022 date was never found** — this remains the
bottleneck of the genre (see "What remained narrow"). No paid APIs were called: only streaming reads from HF and local
TF-IDF/SVD; the spending log was not touched.

### Added sources

| Genre | id | Source (HF) | Date (proof of "before 2022") | License (as stated) | What we take and how we clean it |
|---|---|---|---|---|---|
| essay | `cmv` | `HuggingFaceGECLM/REDDIT_submissions`, `data/changemyview-*.parquet` (257 thousand posts 2013–2023) | **row** `created_utc`, filter `< 2022-01-01` (≈ 90% of the dump is before 2022) | Reddit/Pushshift, no license on the card (same as the current `reddit`) | original posts of r/changemyview — a reasoned opinion of an adult author (thesis in the title, the text is the justification). Everything from the first `EDIT:/UPDATE:` or `_____` separator onward (replies to commenters, "delta awarded", the moderators' footnote) is cut off; markdown → text; `[removed]`/`[deleted]` are skipped |
| essay | `c4_opinion` | `allenai/c4`, `realnewslike/*.json.gz` (13.8 million news-site pages) | **row** `timestamp` — Common Crawl download time (Apr. 2019), filter `< 2022` | ODC-BY 1.0 (+ Common Crawl terms); texts © publishers | pages with URL `/opinion/`, `/op-ed/`, `/commentary/`, `/editorial/` (columns, op-eds, editorials); no letters, interviews/Q&A (two voices), section listings or roundups. HTML entities (`&apos;`) are decoded; cut out are weather-widget lines, the section kicker ("Letter: …"), the print-edition note ("A version of this article appeared…"), the editorial subheading ("Writer feels that …"), the final author signature/bio ("Jane Doe is a retired teacher…") |
| essay | `argsme` | `webis/args_me`, `args-me.jsonl` (387 thousand arguments) | **row** `context.acquisitionTime` — portal crawl (Apr. 2019) | CC BY 4.0 | only the edited portals **idebate.org, debatewise.org, debatepedia.org** (~5.7 thousand usable texts); the repo has a loading script, so the raw jsonl is read by the json builder (`Source.builder="json"`). Cleaning: `don"t` → `don't`, glued paragraphs (`end.Next`) are split, the reference list and `[n]` markers are removed |
| email_letter | `c4_letters` | `allenai/c4`, `realnewslike/*.json.gz` | **row** `timestamp` (Apr. 2019), filter `< 2022` | ODC-BY 1.0; texts © authors/publications | letters to the editor: a letter URL (`letters-to-the-editor`, `/letters/`, …), **not** a listing/roundup of the day (`tuesday-letters…`, `letters-to-the-editor-2-16-17`, `?page=`), and the whole page ≤ 350 words — i.e. one page = one letter = one author. Signatures "Name, City." and "The writer is …" are trimmed. About 1 letter turns up per 900 pages |

**Weights within a genre** (`Source.weight`): essay — `cmv` 0.35, `c4_opinion` 0.15, `argsme` 0.15, `persuade2` 0.25,
`ellipse` 0.10; email_letter — `enron` 0.55, `c4_letters` 0.45. The order in the registry is both the priority in
deduplication and the order of "top-up" on shortfall, so in essays CMV comes first (the top-up comes from adult
texts, not from PERSUADE).

**Why these sources meet the Q-11 conditions.** (a) All four have a per-row date: Reddit — post time,
C4 and args.me — page download time (the text was written no later; everything is before 2022, i.e. before ChatGPT).
(b) Length: after filters, 150–350 words is passed by ~21% of scanned CMV posts, ~90% of selected op-eds
(as an excerpt from the start of the article), ~12% of arguments from the edited args.me portals (the rest are shorter than 150
words); letters are taken only whole. (c) Licenses are recorded as stated. (d) None is synthetic or
mixed with LLM text: CMV is a raw Pushshift dump, C4 and args.me are 2019 crawls.

### Checked and rejected candidates (added to `REJECTED`)

| Genre | Source (HF) | | Reason |
|---|---|---|---|
| essay | `webis/args_me` — the **debate.org** part (87% of the corpus) | ⚪ | debate rounds quote the opponent directly in the text (someone else's words inside a "human" text), round templates ("my opponent", "Con:"), HTML-stripping artifacts |
| essay | `notaphoenix/debateorg_w_effect_for_*` | ⚪ | the same debate.org |
| essay | `simonycl/cmv_2017_2025`, `idodah/webis-cmv-20-pairs`, `nailiamirzakhmedova/cmv_op_10k`, `JingyangYi/CMV_op` | ⚪ | subsets of r/changemyview (pairs repeat the post, some without dates or in prompt format) — the full GECLM dump was taken |
| essay | `Siddish/change-my-view-subreddit-cleaned`, `launch/CMV`, `QCRI/cmv` | ⚪ | chat format `### Human/### Assistant` / test-json only / closed audio corpus |
| essay | `mmmaurer/winning-arguments-ling-enriched`, `tjb001/convokit-winning-arguments` | ❌ | features only, no text / empty repository |
| essay | `DriftLogic/Annotated_Persuasive_Essays` | ⛔ | according to the card — LLM-generated essays |
| essay | `Efstathios/guardian_authorship` | ❌ | loading script; moreover ~450 columns by 13 authors |
| essay | `ndtran0101/writing9-ielts-essays` | ⚪ | 163 thousand IELTS essays, collected in 2026; before 2022 (by the time in the ObjectId) only ~9 thousand from 2020–21, license unknown, the L2 register is already covered (ELLIPSE) |
| essay | `chillies/IELTS-writing-task-2-evaluation`, `Atharva1244/college_essays` | ⛔ | no date and provenance / ~100 essays |
| essay | `vblagoje/cc_news` (opinion/letters URLs) | ⚪ | only 5.6 thousand opinion and 1.4 thousand letter URLs out of 708 thousand articles; C4 realnewslike has ~10 times more |
| email_letter | `fddemarco/pushshift-reddit` (r/UnsentLetters etc.) | ⚪ | there are ~1 thousand "unsent letters" per month (2012–2018), but `selftext` in the dump is lowercased and without punctuation — an artifact visible to a detector |
| email_letter | `Hieuman/gmane_dataset` | ⛔ | mailing lists without dates (Gmane archives after 2022 as well), authors anonymized, multilingual spam mixed in |
| email_letter | `dwablimol/r-mailing-lists-raw` | ⛔ | R mailing lists without a date field (r-help runs to 2024), mostly questions with code |
| email_letter | `HypernetworkRG/email-W3C` | ❌ | a recipient hypergraph (74 KB), no text |
| email_letter | `talby/spamassassin` | ❌ | loading script, data is downloaded from apache.org, nothing to stream on the Hub |
| email_letter | `from-our-page/hillary-clinton-emails-wikileaks` | ⚪ | line-by-line OCR text of FOIA PDFs (State Department headers, redaction codes), ~9.6 thousand files, mostly forwarded news; no license |
| email_letter | `OwnedByDanes/Usenet-Corpus-1980-2013-*`, `wizzense/Usenet-Corpus-1980-2013` | ⚪ | the full corpus is under a commercial license (only samples are free); Usenet is forum threads, not letters |
| email_letter | `devplex97/usenet-humanities` | ⛔ | Usenet without dates and provenance (probably derived from a licensed corpus) |
| email_letter | `tim-a-wood/fda-warning-letters`, `emperor-mew/sec-comment-letters` | ⚪ | template regulatory letters (dates exist, but it is solid boilerplate) / card only, data behind an API |
| email_letter | `ISTA-DASLab/Panza-emails`, `eagle0504/warren-buffett-annual-letters-*` | ⚪ | < 1 thousand letters from three people / a single author, ~45 shareholder letters |
| email_letter | `notesbymuneeb/epstein-emails` and mirrors | ⛔ | not opened: correspondence from a 2025 release with personal data of third parties — excluded on ethical grounds |
| email_letter | `Lucius-Morningstar/enron-correspondence-dedup`, `Hellisotherpeople/enron_emails_parsed` | ⚪ | the same Enron again |

### Code changes

- `sources.py`: 4 sources and their cleaning/selection rules (`clean_cmv`, `argsme_curated_portal`, `clean_argsme`,
  `c4_letter_page`, `c4_opinion_page`, `clean_c4_news`), 22 new `REJECTED` entries, new genre weights.
  Fields `Source.builder` (raw repo files with an unsupported script — args.me) and
  `Source.max_rows_per_candidate` (a row budget for sources where the URL/portal filter leaves < 1% of rows),
  `Source.parallel_files` (streaming over `data_files` slices in several workers with the quota and budget split between them:
  `c4_letters` reads all 512 realnewslike shards; with one stream that takes ~1 h, with eight — minutes;
  `source_jobs`/`merge_reports` in `build.py`, candidates are concatenated in file order).
- `build.py`: `BuildConfig.stride_divisor` / `--stride-divisor` — the `stride` in the registry is tuned for dev (3000 per
  genre); at 30 000 it must be divided by 10, otherwise sources with thinning run out before the quota
  (`stride 50 → 5`, never less than 1); the resulting `stride` is written to `manifest.json`.
- `build.py`: build workers are started with a single BLAS thread (`single_threaded_children`:
  `OMP/OPENBLAS/MKL_NUM_THREADS=1` only for child processes). Otherwise each of the 20 workers spun up
  ~64 OpenBLAS threads for tiny langid matrix multiplications, and the 128 cores choked: measured — 335 → 67 ms
  per document, langid 227 → 21 ms. The first full-build attempt (without this) finished one source in 25 min
  (`ellipse`, 3.9 thousand rows in 14 min) and was stopped; after the fix the same volume takes ~1 min.
  This probably also slowed down the dev build (16 min). The script output is now line-by-line (progress is visible in the log).
- `clean.py`: a new rejection reason `encoding` — text with the U+FFFD replacement character ("don�t"): an artifact
  of the crawler's decoding, not of the author (occurs in C4).
- Tests (`tests/pool/`): CMV rules (EDIT/footnote/markdown, Reddit dates), args.me (portals, crawl date,
  fixing quotes/glued text/references, json builder in `iter_rows`), C4 (selection of letter/op-ed URLs and roundups, ≤ 350 words,
  widgets/kicker/subheading/signature), `stride_divisor`, per-source row budget, a single BLAS thread in
  workers, `encoding`.
- **Check results:** `tests/pool` — 120 offline tests (was 107), all green; the whole offline suite of the repository —
  298 passed; live tests of the 4 new sources (`pytest -m live tests/pool -k "cmv or c4_ or argsme"`) —
  4/4; `ruff check` / `ruff format --check` — clean.

### Full build (`var/pool-full`)

```bash
.venv/bin/python scripts/build_pool_dev.py --out /workspace/sn32_owner/humanizer_eval/var/pool-full \
    --per-genre 30000 --min-cluster-size 300 --stride-divisor 10 --workers 24 [--reuse-candidates]
```

Build of 2026-11-09, 20 sources, target 30 000 per genre, cluster threshold **300** (D-51), k = n // 900.
**239 431 texts**, pool `content_sha256` `a0607633…99acee22` (in `manifest.json`). 617 MB on disk (of which
470 MB is the candidate cache `candidates/`, from which the cluster rebuild runs without network); everything is on `/workspace`,
the root disk was not touched (HF cache 85 MB — streaming does not download anything in full). Time: streaming reads
— 1–5 min per source in parallel (the longest are `enron` at 5 min and `c4_letters` at 5 min on 8 workers over all
13.8 million pages of C4 realnewslike), deduplication of 324 thousand candidates + clusters + writing — ~9 min; the final
run took 14 min (17 sources from the cache, 3 re-read after fixes). Memory: workers ≤ 6 GB, main process
~2.3 GB, total < 25 GB (the machine has 440 GB). History: the first attempt without the BLAS thread limit was
stopped (see "Code changes"); the second — after a spot check of C4 texts (HTML entities, print-edition
notes, interviews) — `clean_c4_news`/`c4_opinion_page` were fixed, and the CMV step was increased so that the sample
covers all years; logs of all attempts are alongside (`build*.log`).

**By genre** (k — number of clusters ≥ 300; the 300–350 bin is present in all genres):

| Genre | Texts | 150–199 | 200–249 | 250–299 | 300–350 | formal | informal | Mean words | k (≥300) | Min./median/max. cluster |
|---|---|---|---|---|---|---|---|---|---|---|
| essay | 30 000 | 9 251 | 9 075 | 6 961 | 4 713 | 19 282 | 10 718 | 236 | 31 | 327 / 929 / 1778 |
| article_blog | 30 000 | 9 057 | 9 668 | 7 198 | 4 077 | 13 016 | 16 984 | 234 | 33 | 554 / 774 / 1749 |
| academic | 30 000 | 12 426 | 9 592 | 5 643 | 2 339 | 29 997 | 3 | 220 | 33 | 441 / 931 / 1356 |
| news | 30 000 | 8 385 | 9 327 | 7 670 | 4 618 | 25 742 | 4 258 | 238 | 32 | 325 / 873 / 1885 |
| email_letter | **29 431** | 10 417 | 8 575 | 6 233 | 4 206 | 24 756 | 4 675 | 231 | 32 | 305 / 887 / 1627 |
| review | 30 000 | 13 140 | 8 231 | 5 248 | 3 381 | 6 377 | 23 623 | 221 | 32 | 390 / 865 / 2033 |
| forum_qa | 30 000 | 14 676 | 7 965 | 4 681 | 2 678 | 12 492 | 17 508 | 214 | 33 | 488 / 809 / 2185 |
| creative | 30 000 | 7 651 | 9 796 | 7 811 | 4 742 | 5 921 | 24 079 | 240 | 33 | 416 / 874 / 1566 |
| **total** | **239 431** | 85 003 | 72 229 | 51 445 | 30 754 | | | | **259** | all ≥ 300 ✅ |

`check_clusters`: 0 problems — all 259 clusters are ≥ 300, all labels pass `check_label`. The 300–350-word bin:
from 2 339 (academic, 7.8%) to 4 742 (creative, 15.8%) texts per genre, 30 754 in total (12.8%).

**By source** (rows scanned → candidates that passed the filters → in the pool after deduplication and quotas;
"exhausted" — the source ran out before the quota at this step/filter):

| Source | Genre | Rows | Candidates | In pool | Quota | Exhausted | Dates in pool | Main rejection reasons |
|---|---|---|---|---|---|---|---|---|
| cmv | essay | 64 333 | 13 408 | 10 500 | 13 650 | yes | 2013-01 … 2020-02¹ | empty 36 927, date_too_late 6 699, too_short 6 339 |
| c4_opinion | essay | 461 053 | 5 850 | 4 500 | 5 850 | no | 2019-04 (crawl) | not_opinion_url 453 074, letter_url 868, listing_url 594 |
| argsme | essay | 382 545 | 5 698 | 4 500 | 5 850 | yes | 2019-04 (crawl) | debate_org_round 333 473, too_short 42 077 |
| persuade2 | essay | 10 232 | 9 750 | 7 500 | 9 750 | no | corpus | placeholder 418, repetitive 31 |
| ellipse | essay | 3 911 | 3 221 | 3 000 | 3 900 | yes | corpus | placeholder 551, repetitive 85 |
| medium | article_blog | 29 444 | 23 400 | 18 000 | 23 400 | no | 1985-01² … 2021-12 | too_short 3 767, code 1 933, markup 1 667 |
| blog_authorship | article_blog | 41 857 | 15 600 | 12 000 | 15 600 | no | corpus (2004) | too_short 24 670, pii_url 719 |
| pubmed_abstracts | academic | 28 125 | 17 550 | 13 500 | 17 550 | no | corpus (Pile) | too_short 10 332 |
| arxiv_abstracts | academic | 50 361 | 13 650 | 10 500 | 13 650 | no | 2007-04 … 2020-02 | too_short 29 564, markup 6 673 |
| nih_exporter | academic | 9 041 | 7 800 | 6 000 | 7 800 | no | corpus (Pile) | too_short 1 197 |
| cc_news | news | 47 302 | 27 300 | 21 000 | 27 300 | no | 2017-01 … 2018-10 | too_short 12 425, date_missing 4 350, pii_url 2 147 |
| cnn | news | 12 355 | 11 700 | 9 000 | 11 700 | no | corpus (2007–15) | pii_url 1 000 |
| enron | email_letter | 406 047 | 36 465 | **13 737** | 36 465 | no | 1980-01³ … 2002-07 | too_short 249 972, external_sender 70 968, **exact_dup 18 929, near_dup 3 799** |
| c4_letters | email_letter | 13 799 838 | 15 711 | 15 694 | 17 550 | yes | 2019-04 (crawl) | not_letter_url 13 755 690, page_too_long 12 449, too_short 10 451, listing_url 3 653 |
| yelp | review | 130 988 | 25 350 | 19 500 | 25 350 | no | 2005-07 … 2018-10 | too_short 105 281 |
| imdb | review | 23 020 | 13 650 | 10 500 | 13 650 | no | corpus (2011) | too_short 9 134 |
| reddit | forum_qa | 535 399 | 23 400 | 18 000 | 23 400 | no | 2008-10 … 2018-12 | too_short 470 989, empty 39 335 |
| stackexchange | forum_qa | 68 534 | 9 750 | 7 500 | 9 750 | no | corpus (Pile) | too_short 101 343, markup 21 132, code 12 447 |
| hackernews | forum_qa | 105 263 | 5 850 | 4 500 | 5 850 | no | 2007-03 … 2021-12 | too_short 83 911, date_too_late 12 678 |
| writingprompts | creative | 257 933 | 39 000 | 30 000 | 39 000 | no | 2015-02 … 2020-03 | empty 179 414, too_short 38 312 |

¹ CMV candidates cover all years 2013–2021 (1.1–1.9 thousand per year); the build takes the first ones in file order into the pool,
so the pool has 2013 … 2020-02. ² Medium stores migrated old posts with their original date. ³ Broken Enron dates
1980-01-01 (a corpus artifact; still before 2022).

**Essay composition:** CMV 10 500 (35%), PERSUADE 7 500 (25%), C4 op-ed 4 500 (15%), args.me 4 500 (15%),
ELLIPSE 3 000 (10%) — school essays are 35% instead of 100% in dev. **Letter composition:** letters to the editor 15 694 (53%),
Enron 13 737 (47%).

**Clusters and sources.** In essays, 9 of 31 clusters are > 90% from a single source: 7 are PERSUADE prompt topics
("summer, projects, students", "extracurricular, activities", "driving, phone, phones", "car, cars, usage"…),
one each from ELLIPSE and CMV; the remaining 22 are mixed (for example "tax, money, income", "war, nuclear, military",
"rights, government, state"). In letters, 29 of 32 clusters are homogeneous by source (12 Enron: "meeting, group,
discuss", "trading, risk, price", "agreement, contract"…; 17 letters to the editor: "vote, party, election",
"school, students, schools", "town, housing, building"…) — business mail and letters to newspapers barely
overlap in topics, so the slot's cluster effectively also determines the subtype of letter.

### What remained narrow, risks, blockers

- **email_letter: 29 431 of 30 000 (−569).** The limit is the sources, not the code: of 36 thousand Enron candidates
  only 13.7 thousand are unique (one e-mail sits in several folders — 22.7 thousand duplicates), and letters to the
  editor were collected from **all** 13.8 million pages of C4 realnewslike. The rest can be topped up with the same URL filter over C4 `en`
  (305 GB, ~25 times more pages) or over `stanford-oval/ccnews` 2016–2021 — that is hours of streaming; not done.
- **There is no personal correspondence.** "Letters" = Enron business mail + public letters to the editor (in content
  these are reasoned opinions, closer to essays; formal 87% versus 81% for Enron). Private letters/personal e-mails with a
  provable pre-2022 date were not found on HF in a usable form (r/UnsentLetters exists only in a dump without case and punctuation). **The decision is yours:** whether to consider
  this composition sufficient for the "letters" genre or to remove the genre from the mix under the Q-11 rule. Technically the genre
  is filled (29.4 thousand, 32 clusters ≥ 300).
- **Detector training data.** CMV is part of the human class of MAGE (Li et al., 2023), as are Yelp and
  WritingPrompts; PERSUADE is in Kaggle "LLM Detect AI". The expansion lowers the share of PERSUADE but does not remove the risk
  of a "too easily human" reference set: check on canaries (phase 3).
- **The date for C4 and args.me is the crawl time (Apr. 2019)**, not the writing time: it proves "before 2022", but the text itself
  may be older (letter URLs show years 2009–2019).
- **Licenses:** Reddit (CMV) has no license on the card; C4 — ODC-BY, but texts © publications; args.me CC BY 4.0.
  A legal review is needed before production, as for the other sources.
- **Minor artifacts** (fractions of a percent): C4 has sentences glued together without a space ("Montiel.Ernst");
  HTML entities remain in `blog_authorship` (108) and `writingprompts` (131), "Edit:" lines in `reddit` (~1 thousand
  of 18 thousand) — these are old sources; not touched.
- **Order of taking candidates.** The build takes the first N candidates of a source in file order (which is why CMV in
  the pool has no 2020–21, although candidates exist). For production, a uniform subsample of the candidates should be taken.

## Pool v2: genres following the production distribution (2026-11-09)

The owner measured the production distribution: 2000 English texts that users send to the
humanizer. The pool was rebuilt in three places to match it. Three new genres were added. Academic texts are now
mostly article paragraphs rather than abstracts. Essays gained non-argumentative subtypes. No paid APIs were
called: only streaming reads from HF and local TF-IDF/SVD. Lium was not touched, nor was the spending log.

**Outcome.** `var/pool-full-v2` holds **320 021 texts** in 11 genres. There are **344 clusters, all ≥ 300** (the smallest is
306), and all labels passed the check.

- `academic`: 88% article paragraphs and 12% abstracts, as in production.
- `application`: **filled**, 21 028 admission essays from essayforum.com. The genre is kept, but with a caveat about the
  date: the corpus was published on 2022-09-02, see "Blockers".
- `business_marketing` and `homework_answer`: 30 000 texts each.
- The bottleneck is **essay composition**. Only ~6.5 thousand non-argumentative school essays with a provable date were collected.
  Hence argumentative essays are 78.5% instead of 24%, and narrative ones 0.5% instead of 17%.

### Production distribution → pool genres → mix weights

| Production category | Share | Pool genre | `DEFAULT_GENRE_PROPORTIONS` / `configs/period.json` |
|---|---|---|---|
| academic_writing | 39% | `academic` | 0.39 / 39 |
| school_essay | 15% | `essay` | 0.15 / 15 |
| application | 7% | **`application`** (new) | 0.07 / 7 |
| business_marketing | 7% | **`business_marketing`** (new) | 0.07 / 7 |
| forum_social | 6% | `forum_qa` | 0.06 / 6 |
| homework_answer | 6% | **`homework_answer`** (new) | 0.06 / 6 |
| article_blog | 4.5% | `article_blog` | 0.045 / 4.5 |
| creative | 2.2% | `creative` | 0.022 / 2.2 |
| review | 0.9% | `review` | 0.009 / 0.9 |
| news | 0.7% | `news` | 0.007 / 0.7 |
| — | — | `email_letter` | in the pool, outside the mix (Q-11) |

- The weights are relative: they sum to 0.883; the "other" category is not counted.
- That the two places match is checked by the test `test_period_config_genre_proportions_match_the_pool_defaults`.
- There are no generation templates for the new genres in `period/generate.py` yet; the owner will add them. Until then
  the generator uses the article_blog template for them.
- Sampling demo on v2: `sample_slots(1000, DEFAULT_GENRE_PROPORTIONS)` gave academic 442, essay 170,
  application 79, business_marketing 79, forum_qa 68, homework_answer 68, article_blog 51, creative 25,
  review 10, news 8 (2 575 strata in the pool). `sample_reference` for the same slots returned 1000/1000 exact
  stratum matches.

### Code changes

**The length window became a build parameter; the candidate cache was widened to 100–500 words.**
- Workers collect and cache candidates in the window `cache_min_words..cache_max_words`, 100–500 by default.
  A document longer than 500 words yields the longest prefix of whole paragraphs that fits within 500.
- The pool window is set by `BuildConfig.min_words/max_words` (flags `--min-words/--max-words`). The default is
  **150–350, unchanged**. It is applied at build time by the function `clean.fit_window`:
  - a candidate inside the window is taken whole;
  - a candidate shorter than the window is discarded (`window_short`);
  - a candidate longer than the window is cut by the same `excerpt` rule as before (N(230, 55), by paragraphs), and
    the piece goes through the quality and PII checks again.
- The result was checked against the previous one. A test shows that the piece matches a direct cut of the document. A full
  build of the old 8 genres via the cache gave the same length bins as `var/pool-full`, with a discrepancy of ≤ 1–2 p.p.
  The volumes also matched, except for email_letter: 28 993 instead of 29 431. The reason: the cache checks PII and quality on
  500 words, which is slightly stricter.
- A worker's quota counts only candidates ≥ `min_words`. Short ones (100–149 words) are cached for later.
- The cache report stores its window (`SourceReport.window`), and a cache with a different window is not reused.
- Length bins are built from the window (`metadata.make_length_bins`, step of 50 words). For 150–350 these are the same 4 bins as before.
- `window`, `cache_window`, `length_bins` were added to `stats.json`.
- The window can be changed within 100–500 with `--reuse-candidates`, without network: a rebuild from the cache takes ~18 min.

**Other.**
- `BuildConfig.per_genre_overrides` (flag `--genre-target GENRE=N`) sets a genre its own target. It was not
  needed in this build.
- `Source.subtype` — a subtype within a genre. It goes into `stats.json` (`genres.*.by_subtype`) and into the
  "within-genre mix" line of the build output. The achieved composition versus the target is computed from it.
- `Source.builder_kwargs` is needed for ASAP: tsv in latin-1 encoding.
- `Source.row_parser` reads JSON lines via the `text` builder. Without it, the json builder infers the type of `created` in
  Common Pile PubMed as a timestamp and crashes on the value "2021-8-13".
- `parse_date` now understands dates without leading zeros ("2003-8-18").
- New parsers in `sources.py`:
  - article paragraphs: `pandoc_blocks` for PMC, `grobid_blocks` for peS2o, `paper_paragraph_ok`,
    `paragraph_windows`;
  - PERSUADE selection by prompt;
  - ASAP by set number, with a fix for cp1252 quotes;
  - essayforum cleaning;
  - Amazon descriptions;
  - press releases and "about the company" pages from C4;
  - 10-K sections (EDGAR);
  - Reddit and StackExchange answers.
- Tests: `tests/pool/` now has 146 offline tests, up from 120. They cover windows and the cache, bins, each new
  parser and filter, weight synchronization. Live streaming tests: 40, one per source. All 21 new ones
  passed.

### Sources by genre

The statuses in the rejection tables are the same as above: ⚪ works but not used; ❌ does not load; ⛔ does not
guarantee a human author or a date.

#### academic

Production: 85% are sections of papers and theses (introduction, literature review, methods, discussion; 1–3 paragraphs,
median ~213 words). Abstracts are 11.5%.

| id | Source (HF) | Date | License | What we take |
|---|---|---|---|---|
| `pmc_papers` | `common-pile/pubmed`, shards 0–11 (PMC in pandoc-markdown from the publisher's XML) | **row** `created` = publication date, filter `< 2022` | Common Pile: open licenses only; each article has `metadata.license` (CC BY / BY-SA / CC0) | 1–3 consecutive paragraphs from the introduction, methods or discussion (we do not take results), starting at a random paragraph |
| `pes2o_social` | `common-pile/peS2o`, files 51–101 (S2ORC, Grobid from PDF) | **row** `created` and `metadata.year`, both < 2022; placeholders 0001-01-01 and 1970-01-01 are treated as a missing date | Common Pile: each article has `metadata.oa_license` (CC BY / BY-SA / PD) | psychology, education, economics, sociology, political science, business, philosophy, history, linguistics, law, art, geography; no medicine or biology |
| `pes2o_stem` | `common-pile/peS2o`, files 0–50 | same | same | physics, CS, engineering, mathematics, materials science, chemistry; no medicine or biology |
| `pubmed_abstracts`, `arxiv_abstracts`, `nih_exporter` | as before | as before | as before | abstracts; total weight 0.12 instead of 1.0 |

Paragraph rule (`paper_paragraph_ok`):
- 25–350 words and at least 2 sentences;
- starts and ends like a sentence;
- not a figure or table caption and not a list item;
- no formulas, LaTeX, or more than 3 Greek letters;
- no statistical insertions: more than 3 p</CI/%/± tokens, more than 5% digits, more than 6% single-letter tokens;
- no PDF traces (glued "end.Next", a lowercase letter after a period) and no markup remnants.

References like `[@B1]` and `[1–3]` are removed. But if a reference is part of the sentence ("as shown in [12]", "Smith et
al. [@x]"), the paragraph is skipped entirely. Headings, captions, tables, formulas, lists and discarded paragraphs
**break** the window, so a window always consists of consecutive paragraphs. A paragraph longer than 350 words is not taken:
otherwise it would have to be cut in the middle.

#### essay

Production: 49% expository, 24% argumentative, 17% narrative/personal, 10% literary analysis.

| id | Source | Subtype | Date | License | What we take |
|---|---|---|---|---|---|
| `persuade2_expository` | `nlpatunt/D_persuade_2` (PERSUADE 2.0 with a `prompt_name` column) | expository | corpus: school essays 2010–2020, Kaggle Feedback Prize (Dec. 2021) | upstream CC BY 4.0 | prompts "Car-free cities" (an explanatory essay based on a set of texts) and "Seeking multiple opinions"; the essay's title line is stripped |
| `persuade2_analysis` | same | literary (text analysis) | same | same | prompt "Exploring Venus": evaluate how the article's author supports a claim. This is an analysis of a source text, so it is classified as literary analysis |
| `asap_literary` | `TasfiaS/ASAP-AES` (tsv) | literary | corpus: Hewlett ASAP, Kaggle, Feb. 2012 (grades 7–10) | competition data, no open license | sets 3–5: analysis of a story or memoir; only essays without `@CAPS` placeholders |
| `asap_expository` | same | expository | same | same | set 6: explanation based on an informational text |
| `asap_narrative` | same | narrative | same | same | sets 7–8: personal stories ("patience", "laughter"); most have placeholders |
| `persuade2` | mirror **changed** to `nlpatunt/D_persuade_2` | argumentative | as before | upstream CC BY 4.0 (previously marked "to verify") | only the 12 argumentative prompts |
| `cmv`, `c4_opinion`, `argsme`, `ellipse` | as before | argumentative | as before | as before (ELLIPSE upstream CC BY 4.0) | weights lowered |

The order of sources in the registry sets the priority in deduplication and the top-up order. Rare subtypes come
first. If they fall short of their share, the shortfall is topped up by CMV (adult texts), not by school essays.

#### application (new genre)

This covers personal statements, admission and scholarship essays, and motivation letters.

| id | Source | Date | License | What we take |
|---|---|---|---|---|
| `essayforum` | `nid989/EssayFroum-Dataset`: 25.6 thousand posts from essayforum.com, column `Cleaned Essay` | **corpus**: the repository was created and its only data commit was made on 2022-09-02, before ChatGPT; there is no per-row date | apache-2.0 on the card; texts © authors | college admission essays, supplements, statements of purpose, scholarship essays |

How `essayforum` is cleaned:
- the preamble is cut off: the request ("Please give feedback…"), "Prompt: …", the prompt text, everything up to the
  `-----` separator;
- thanks at the end and everything after a late request are cut off;
- the space is restored in glued text like "end.Next";
- posts where the dataset glued words together when removing "/" (`andor`, `SirMadam`) are discarded;
- the `Correct Grammar` column (machine correction) is not used.

#### business_marketing (new genre)

| id | Source | Subtype | Date | License | What we take |
|---|---|---|---|---|---|
| `amazon_sports`, `amazon_toys`, `amazon_beauty` | `smartcat/Amazon_{Sports_and_Outdoors,Toys_and_Games,All_Beauty}_2018`, config `metadata` | product description | corpus: Amazon Review Data 2018 (Ni, Li, McAuley, EMNLP 2019), metadata collected in 2018 | McAuley lab, for research; no card | the `description` field: HTML → text, `*` bullets → separate lines |
| `c4_press` | `allenai/c4` realnewslike | press release | **row** `timestamp` (crawl Apr. 2019) | ODC-BY; © authors | pages with URL `press-release(s)`, `/press/`, `/news-releases/`, `/pressroom/`, `/media-releases/`, except hosts of government bodies and organizations such as the UN (`.gov`, `.mil`, `un.org`…: 1 050 discarded). The agency dateline (PRNewswire, Business Wire, GlobeNewswire, Marketwired, ACCESSWIRE, "… (FSCWire via Comtex) -") and the About / Contact / SOURCE / forward-looking tail are cut off |
| `c4_about` | `allenai/c4` `en`, every 16th of 1024 shards | website text | **row** `timestamp` (Apr. 2019) | ODC-BY; © owners | pages `/about(-us)/`, `/our-story/`, `/who-we-are/`, `/company/`, `/services/…` |
| `edgar_business`, `edgar_mdna` | `eloukas/edgar-corpus` (10-K, json) | business report | **row** `year` = filing year; 2001–2020 taken | Apache-2.0; the reports are public SEC documents | Item 1 (Business) and Item 7 (MD&A), from different samples of reports. Removed: headings (including all-caps ones ending in a period), table rows, lines where more than 20% of words contain digits, contacts and standard legal disclaimers (forward-looking, "should be read in conjunction…") |

#### homework_answer (new genre)

A direct explanatory answer to a study question, no shorter than 150 words.

| id | Source | Subtype | Date | License | What we take |
|---|---|---|---|---|---|
| `eli5` | `HuggingFaceGECLM/REDDIT_comments`, r/explainlikeimfive, files 0–15 (one stream per file) | eli5 | **row** `created_utc` < 2022 | Reddit/Pushshift; no license on the card | only top-level answers (`parent_id` = the post) with score ≥ 3; no bots or moderator templates |
| `askscience` | same, r/askscience | askscience | **row** `created_utc` < 2022 | same | same, plus no `AskScienceModerator` |
| `se_science` | `flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl`: physics, chemistry, biology, astronomy, earthscience | stackexchange | **corpus**: data files committed on 2021-07-11 from the archive.org dump of 2021-07-01 | CC BY-NC-SA 4.0 on the card; SE itself — CC BY-SA | the best-voted answer; quotes and indented code are removed; answers with LaTeX are cut by the markup filter |
| `se_humanities` | same: history, philosophy, english, linguistics, economics | stackexchange | same | same | same |

These sources do not overlap with `forum_qa`. That genre has r/AskHistorians, DIY, Fitness, LifeProTips,
EatCheapAndHealthy, Games and Pile StackExchange; possible overlaps of answers with StackExchange are removed by
global deduplication.

### Checked and rejected candidates

All were added to `REJECTED`, which now has 81 entries.

| Genre | Source (HF) | | Reason |
|---|---|---|---|
| academic | `timaeus/pile-pubmed_central` | ⚪ | the same PMC, but only the first 100 thousand Pile documents, without id, per-row date or license (the licenses include NC) |
| academic | `timaeus/pile-arxiv`, `suolyer/pile_arxiv` | ⚪ | `$…$` formulas in 95% of papers, YAML headers; the cut-out `\cite/\ref` break sentences. A window is found in only 30% of papers |
| academic | `common-pile/arxiv_papers` | ⛔ | `created` is the date of version v1, but the text is the latest version, downloaded in 2024. The date does not bound the text |
| academic | `ccdv/arxiv-summarization`, `ccdv/pubmed-summarization` (Cohan et al. 2018) | ⚪ | lowercased text, PTB tokenization, `@xcite`/`@xmath` |
| academic | `armanc/scientific_papers` | ❌ | loading script only |
| academic | `allenai/peS2o` (v2), `laion/Pes2oX-fulltext` | ⚪ | loads, but has no fields of study or per-paper license (these exist in the Common Pile copy) / a repackaging of the same |
| academic | `claran/modular-s2orc-parquet`, `timaeus/pile-philpapers` | ⚪ | no license, texts before 2000 are OCR / no paragraph splitting, running headers inside the text |
| academic | `AlgorithmicResearchGroup/s2orc_full`, `casperhansen/pmc-oa-markdown`, `SIRIS-Lab/scilake-fulltext-corpus` | ⛔ | no date |
| academic | `pszemraj/scientific_lay_summarisation-plos-norm` | ⚪ | 24.8 thousand PLOS articles that are already in PMC |
| essay | `realbenpope/PERSUADE_manageable` | ⚪ | the previous PERSUADE mirror without the prompt column; replaced |
| essay | `allenai/hippocorpus` | ❌ | a loading script that requires a manual download from a closed site |
| essay | r/tifu (`HuggingFaceGECLM/REDDIT_submissions`) | ⚪ | ~150 thousand dated personal stories, but these are humorous posts by adults on Reddit (profanity, TL;DR), not school narrative essays. **Left for the owner's decision** |
| essay | `dim/essayforum_raw_writing_10k` / `dim/essayforum_writing_prompts_6k` | ⚪ / ⛔ | 52% are IELTS argumentative essays, 20% are template chart descriptions, narrative ~1% / no dates |
| essay | `trentmkelly/pre-chatgpt-collected-student-writings` | ⛔ | school literary magazines from PDF: mostly poems and prose, several authors in one document; 40% of the text has no year, and there are 2022 issues |
| essay | `trentmkelly/lots-of-essays`, `artfultom/human-essays`, `thisisHJLee/human_essay`, `danibor/human-essay-*`, `polsci/ghostbuster-essay-cleaned`, `ChristophSchuhmann/essays-with-instructions` | ⛔ | repackagings of ASAP 2.0, PERSUADE, ELLIPSE and IvyPanda without dates; some come paired with AI text or with LLM instructions |
| essay | `ahs10/essays-big5`, `Ankonbh/aeon-essays-preprocessed`, `Xiaochr/Chinese-Student-English-Essay`, `UniversalCEFR/icle500_en` | ⚪ | stream of consciousness, not essays / magazine essays without date or license / gaokao letter-writing tasks, median 130 words / 500 argumentative essays, the CC0 is doubtful (ICLE is a paid corpus) |
| essay | `bea2019st/wi_locness` and copies, `matejklemen/clc_fce` | ❌ | loading scripts or tokenized sentence pairs. BAWE, MICUSP and TOEFL11 are not on HF |
| application | `qsardor/viorra-admissions-essays` | ⛔ | collected in 2026 from blogs, has entries after 2022, no dates, the text was "anonymized by an AI agent" |
| application | `Ayesha490/personalstatement`, `tekriwar/Statement_of_Purpose`, `DharamArora/GAMSAT-Essays` | ⛔ | LLM "mentor/answer" pairs / data for fine-tuning Llama-2 / instructions |
| application | cover letters: `kwanyick/*`, `lav-ml`, `dhruvvaidh`, `akhan02`, `justinthelaw`, `jenrajaseharan`, `Bharath1121` | ⛔ | copies and derivatives of the already rejected template set, letters rewritten by an LLM, synthetic data, or only a handful of rows |
| application | admission subreddits in `anhchanghoangsg/reddit_pushshift_dataset_cleaned` | ⚪ | dates exist, but there are only ~400 essays, usually quoted via `>` inside a request |
| business_marketing | metadata of `McAuley-Lab/Amazon-Reviews-2023` and copies | ⛔ | product pages were downloaded in 2023 |
| business_marketing | `milistu/Amazon_*_2014`, `genghaobuaa/Amazon-Review-Data-2018`, `huseinzol05/translated-metadata-amazon-review`, `calmgoose/amazon-product-data-2020` | ⚪ | overlaps with the 2018 data / a single category / rows in Python-repr format with a Malay translation / bullets separated by "\|" |
| business_marketing | `c3po-ai/edgar-corpus`, `BEE-spoke-data/edgar-corpus`, `TeraflopAI/8-K-text` | ⚪ | the same EDGAR-CORPUS / 8-K cover pages without EX-99 press releases |
| business_marketing | `vblagoje/cc_news` (press-agency domains) | ⚪ | only ~3.3 thousand rows, of which 1.3 thousand have no date; C4 has roughly 10 times more such pages |
| business_marketing | `victor/real-or-fake-fake-jobposting-prediction` | ⚪ | EMSCAD job postings 2012–14 (CC0): `#URL_…#` tokens and glued sentences in the text. A possible addition later |
| business_marketing | C4 `en`, URL `/products/`, `/shop/` | ⚪ | mostly bulleted lists; Amazon already provides product descriptions |
| business_marketing | Kickstarter sets, Steam sets, synthetic advertising | ⚪ / ⛔ | descriptions ≤ 135 characters / descriptions from 2023–26 and LLM rewrites / synthetic |
| homework_answer | `vblagoje/lfqa`, `Pavithree/eli5` (+ askScience, askHistorians) | ⚪ | ELI5/KILT without dates and with the same comments as in GECLM / 2022 re-uploads |
| homework_answer | `defunct-datasets/eli5`, `eli5_category` | ❌ | loading scripts |
| homework_answer | `DataDump1/{askphilosophy, AskEngineers, AskPhysics, askmath, NoStupidQuestions}_reddit` | ❌ | restricted access (403), uploaded in 2025, no card |
| homework_answer | GECLM: science, space, philosophy, history; SE: skeptics, law, politics, academia | ⚪ | link discussions, not question-answer / argument and advice instead of study answers |

### Achieved composition within genres and targets

| Genre | Subtype | Texts | Share | Production target |
|---|---|---|---|---|
| academic | papers: PMC 10 200 + peS2o social 9 000 + peS2o stem 7 200 | 26 400 | **88.0%** | ~88% |
| academic | abstracts: PubMed 1 500 + arXiv 1 200 + NIH 900 | 3 600 | **12.0%** | ~12% |
| essay | argumentative: CMV 8 957, args.me 4 500, C4 op-ed 4 200, PERSUADE 3 000, ELLIPSE 2 893 | 23 550 | **78.5%** | 24% |
| essay | expository: PERSUADE 3 165, ASAP 518 | 3 683 | **12.3%** | 49% |
| essay | literary / text analysis: PERSUADE "Venus" 1 850, ASAP 778 | 2 628 | **8.8%** | 10% |
| essay | narrative: ASAP 139 | 139 | **0.5%** | 17% |
| application | undergraduate admission essays and supplements* | ~20 400 | ~97% | — |
| application | statements of purpose / scholarship / cover letters* | ~320 / ~200 / ~100 | 1.5 / 0.9 / 0.5% | — |
| business_marketing | product descriptions (Amazon 2018) | 9 593 | 32% | — |
| business_marketing | business reports (10-K: Item 1 — 4 500, MD&A — 3 000) | 7 500 | 25% | — |
| business_marketing | press releases (C4) | 6 907 | 23% | — |
| business_marketing | website texts "about the company / services" (C4) | 6 000 | 20% | — |
| homework_answer | StackExchange (sciences 7 500, humanities 6 000) | 13 500 | 45% | — |
| homework_answer | r/explainlikeimfive / r/askscience | 10 500 / 6 000 | 35 / 20% | — |

\* An estimate by keywords on the final pool. Production gave no breakdown within application,
business_marketing and homework_answer: the shares there are registry weights set by me.

**Why essays fall short of the target.**
- All non-argumentative sources are used **in full**: PERSUADE "explanatory" and "Venus", ASAP sets
  3–8 (with restrictions: no placeholders, no shorter than 150 words).
- Only 139 narrative school essays from before 2022 without anonymizing placeholders were found.
- The remainder up to 30 000 is topped up with argumentative essays: CMV first, then as before.
- Within a genre, slots replicate the strata frequencies of the pool. So in a period, essays will be ~78% argumentative.

### Full build (`var/pool-full-v2`)

```bash
.venv/bin/python scripts/build_pool_dev.py --out /workspace/sn32_owner/humanizer_eval/var/pool-full-v2 \
    --per-genre 30000 --min-cluster-size 300 --stride-divisor 10 --workers 48 [--reuse-candidates]
```

Build of 2026-11-09. 40 sources, target 30 000 per genre, cluster threshold **300** (D-51), pool window 150–350,
cache 100–500.
- **320 021 texts**, `content_sha256` `ef444753…2051a064` (stored in `manifest.json`).
- 1.2 GB on disk, of which the candidate cache `candidates/` is ~1 GB. HF cache 199 MB. Everything is on `/workspace`, the root
  disk was not touched.
- Time:
  - stage 1 — the old 20 sources in the new cache window, 36 min;
  - streaming reads of the 21 new sources — 1–5 min each, in parallel, 48 workers with one BLAS thread each;
  - build from the cache (deduplication of ~0.5 million candidates, windows, clusters, writing) — 18 min.
- Stage logs are alongside (`build.stage*.log`). The final one is `build.log`.
- Fixed along the way:
  - PMC crashed on a date without zeros; it was switched to `row_parser`;
  - EDGAR still had all-caps headings, press releases still had the Comtex dateline;
  - press releases of government bodies (FBI, UN) are now excluded;
  - peS2o placeholder dates 0001-01-01 and 1970-01-01 are no longer treated as a date.
- Intermediate statistics are saved in `stage1/` and `stage3-before-press-fix/`.

**By genre.** k — number of clusters ≥ 300. The 300–350 bin is present in all genres.

| Genre | Texts | 150–199 | 200–249 | 250–299 | 300–350 | formal | informal | Mean words | k | Min./median/max. cluster |
|---|---|---|---|---|---|---|---|---|---|---|
| essay | 30 000 | 9 791 | 8 746 | 6 751 | 4 712 | 19 935 | 10 065 | 234 | 29 | 339 / 891 / 2279 |
| article_blog | 30 000 | 8 842 | 9 414 | 7 397 | 4 347 | 12 805 | 17 195 | 236 | 33 | 506 / 787 / 1765 |
| academic | 30 000 | 10 340 | 8 479 | 6 640 | 4 541 | 29 978 | 22 | 232 | 33 | 464 / 900 / 1644 |
| news | 30 000 | 8 067 | 9 255 | 7 811 | 4 867 | 25 815 | 4 185 | 240 | 33 | 338 / 874 / 1901 |
| email_letter | 28 993 | 10 306 | 8 390 | 6 117 | 4 180 | 24 283 | 4 710 | 231 | 32 | 306 / 890 / 1564 |
| review | 30 000 | 13 135 | 8 234 | 5 216 | 3 415 | 6 365 | 23 635 | 221 | 31 | 381 / 977 / 1673 |
| forum_qa | 30 000 | 14 882 | 7 836 | 4 635 | 2 647 | 12 319 | 17 681 | 214 | 33 | 502 / 908 / 1902 |
| creative | 30 000 | 7 548 | 9 659 | 7 791 | 5 002 | 5 928 | 24 072 | 241 | 33 | 316 / 819 / 2011 |
| **application** | **21 028** | 5 587 | 6 661 | 5 442 | 3 338 | 11 171 | 9 857 | 239 | 22 | 546 / 848 / 1955 |
| **business_marketing** | 30 000 | 9 528 | 9 011 | 6 934 | 4 527 | 27 269 | 2 731 | 234 | 32 | 436 / 966 / 1423 |
| **homework_answer** | 30 000 | 11 696 | 8 551 | 5 952 | 3 801 | 22 022 | 7 978 | 226 | 33 | 446 / 908 / 2320 |
| **total** | **320 021** | 109 722 | 94 236 | 70 686 | 45 377 | | | | **344** | all ≥ 300 ✅ |

- `check_clusters`: 0 problems. All 344 clusters are ≥ 300, all labels pass `check_label`.
- academic: median 226 words (~213 in production). The 300–350 bin grew from 7.8% to 15.1%, because windows of 1–3
  paragraphs are longer than abstracts.
- The 300–350 bin by genre: from 2 647 (forum_qa, 8.8%) to 5 002 (creative, 16.7%). The new genres have
  12.7–15.9%.

**New sources.** Rows scanned → candidates that passed the filters → in the pool after deduplication, window and quotas.

| Source | Genre | Rows | Candidates | In pool | Dates in pool | Main rejection reasons |
|---|---|---|---|---|---|---|
| pmc_papers | academic | 17 334 | 13 260 | 10 200 | 1989 … 2021-08, median 2017-10 | date_too_late 2 307, empty¹ 1 756 |
| pes2o_social | academic | 116 855 | 11 700 | 9 000 | 1972 … 2021-12, median 2019-03 | other_field 94 515, biomedical_field 5 950, date_too_late 4 347 |
| pes2o_stem | academic | 61 317 | 9 360 | 7 200 | 1938 … 2021-12, median 2018-12 | other_field 41 970, biomedical_field 4 210, date_too_late 3 840, empty¹ 1 899 |
| persuade2_expository | essay | 25 996 | 3 178 | 3 165 | corpus | other_prompt 22 485, placeholder 205, repetitive 136 |
| persuade2_analysis | essay | 25 996 | 1 851 | 1 850 | corpus | other_prompt 24 134 |
| persuade2 (new mirror) | essay | 6 735 | 4 496 | 3 000 | corpus | other_prompt 1 384, placeholder 764 |
| asap_literary | essay | 21 448 | 2 072² | 778 | corpus (2012) | placeholder 3 666, too_short 3 063, window_short 1 291 |
| asap_expository | essay | 21 448 | 979² | 518 | corpus (2012) | placeholder 1 727, window_short 461 |
| asap_narrative | essay | 21 448 | 254² | 139 | corpus (2012) | placeholder 3 205, too_short 209 |
| essayforum | application | 25 571 | 22 301² | 21 028 | corpus (2022-09) | too_short 2 439, window_short 1 258, slash_merged 555 |
| amazon_sports | business_marketing | 47 874 | 9 418² | 4 600 | corpus (2018) | too_short 32 684, empty 5 103, exact/near_dup 1 245 |
| amazon_toys | business_marketing | 43 712 | 11 408² | 3 858 | corpus (2018) | too_short 28 017, window_short 5 558, exact_dup 1 774 |
| amazon_beauty | business_marketing | 32 891 | 2 496² | 1 135 | corpus (2018) | empty 17 793, too_short 12 320 |
| c4_press | business_marketing | 3 339 781 | 8 338 | 6 907 | 2019-04 (crawl) | not_press_url 3.3 million, pii_url 4 293, public_sector 1 050 |
| c4_about | business_marketing | 1 442 633 | 9 877 | 6 000 | 2019-04 (crawl) | not_about_url 1.43 million, too_short 2 695, pii_url 521 |
| edgar_business | business_marketing | 7 003 | 5 887 | 4 500 | 2001 … 2020 (evenly across 2001–2015) | empty 1 019, pii_url 210 |
| edgar_mdna | business_marketing | 4 745 | 3 910 | 3 000 | 2001 … 2020 | empty 708, too_short 114 |
| eli5 | homework_answer | 1 066 130 | 23 931² | 10 500 | 2011-07 … 2021-10, median 2015-09 | not_top_level 680 450, low_score 179 735, bot 114 624, too_short 53 053, mod_template 12 785 |
| askscience | homework_answer | 553 467 | 12 846² | 6 000 | 2010-04 … 2020-07 | not_top_level 348 767, bot 116 705, low_score 45 450 |
| se_science | homework_answer | 34 235 | 14 037² | 7 500 | corpus (2021-07) | too_short 10 511, markup (LaTeX) 9 022 |
| se_humanities | homework_answer | 18 947 | 10 444² | 6 000 | corpus (2021-07) | too_short 6 533, markup 1 275 |

¹ "empty" for papers means a paper without a single window of 1–3 consecutive clean body paragraphs (no IMRaD headings,
formulas, tables, results only).
² Candidates are counted in the 100–500 cache window, so they also include short ones (100–149 words), which go into the pool
only with a lower window (`window_short`).

The old 20 sources gave the same volumes as in `var/pool-full`. Exceptions: the academic and essay weights were lowered,
so less is taken from the cache; email_letter has 28 993 texts (see above).

**Clusters and sources.**
- **academic**: only 2 of 33 clusters are > 90% from a single source. Examples: "patients, treatment,
  clinical" (1 644, PMC), "students, learning, teachers" (1 232, peS2o social), "surface, materials, reaction"
  (1 325, peS2o stem), "stars, star, galaxies" (464, arXiv abstracts).
- **essay**: 27 of 29 clusters are homogeneous by subtype. The non-argumentative subtypes are prompt clusters:
  "car, cars, usage" (1 929), "planet, author, surface" (1 979), "cyclist, water, setting" (354, ASAP 3).
  That is, the slot's cluster effectively also determines the essay subtype.
- **application**: 22 clusters from a single source, by essay topic: "college, students, school", "parents,
  family, school", "community, school, club", "art, design, drawing", "computer, computers, science".
- **business_marketing**: 11 of 32 clusters from a single source. Examples: "services, products, business"
  (10-K), "play, fun, children" (toys), "blade, steel, knife" (sporting goods), "data, solutions, software"
  (press releases), "food, art, family" (websites).
- **homework_answer**: 2 of 33 clusters from a single source. Examples: "philosophy, true, logic" (SE),
  "speed, force, moving" (askscience), "orbit, planet, solar", "computer, file, code" (ELI5), "vowel,
  vowels, sound" (SE linguistics).

### Blockers, risks, what the owner needs to decide

1. **Essay composition (the main issue).** There are few non-argumentative school essays with a provable date: about 6.5 thousand after
   filters, and all of them are tied to 9 prompts (3 PERSUADE and 6 ASAP sets). Expository came out at 12% versus a target of 49%, narrative at 0.5%
   versus 17%. Options:
   - (a) accept as is: essays in a period will be ~78% argumentative;
   - (b) cut the genre down, for example to 12 thousand with 50% argumentative. Then essay topics will collapse to a few
     prompts;
   - (c) take adult personal stories from r/tifu (~150 thousand, with dates) as a replacement for narrative. The register is different,
     so this is postponed until a decision;
   - (d) licensed corpora (BAWE, MICUSP, ICLE/LOCNESS) — they are not on HF.
2. **application: the genre is filled, but the date proof is weaker than for the others.** There is a single source: essayforum.
   - **Date.** It exists only at the corpus level: the repository was published on 2022-09-02. This passes rule D-50
     (the corpus was released before November 2022, like ELLIPSE), but posts from January–August 2022 cannot be ruled out: they
     are later than `CUTOFF`, although before ChatGPT.
   - **Cleaning artifacts of the dataset itself.** Line breaks were removed, so each text is a single paragraph. The
     "/" character was removed: known merges are filtered out, but "24/7" → "247" may remain. ".I" merges
     are fixed. Occasionally a request remains at the start ("I would truly appriciate it.").
   - **Volume.** 21 028 texts instead of 30 000: the source is exhausted.
   - **Composition.** ~97% are undergraduate admission essays. Motivation letters and cover letters are ~0.5%: there is no acceptable
     source of such letters from before 2022.
   - **A decision is needed:** keep the genre on these terms or remove it under the "no source — no genre" rule.
3. **business_marketing.** There are Amazon descriptions copied from encyclopedias; book blurbs; personal-blog
   "about me" pages among the "about the company" pages; NGO statements among press releases; the legal language of
   10-K (MD&A of earlier years). Amazon dates are confirmed only at the corpus level (collected in 2018).
4. **homework_answer.** ELI5 answers are conversational; some of them are a comment rather than an explanation. The date for
   StackExchange (flax) is only at the corpus level (July 2021). Reddit-markdown escaping leaves
   stray "\" characters (for example, "current\resistance"). This is the old `clean_reddit` function shared with `reddit`/`cmv`,
   so it was not touched.
5. **academic.**
   - peS2o occasionally has PDF-extraction scars: lost hyphens ("selfefficacy"), running headers
     like "PLOS" in the middle of a sentence.
   - 42% of peS2o social windows are from 2020–21, with many COVID topics.
   - PMC years are uneven: the shards are ordered by id, and 2005–2010 is almost absent.
   - The "results" section is not taken (it has numbers).
6. **As before:** some of the corpora are probably in detector training data (PERSUADE, ELI5, Reddit).
   Licenses need to be checked by lawyers before production: Amazon — for research, ASAP — competition data,
   essayforum — authors' texts under an apache-2.0 card.
7. **Generation templates** for `application`, `business_marketing`, `homework_answer` will be added by the owner
   (`period/`). The period, gate and scoring code was not changed.
