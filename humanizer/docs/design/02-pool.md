# 02. Human pool (dev scale)
Status: draft, implemented autonomously
Implements decisions: D-50, D-51, D-54, D-55 (and also D-59 — English only)

## Goal
Build a pool of **guaranteed human** texts (published before 2022-01-01) across the product's genres
(essay, article_blog, academic, news, email_letter, review, forum_qa, creative; since 2026-11, following the production
distribution, also application, business_marketing, homework_answer — see report 02, section
"Pool v2") and
public metadata for them. From the metadata, the period builder (phase 3) selects "slots" for generating
AI inputs; from the texts themselves, the human reference set H with the same counts per group (D-54). At this phase — dev scale:
a few thousand texts per genre, free (HF + local embeddings), no LLM.

## Inputs and outputs
- **Input:** a registry of corpora on Hugging Face (`pool/sources.py`): dataset id/config/split, text field,
  genre, license, **date evidence** — a per-row timestamp (filter `< 2022-01-01`) or the corpus release
  date. Only parquet/csv/jsonl without loading scripts, streaming reads.
- **Output** (`var/pool/<build>/`, not committed to git):
  - `pool.parquet` — texts + metadata; `metadata.parquet` — the same **without text** (the public part);
  - `clusters.json` (clusters: id, label, size, top terms), `stats.json`, `manifest.json` (config,
    sources, versions, checksum), `samples.html` (~20 random texts per genre for review).

## Pipeline
1. **Reading and date.** A row with no date, or with a date ≥ 2022 when a per-row field exists, is dropped.
2. **Cleaning** (`clean.py`): source rules (quotes/forwards in emails, `<br />`, Reddit markdown,
   headers), NFKC and removal of invisible characters, cutting out boilerplate lines ("Photo by … on Unsplash",
   "Click here…"); rejection: not English (langid, p ≥ 0.9), code/markup/LaTeX, anonymization
   placeholders (`Generic_Name`, `@CAPS1`), PII (e-mail, phone, URL → **rejection**, no masks —
   a mask by itself would distinguish the reference set from AI texts), lists/tables, repetitions, all caps.
3. **Excerpts** (D-55): a document of 150–350 words is taken whole; a long one is cut into contiguous
   excerpts along paragraph boundaries (for a huge paragraph — along sentences) with a target length drawn from
   N(230, 55), clipped to [150, 350]; by default one excerpt per document (diversity).
   *Since 2026-11:* candidates are collected and cached in a wide window of **100–500 words** (a long document yields
   the longest prefix of whole paragraphs ≤ 500), and the pool window (`BuildConfig.min_words/max_words`,
   `--min-words/--max-words`, default 150–350) is applied at build time (`clean.fit_window`: the same
   cut as before, plus repeated quality/PII checks of the piece). Changing the window within 100–500 takes
   only `--reuse-candidates`, without re-streaming; length bins are built from the window (50 words each).
   Papers (academic) are cut differently: a window of 1–3 consecutive paragraphs of the paper body (introduction, methods,
   discussion) starting from a random paragraph.
4. **Deduplication**: exact (hash of the normalized text) + near-duplicates (MinHash LSH, word 5-shingles,
   Jaccard ≥ 0.8), globally across the whole pool.
5. **Metadata** (`metadata.py`): `text_id` = sha256 of the text (20 hex), source, genre, length (bins
   150–199/200–249/250–299/300–350), register formal/informal (deterministic heuristic: contractions,
   I/you, "!", colloquial words versus nominalizations and bookish connectives; threshold 0).
6. **Clusters** (`cluster.py`): an `Embedder` protocol; in dev — TF-IDF + TruncatedSVD (local, no API;
   a remote embedder is plugged in later via the same protocol). MiniBatch/KMeans within a genre,
   k = n / (3·min_size), then small clusters are merged into the nearest ones until **each is ≥ min_size**
   (full scale 300 per D-51, dev — configurable, default 50). The label is 3 common words by
   c-TF-IDF with a filter: only lowercase alphabetic words, no stop words, no names/place names (share of
   capitalized spellings not at the start of a sentence < 0.3), no digits. `check_clusters()` automatically
   checks sizes and labels.

## Interfaces
- `sources.SOURCES: list[Source]`, `sources.REJECTED: list[RejectedSource]` (what was tried and why it was not used).
- `clean.clean_document(text, source) -> CleanResult`, `clean.excerpt(text, key) -> list[str]`,
  `clean.Deduplicator`.
- `metadata.length_bin(n)`, `metadata.register(text)`, `metadata.text_id(text)`.
- `cluster.TfidfSvdEmbedder`, `cluster.cluster_genre(...)`, `cluster.check_label(...)`, `cluster.check_clusters(...)`.
- `store.PoolStore.load(path)`; `store.sample_slots(meta, n, genre_proportions, seed) -> list[Slot]`, where
  `Slot = (genre, cluster_id, cluster_label, length_bin, register)` — **no text and no text_id** (D-51);
  `store.sample_reference(pool, slot_counts, seed, exclude_ids=()) -> list[ReferenceText]` (D-54).
- Sampling is deterministic and does not depend on the numpy version: ranking by `sha256(seed ‖ label ‖ text_id)`;
  distribution across genres — the largest remainder method; within a genre the slots reproduce the pool's stratum frequencies.
- `scripts/build_pool_dev.py` — build, statistics, `samples.html`.

## What we do not do (boundaries)
Million-text scale and remote embeddings (OpenRouter) — later, via the same `Embedder`; topics from
It's AI logs (D-52) — phase 3; languages other than English (D-59); LLM labeling of genre/register;
legal review of licenses (we record what is declared).

## Tests and acceptance criteria
Offline tests (`tests/pool/`): language, duplicates and near-duplicates, length, PII, excerpt boundaries (by paragraphs,
150–350 words), length bins, register on obvious examples, clusters ≥ min_size and labels without names/digits,
sampling determinism and exact counts per stratum, `sample_slots` never returns text/ID.
Live source-loading tests — `@pytest.mark.live`. Dev build criterion: ≥ 2000 texts in every
available genre, all clusters ≥ min_size, all labels pass the check.

## Risks and open questions
- Essays and letters: essays — only school ones (PERSUADE 2.0, ELLIPSE), letters — only Enron (business,
  2000–2001); we found no personal letters or university essays dated before 2022.
  *Q-11 (2026-11-09):* added r/changemyview, op-eds and letters to the editor from C4 realnewslike, args.me;
  still no personal correspondence — see report `docs/reports/02-pool.md`.
- Public corpora are probably in the detectors' training data — the reference set may be "too easily
  human"; comparison with production genres — after the It's AI logs.
- The register heuristic is crude; at full scale, replace it or calibrate it on labeled data.
- Licenses of many HF mirrors are not specified; production needs a legal review.
