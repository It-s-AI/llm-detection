"""Topic clusters within each genre and generic cluster labels (D-51).

* ``Embedder`` is a protocol; the dev implementation is local TF-IDF + TruncatedSVD (no API, no torch).
  A remote embedder (OpenRouter, D-90) can be plugged in later with the same two methods.
* Clusters: k-means with ``k = n // (min_size * size_factor)``, then the smallest cluster is dissolved
  into its points' nearest remaining centroids until every cluster has ``>= min_size`` texts.
* Labels: top terms by class-based TF-IDF, restricted to *generic* words: lowercase alphabetic, not stop
  words, frequent inside the cluster, and not proper nouns. A word counts as a proper noun when it is
  capitalized in ``>= PROPER_RATIO`` of its non-sentence-initial occurrences in the genre (names, places,
  brands, months, nationalities). ``check_label``/``check_clusters`` re-verify all of this automatically.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

PROPER_RATIO = 0.3
_LABEL_TERM = re.compile(r"^[a-z]{3,}$")
# words of >= 3 letters, keeping contractions whole ("don't" must not become the label term "don")
_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z'\u2019]+[a-zA-Z]\b"  # noqa: S105 (regex, not a secret)

# generic words that carry no topic even though they are not in sklearn's stop list
_EXTRA_STOP = frozenset(
    """
    said says say would could should also like just one two three get got really much many make made
    well even know think time year years people thing things way new use used using good great dont
    didnt doesnt isnt wasnt im ive id youre theyre thats going want see need lot lots may might must
    shall per etc however therefore thus first second last next back still yet ever every something
    anything nothing someone anyone everyone day days week weeks month months today right left take
    took come came give gave look looked feel felt sure let told tell put went done able around away
    another little big long high large small number part place point case fact end life world work
    best better bit far yes okay always never often sometimes maybe probably actually pretty quite
    did does doing mean means meant knew know thought try tried trying looking getting wanted started
    began seen heard asked happened showed based times different example old whole kind sort yeah
    fuck fucking fucked shit shitty damn crap ass bitch
    """.split()
)


def stop_words() -> frozenset[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(ENGLISH_STOP_WORDS) | _EXTRA_STOP


# --------------------------------------------------------------------------- embedders
@runtime_checkable
class Embedder(Protocol):
    name: str

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray: ...

    def transform(self, texts: Sequence[str]) -> np.ndarray: ...


class TfidfSvdEmbedder:
    """Local, dependency-light embedder: TF-IDF (sublinear, English stop words) -> TruncatedSVD -> L2."""

    name = "tfidf-svd"

    def __init__(
        self,
        n_components: int = 100,
        max_features: int = 50_000,
        min_df: int = 2,
        max_df: float = 0.5,
        seed: int = 0,
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.seed = seed
        self._tfidf = None
        self._svd = None

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        n = len(texts)
        self._tfidf = TfidfVectorizer(
            sublinear_tf=True,
            stop_words=sorted(stop_words()),
            token_pattern=_TOKEN_PATTERN,
            min_df=self.min_df if n >= 50 else 1,
            max_df=self.max_df if n >= 50 else 1.0,
            max_features=self.max_features,
        )
        x = self._tfidf.fit_transform(texts)
        k = min(self.n_components, x.shape[1] - 1, n - 1)
        if k < 2:
            self._svd = None
            return normalize(x.toarray())
        self._svd = TruncatedSVD(n_components=k, random_state=self.seed)
        return normalize(self._svd.fit_transform(x))

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.preprocessing import normalize

        if self._tfidf is None:
            raise RuntimeError("fit_transform first")
        x = self._tfidf.transform(texts)
        return normalize(self._svd.transform(x) if self._svd is not None else x.toarray())


# --------------------------------------------------------------------------- proper nouns
_WORD = re.compile(r"[A-Za-z]+")
_SENT_END = set(".!?:;\n")
_OPENERS = set(" \t\"'([{“‘*-")


def capitalization_ratios(texts: Iterable[str]) -> dict[str, float]:
    """For each lowercase word: share of its informative occurrences that are capitalized.

    Sentence-initial occurrences say nothing and are skipped -- except on short all-capitalized lines
    (greetings, sign-offs, signatures such as "Phillip Allen"), which are counted as capitalized: names
    in letters often appear only there.
    """
    total: Counter[str] = Counter()
    caps: Counter[str] = Counter()
    for t in texts:
        for line in t.split("\n"):
            words = _WORD.findall(line)
            if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words):
                for w in words:
                    total[w.lower()] += 1
                    caps[w.lower()] += 1
                continue
            for m in _WORD.finditer(line):
                j = m.start() - 1
                while j >= 0 and line[j] in _OPENERS:
                    j -= 1
                if j < 0 or line[j] in _SENT_END:
                    continue  # sentence-initial: capitalization says nothing
                w = m.group()
                lw = w.lower()
                total[lw] += 1
                if w[0].isupper():
                    caps[lw] += 1
    return {w: caps[w] / c for w, c in total.items()}


def proper_from_ratios(ratios: Mapping[str, float], ratio: float = PROPER_RATIO) -> set[str]:
    return {w for w, r in ratios.items() if r >= ratio}


def proper_terms(texts: Iterable[str], ratio: float = PROPER_RATIO) -> set[str]:
    return proper_from_ratios(capitalization_ratios(texts), ratio)


# --------------------------------------------------------------------------- clustering
@dataclass
class ClusterInfo:
    id: str
    genre: str
    label: str
    size: int
    top_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenreClusters:
    genre: str
    assignments: list[str]  # cluster id per input text (same order)
    clusters: list[ClusterInfo]
    embedder: str


def merge_small_clusters(x: np.ndarray, labels: np.ndarray, min_size: int) -> np.ndarray:
    """Dissolve the smallest cluster into the nearest other centroids until all are >= min_size."""
    labels = labels.copy()
    while True:
        ids, sizes = np.unique(labels, return_counts=True)
        if len(ids) <= 1 or sizes.min() >= min_size:
            return labels
        small = ids[np.argmin(sizes)]  # ties -> lowest id, deterministic
        others = [c for c in ids if c != small]
        cents = np.stack([x[labels == c].mean(axis=0) for c in others])
        norms = np.linalg.norm(cents, axis=1, keepdims=True)
        cents = cents / np.where(norms == 0, 1, norms)
        idx = np.where(labels == small)[0]
        nearest = np.argmax(x[idx] @ cents.T, axis=1)
        labels[idx] = np.array(others)[nearest]


def cluster_terms(
    texts: Sequence[str],
    labels: np.ndarray,
    n_top: int = 12,
    caps: Mapping[str, float] | None = None,
) -> dict[int, list[str]]:
    """Generic top terms per cluster by class-based TF-IDF (BERTopic-style), proper nouns excluded.

    ``caps``: capitalization ratios (``capitalization_ratios``), ideally over the whole pool; words never
    seen in an informative position count as proper (excluded)."""
    from sklearn.feature_extraction.text import CountVectorizer

    stops = stop_words()
    cv = CountVectorizer(
        lowercase=True,
        token_pattern=_TOKEN_PATTERN,
        stop_words=sorted(stops),
        min_df=2 if len(texts) >= 20 else 1,
    )
    x = cv.fit_transform(texts).tocsr()
    vocab = cv.get_feature_names_out()
    caps = caps if caps is not None else capitalization_ratios(texts)
    allowed = np.array(
        [bool(_LABEL_TERM.match(t)) and t not in stops and caps.get(t, 1.0) < PROPER_RATIO for t in vocab]
    )
    ids = sorted(set(labels.tolist()))
    counts = np.vstack([np.asarray(x[labels == c].sum(axis=0)).ravel() for c in ids]).astype(float)
    dfs = np.vstack([np.asarray((x[labels == c] > 0).sum(axis=0)).ravel() for c in ids])
    sizes = np.array([(labels == c).sum() for c in ids])
    tf = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    avg = counts.sum() / max(len(ids), 1)
    idf = np.log1p(avg / np.maximum(counts.sum(axis=0), 1))
    score = tf * idf
    out: dict[int, list[str]] = {}
    for row, c in enumerate(ids):
        min_df = max(2, int(0.05 * sizes[row]))
        ok = allowed & (dfs[row] >= min_df)
        order = np.argsort(-score[row], kind="stable")
        out[c] = [str(vocab[j]) for j in order if ok[j]][:n_top]
    return out


def make_labels(terms: Mapping[int, list[str]], n_terms: int = 3) -> dict[int, str]:
    """Join the first ``n_terms`` terms; if a label repeats within the genre, swap in the next terms."""
    used: set[str] = set()
    out: dict[int, str] = {}
    for c in sorted(terms):
        cand = terms[c]
        label = ", ".join(cand[:n_terms]) if cand else "general"
        extra = n_terms
        while label in used and extra < len(cand):
            label = ", ".join([*cand[: n_terms - 1], cand[extra]])
            extra += 1
        if label in used:
            label = f"{label}, misc" if label != "general" else "general, misc"
        used.add(label)
        out[c] = label
    return out


def cluster_genre(
    genre: str,
    texts: Sequence[str],
    *,
    min_size: int = 300,
    size_factor: int = 3,
    embedder: Embedder | None = None,
    seed: int = 0,
    n_label_terms: int = 3,
    caps: Mapping[str, float] | None = None,
) -> GenreClusters:
    """Cluster one genre's texts; every returned cluster has >= ``min_size`` texts when n >= min_size."""
    if not texts:
        raise ValueError(f"no texts for genre {genre!r}")
    emb = embedder or TfidfSvdEmbedder(seed=seed)
    x = emb.fit_transform(texts)
    n = len(texts)
    k = max(1, n // max(1, min_size * size_factor))
    if k == 1:
        raw = np.zeros(n, dtype=int)
    else:
        from sklearn.cluster import KMeans, MiniBatchKMeans

        algo = (
            MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=4096)
            if n > 50_000
            else KMeans(n_clusters=k, random_state=seed, n_init=5)
        )
        raw = algo.fit_predict(x)
    raw = merge_small_clusters(x, raw, min_size)
    # renumber by size (desc), ties by first appearance -> stable ids
    sizes = Counter(raw.tolist())
    first = {c: i for i, c in reversed(list(enumerate(raw.tolist())))}
    order = sorted(sizes, key=lambda c: (-sizes[c], first[c]))
    remap = {c: i for i, c in enumerate(order)}
    labels = np.array([remap[c] for c in raw.tolist()])
    terms = cluster_terms(texts, labels, caps=caps)
    names = make_labels(terms, n_label_terms)
    clusters = [
        ClusterInfo(
            id=f"{genre}.c{c:02d}",
            genre=genre,
            label=names[c],
            size=int((labels == c).sum()),
            top_terms=terms[c][:10],
        )
        for c in range(len(order))
    ]
    return GenreClusters(
        genre=genre,
        assignments=[f"{genre}.c{c:02d}" for c in labels.tolist()],
        clusters=clusters,
        embedder=getattr(emb, "name", type(emb).__name__),
    )


# --------------------------------------------------------------------------- automatic checks
def check_label(label: str, proper: Iterable[str] = (), max_terms: int = 5) -> list[str]:
    """Violations of the 'generic label' rule (empty list = OK)."""
    problems: list[str] = []
    if not label or not label.strip():
        return ["empty"]
    if any(ch.isdigit() for ch in label):
        problems.append("digit")
    if any(ch.isupper() for ch in label):
        problems.append("uppercase")
    terms = [t.strip() for t in label.split(",")]
    if len(terms) > max_terms:
        problems.append("too_many_terms")
    stops = stop_words() - {"misc", "general"}
    proper_set = set(proper)
    for t in terms:
        if t in {"misc", "general"}:
            continue
        if not _LABEL_TERM.match(t):
            problems.append(f"non_generic_term:{t}")
        elif t in proper_set:
            problems.append(f"proper_noun:{t}")
        elif t in stops:
            problems.append(f"stopword:{t}")
    return problems


def check_clusters(
    rows: Iterable[Mapping],
    min_size: int,
    proper_by_genre: Mapping[str, Iterable[str]] | None = None,
) -> list[str]:
    """Check every (genre, cluster) in pool rows: size >= min_size, one consistent generic label per
    cluster, labels unique within a genre. Returns human-readable violations (empty = OK)."""
    sizes: Counter[tuple[str, str]] = Counter()
    labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    problems: list[str] = []
    for r in rows:
        g, cid = r["genre"], r.get("cluster_id") or ""
        if not cid:
            problems.append(f"unassigned:{r.get('text_id', '?')}")
            continue
        sizes[(g, cid)] += 1
        labels[(g, cid)].add(r.get("cluster_label") or "")
    seen: dict[str, dict[str, str]] = defaultdict(dict)
    for (g, cid), n in sorted(sizes.items()):
        if n < min_size:
            problems.append(f"size:{cid}={n}<{min_size}")
        labs = labels[(g, cid)]
        if len(labs) != 1:
            problems.append(f"inconsistent_label:{cid}")
            continue
        lab = next(iter(labs))
        proper = (proper_by_genre or {}).get(g, ())
        for p in check_label(lab, proper):
            problems.append(f"label:{cid}:{p}")
        if lab in seen[g]:
            problems.append(f"duplicate_label:{cid}={seen[g][lab]}")
        seen[g][lab] = cid
    return problems
