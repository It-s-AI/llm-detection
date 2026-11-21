"""Pool storage (parquet + JSON sidecars) and deterministic stratified sampling.

D-51: ``sample_slots`` returns *only* public metadata tuples ``(genre, cluster_id, cluster_label,
length_bin, register)`` -- never a text and never a text id -- so AI-input generation cannot depend on
any specific human text.

D-54: ``sample_reference`` draws human texts with exactly the requested count per stratum (slot), from
an independent hash stream, optionally excluding ids (e.g. texts already used in earlier periods).

Determinism without numpy: every choice ranks candidates by ``sha256(seed | tag | text_id)``, so anyone
holding the published seed and pool metadata can recompute the draw with a few lines of any language.
Genre counts use the largest-remainder method (ties broken by genre name).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

from humanizer_eval.pool.metadata import DEFAULT_GENRE_PROPORTIONS, PUBLIC_FIELDS, RECORD_FIELDS

POOL_FILE = "pool.parquet"
METADATA_FILE = "metadata.parquet"
CLUSTERS_FILE = "clusters.json"
MANIFEST_FILE = "manifest.json"


class Slot(NamedTuple):
    """Public stratum of the pool: everything a generator may know about a slot."""

    genre: str
    cluster_id: str
    cluster_label: str
    length_bin: str
    register: str


SLOT_FIELDS: tuple[str, ...] = Slot._fields


def slot_of(row: Mapping) -> Slot:
    return Slot(*(str(row[f]) for f in SLOT_FIELDS))


@dataclass(frozen=True)
class ReferenceText:
    text_id: str
    text: str
    source_id: str
    slot: Slot  # the stratum it was drawn for
    actual: Slot  # its own stratum (differs from ``slot`` only if the draw had to relax)
    match: str  # "exact" | "any_register" | "genre_length"


# --------------------------------------------------------------------------- hashing helpers
def seed_bytes(seed: int | str | bytes) -> bytes:
    if isinstance(seed, bytes):
        return seed
    if isinstance(seed, int):
        return str(seed).encode("ascii")
    return str(seed).encode("utf-8")


def rank_key(seed: bytes, tag: str, text_id: str) -> bytes:
    return hashlib.sha256(seed + b"\x1f" + tag.encode("utf-8") + b"\x1f" + text_id.encode("utf-8")).digest()


def allocate(
    n: int,
    proportions: Mapping[str, float],
    available: Iterable[str],
    missing: Literal["error", "renormalize"] = "error",
) -> dict[str, int]:
    """Split ``n`` across genres by the largest-remainder method."""
    if n < 0:
        raise ValueError("n must be non-negative")
    avail = set(available)
    props = {g: float(p) for g, p in proportions.items() if p and p > 0}
    absent = sorted(set(props) - avail)
    if absent and missing == "error":
        raise ValueError(f"genres requested but absent from the pool: {absent}")
    props = {g: p for g, p in props.items() if g in avail}
    total = sum(props.values())
    if total <= 0:
        raise ValueError("no positive proportions for genres present in the pool")
    quotas = {g: n * p / total for g, p in props.items()}
    out = {g: math.floor(q) for g, q in quotas.items()}
    rest = n - sum(out.values())
    for g in sorted(props, key=lambda g: (-(quotas[g] - out[g]), g))[:rest]:
        out[g] += 1
    return out


# --------------------------------------------------------------------------- sampling
def sample_slots(
    meta: Sequence[Mapping],
    n: int,
    genre_proportions: Mapping[str, float] | None = None,
    seed: int | str | bytes = 0,
    missing: Literal["error", "renormalize"] = "error",
) -> list[Slot]:
    """Draw ``n`` slots: genre counts follow ``genre_proportions``; within a genre, strata appear in
    proportion to their size in the pool (a hash-ranked draw of pool rows, of which only the stratum is
    kept). Returned sorted, as a multiset of strata -- no text, no text id."""
    by_genre: dict[str, list[Mapping]] = defaultdict(list)
    for r in meta:
        by_genre[str(r["genre"])].append(r)
    alloc = allocate(n, genre_proportions or DEFAULT_GENRE_PROPORTIONS, by_genre, missing)
    sb = seed_bytes(seed)
    slots: list[Slot] = []
    for g in sorted(alloc):
        rows = by_genre[g]
        if alloc[g] > len(rows):
            raise ValueError(f"genre {g!r}: {alloc[g]} slots requested, pool has {len(rows)} texts")
        ranked = sorted(rows, key=lambda r: rank_key(sb, f"slots/{g}", str(r["text_id"])))
        slots.extend(slot_of(r) for r in ranked[: alloc[g]])
    return sorted(slots)


_LEVELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("exact", SLOT_FIELDS),
    ("any_register", ("genre", "cluster_id", "length_bin")),
    ("genre_length", ("genre", "length_bin")),  # the scoring group of D-20
)


def sample_reference(
    pool: Sequence[Mapping],
    slot_counts: Mapping[Slot, int] | Iterable[Slot],
    seed: int | str | bytes = 0,
    exclude_ids: Iterable[str] = (),
    relax: bool = True,
) -> list[ReferenceText]:
    """Draw human texts with the same count per stratum as ``slot_counts`` (D-54).

    If a stratum has too few unused texts, the draw relaxes to (genre, cluster, length) and then to
    (genre, length) -- the D-20 scoring group -- unless ``relax=False``; ``match`` records the level.
    Raises ``ValueError`` if even the widest level cannot supply the count.
    """
    counts = Counter(slot_counts) if not isinstance(slot_counts, Mapping) else Counter(dict(slot_counts))
    sb = seed_bytes(seed)
    levels = _LEVELS if relax else _LEVELS[:1]
    index: dict[str, dict[tuple, list[Mapping]]] = {name: defaultdict(list) for name, _ in levels}
    for r in pool:
        for name, fields in levels:
            index[name][tuple(str(r[f]) for f in fields)].append(r)
    for name in index:
        for rows in index[name].values():
            rows.sort(key=lambda r: rank_key(sb, "reference", str(r["text_id"])))
    used = set(exclude_ids)
    need = {s: int(c) for s, c in counts.items() if c > 0}
    picked: dict[Slot, list[ReferenceText]] = defaultdict(list)
    # level by level: every slot is served exactly before any slot is allowed to relax
    for name, fields in levels:
        for slot in sorted(need):
            if need[slot] <= 0:
                continue
            key = tuple(getattr(slot, f) for f in fields)
            for r in index[name].get(key, ()):
                if need[slot] <= 0:
                    break
                tid = str(r["text_id"])
                if tid in used:
                    continue
                used.add(tid)
                picked[slot].append(
                    ReferenceText(
                        text_id=tid,
                        text=str(r["text"]),
                        source_id=str(r.get("source_id", "")),
                        slot=slot,
                        actual=slot_of(r),
                        match=name,
                    )
                )
                need[slot] -= 1
    short = {s: n for s, n in need.items() if n > 0}
    if short:
        s, n = sorted(short.items())[0]
        raise ValueError(
            f"pool cannot supply {counts[s]} texts for {s} ({n} missing; {len(short)} strata short)"
        )
    return [ref for slot in sorted(picked) for ref in picked[slot]]


# --------------------------------------------------------------------------- storage
def content_sha256(rows: Iterable[Mapping]) -> str:
    """Hash over sorted (text_id + public metadata) lines; independent of parquet encoding."""
    lines = sorted(json.dumps({f: r[f] for f in PUBLIC_FIELDS}, sort_keys=True, default=str) for r in rows)
    h = hashlib.sha256()
    for ln in lines:
        h.update(ln.encode("utf-8") + b"\n")
    return h.hexdigest()


def write_pool(
    out_dir: Path | str,
    rows: Sequence[Mapping],
    clusters: Sequence[Mapping],
    manifest: Mapping | None = None,
) -> dict[str, Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (r["genre"], r["cluster_id"], r["text_id"]))
    full = [{f: r[f] for f in RECORD_FIELDS} for r in ordered]
    public = [{f: r[f] for f in PUBLIC_FIELDS} for r in ordered]
    pq.write_table(pa.Table.from_pylist(full), out / POOL_FILE, compression="zstd")
    pq.write_table(pa.Table.from_pylist(public), out / METADATA_FILE, compression="zstd")
    (out / CLUSTERS_FILE).write_text(json.dumps(list(clusters), indent=2, ensure_ascii=False))
    man = dict(manifest or {})
    man["n_texts"] = len(full)
    man["content_sha256"] = content_sha256(full)
    (out / MANIFEST_FILE).write_text(json.dumps(man, indent=2, ensure_ascii=False, default=str))
    return {
        "pool": out / POOL_FILE,
        "metadata": out / METADATA_FILE,
        "clusters": out / CLUSTERS_FILE,
        "manifest": out / MANIFEST_FILE,
    }


class PoolStore:
    """In-memory view of a built pool."""

    def __init__(
        self, rows: Sequence[Mapping], clusters: Sequence[Mapping] = (), manifest: Mapping | None = None
    ):
        self.rows = [dict(r) for r in rows]
        self.clusters = [dict(c) for c in clusters]
        self.manifest = dict(manifest or {})

    @classmethod
    def load(cls, path: Path | str) -> PoolStore:
        import pyarrow.parquet as pq

        p = Path(path)
        rows = pq.read_table(p / POOL_FILE).to_pylist()
        clusters = json.loads((p / CLUSTERS_FILE).read_text()) if (p / CLUSTERS_FILE).exists() else []
        manifest = json.loads((p / MANIFEST_FILE).read_text()) if (p / MANIFEST_FILE).exists() else {}
        return cls(rows, clusters, manifest)

    def __len__(self) -> int:
        return len(self.rows)

    def metadata(self) -> list[dict]:
        """Public metadata rows (no text)."""
        return [{f: r[f] for f in PUBLIC_FIELDS} for r in self.rows]

    def genres(self) -> list[str]:
        return sorted({r["genre"] for r in self.rows})

    def strata_counts(self) -> Counter[Slot]:
        return Counter(slot_of(r) for r in self.rows)

    def sample_slots(
        self,
        n: int,
        genre_proportions: Mapping[str, float] | None = None,
        seed: int | str | bytes = 0,
        missing: Literal["error", "renormalize"] = "error",
    ) -> list[Slot]:
        return sample_slots(self.metadata(), n, genre_proportions, seed, missing)

    def sample_reference(
        self,
        slot_counts: Mapping[Slot, int] | Iterable[Slot],
        seed: int | str | bytes = 0,
        exclude_ids: Iterable[str] = (),
        relax: bool = True,
    ) -> list[ReferenceText]:
        return sample_reference(self.rows, slot_counts, seed, exclude_ids, relax)
