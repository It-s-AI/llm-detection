"""Adapter: the phase-2 PoolStore -> the PoolLike protocol used by the period builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from humanizer_eval.pool.store import PoolStore, Slot


class PoolAdapter:
    def __init__(self, store: PoolStore):
        self.store = store

    @classmethod
    def load(cls, path: Path | str) -> PoolAdapter:
        return cls(PoolStore.load(path))

    def sample_slots(self, n: int, genre_proportions: dict[str, float], seed: int) -> list[dict[str, Any]]:
        total = sum(genre_proportions.values())
        props = {g: v / total for g, v in genre_proportions.items()}
        return [s._asdict() for s in self.store.sample_slots(n, props, seed, missing="renormalize")]

    def sample_reference(self, slot_counts: dict[tuple, int], seed: int) -> list[dict[str, Any]]:
        counts = {Slot(*key): k for key, k in slot_counts.items()}
        refs = self.store.sample_reference(counts, seed)
        return [
            {
                "id": r.text_id,
                "text": r.text,
                "source_id": r.source_id,
                "match": r.match,
                **r.actual._asdict(),
            }
            for r in refs
        ]
