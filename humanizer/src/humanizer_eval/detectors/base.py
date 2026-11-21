"""Common types for AI-text detectors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(text.split())


@dataclass
class DetectorScore:
    """One detector verdict for one text.

    `score` is the primary continuous AI-likelihood in [0, 1] (higher = more AI).
    `candidates` holds alternative continuous scores extracted from the same
    response, so the primary definition can be changed after the admission test
    without re-querying the vendor.
    """

    detector: str
    text_sha256: str
    score: float
    candidates: dict[str, float] = field(default_factory=dict)
    version: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class DetectorError(RuntimeError):
    """Non-retryable detector failure (bad request, auth, malformed response)."""


@runtime_checkable
class Detector(Protocol):
    name: str

    def score(self, texts: list[str]) -> list[DetectorScore]: ...
