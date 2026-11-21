"""Canaries: a fixed set of texts scored each period to detect silent vendor model changes (D-32)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from humanizer_eval.detectors.base import Detector, DetectorScore


@dataclass
class CanaryReport:
    detector: str
    changed: bool
    version_before: str | None
    version_now: str | None
    max_abs_diff: float
    n: int
    errors: int

    def summary(self) -> str:
        return (
            f"{self.detector}: changed={self.changed} version {self.version_before}->{self.version_now} "
            f"max|Δscore|={self.max_abs_diff:.4f} n={self.n} errors={self.errors}"
        )


def run_canaries(
    detector: Detector, texts: list[str], baseline_path: Path | str, tol: float = 1e-3, update: bool = False
) -> CanaryReport:
    """Score canaries uncached and compare with the stored baseline.

    If no baseline exists, one is written and the report says changed=False.
    `detector` should be constructed WITHOUT a cache so the vendor is really queried.
    """
    path = Path(baseline_path)
    scores: list[DetectorScore] = detector.score(texts)
    now = {s.text_sha256: s.score for s in scores if s.ok}
    version_now = next((s.version for s in scores if s.ok and s.version), None)
    errors = sum(1 for s in scores if not s.ok)
    if not path.exists() or update:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": version_now, "scores": now}, indent=1))
        return CanaryReport(detector.name, False, version_now, version_now, 0.0, len(now), errors)
    base = json.loads(path.read_text())
    diffs = [abs(now[k] - v) for k, v in base["scores"].items() if k in now and not math.isnan(now[k])]
    max_diff = max(diffs) if diffs else float("nan")
    changed = (base.get("version") != version_now) or (bool(diffs) and max_diff > tol)
    return CanaryReport(
        detector.name, changed, base.get("version"), version_now, max_diff, len(diffs), errors
    )
