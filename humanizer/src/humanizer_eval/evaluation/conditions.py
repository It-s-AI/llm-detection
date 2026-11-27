"""Identical scoring conditions for the king and every challenger (owner's requirement 2026-11-21).

A crown decision compares two runs. They must be judged and scored under exactly the same conditions,
even when an asymmetry would be unlikely or would favour nobody on average:

* **Scoring fingerprint.** A hash of everything that turns outputs into levels and scores: the source of
  the gates, the evaluation step, the metric, the bootstrap, the crown test and the detector clients,
  the gate configuration (judges, their pinned providers, thresholds), the detector set with each
  client's request configuration, and the crown parameters. The king's run carries the fingerprint it
  was scored under; if it differs from the current one, the king is rescored from his saved outputs
  before any challenger is compared with him (``intake/worker.py``), and the pipeline refuses to
  compare runs with different fingerprints.
* **Detector models.** A vendor can change its model silently, between periods (then cached scores of
  H come from the old model) or in the middle of a period. H, the king and every challenger must be
  scored by the same model. Every scored text is checked by the model version the vendor reports (GPTZero,
  Pangram). On top of that, for every detector, the two texts of H whose scores are closest to 0.5 (where a
  model change shows) are canaries: when the period is loaded they are scored again without the cache and
  must match H's stored scores, and before each evaluation they are scored again and must still match. Any
  difference pauses the evaluations (``ConditionsChanged``) until the owner rescores H and the king
  (``scripts/rescore_period.py``); a problem found when the period is loaded does the same (the server
  stays up, nothing is evaluated). ``canary_tol`` absorbs the vendor's own noise only.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PKG = Path(__file__).resolve().parents[1]
#: the modules that turn a run's outputs into levels, scores and a crown decision
SCORING_MODULES = (
    "gates/judges.py",
    "gates/meaning.py",
    "gates/normalize.py",
    "evaluation/pipeline.py",
    "evaluation/conditions.py",
    "scoring/metric.py",
    "scoring/bootstrap.py",
    "scoring/crown.py",
    "detectors/base.py",
    "detectors/vendors.py",
    "llm/openrouter.py",
)
CANARIES_PER_DETECTOR = 2
#: the paid vendors return exactly the same score for the same text (difference 0, checked live 2026-11-21);
#: a GPU-served model such as It's AI jitters around 1e-5. A model change moves mid-range texts far more.
CANARY_TOL = 1e-3
#: GPTZero answers the same text with one of two close values (two backends?): 0.3892 / 0.3964 on a
#: mid-range H text, measured on the production key 2026-11-25; a model change moves such texts much more
CANARY_TOL_BY_DETECTOR = {"gptzero": 0.02}


def canary_tol(name: str) -> float:
    return CANARY_TOL_BY_DETECTOR.get(name, CANARY_TOL)


def scoring_fingerprint(
    gates_cfg: Mapping[str, Any], detectors: Mapping[str, Any], crown_kw: Mapping[str, Any]
) -> str:
    h = hashlib.sha256()
    for m in SCORING_MODULES:
        h.update(m.encode())
        h.update((PKG / m).read_bytes())
    h.update(json.dumps(gates_cfg, sort_keys=True, default=str).encode())
    dets = {n: getattr(d, "config", None) or type(d).__name__ for n, d in sorted(detectors.items())}
    h.update(json.dumps(dets, sort_keys=True).encode())
    h.update(json.dumps(dict(crown_kw), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def versions(results: Sequence[Any]) -> list[str]:
    """Model versions reported in detector results (sorted, distinct; empty if the vendor reports none)."""
    return sorted({r.version for r in results if getattr(r, "ok", True) and r.version})


class CanaryUnavailable(RuntimeError):
    """A canary got no answer (the detector is down): an outage, never a model change."""


def _score_texts(det: Any, texts: Sequence[str], tries: int = 3) -> list[float]:
    res = det.score(list(texts))
    for _ in range(tries - 1):  # like the H texts: a passing outage is asked again
        bad = [i for i, r in enumerate(res) if not r.ok]
        if not bad:
            break
        for i, r in zip(bad, det.score([texts[i] for i in bad]), strict=True):
            res[i] = r
    bad = [r for r in res if not r.ok]
    if bad:
        raise CanaryUnavailable(f"{getattr(det, 'name', '?')}: canary not scored ({bad[0].error})")
    return [float(r.score) for r in res]


def pick_canaries(texts: Sequence[str], scores: Sequence[float], k: int = CANARIES_PER_DETECTOR) -> list[int]:
    """Indices of the ``k`` texts whose scores are closest to 0.5 (saturated 0 or 1 hide a model change)."""
    order = sorted(range(len(texts)), key=lambda i: (scores[i] in (0.0, 1.0), abs(scores[i] - 0.5)))
    return order[:k]


def take_baseline(
    human_versions: Mapping[str, Sequence[str]],
    watch: Mapping[str, Any] | None,
    human_texts: Sequence[str] = (),
    human_scores: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """The detector models H was scored by: their versions, and canaries (texts of H scored again now,
    without the cache, which must match H's stored scores). ``watch`` holds the same detectors uncached."""
    out: dict[str, Any] = {}
    for name, vers in sorted(human_versions.items()):
        vers = sorted(vers)
        if len(vers) > 1:
            raise RuntimeError(f"{name}: the human reference set was scored across a model change {vers}")
        entry: dict[str, Any] = {"versions": vers}
        hs = (human_scores or {}).get(name)
        if watch is not None and name in watch and human_texts and hs is not None:
            idx = pick_canaries(human_texts, hs)
            texts = [human_texts[i] for i in idx]
            fresh = _score_texts(watch[name], texts)
            stale = max((abs(f - float(hs[i])) for f, i in zip(fresh, idx, strict=True)), default=0.0)
            if stale > canary_tol(name):
                raise RuntimeError(
                    f"{name}: H's stored scores differ from the vendor's current model by {stale:.4g} "
                    "(cached from an older model?): rescore with scripts/rescore_period.py"
                )
            entry["canary"] = {"texts": texts, "scores": fresh}
        out[name] = entry
    return out


def check(baseline: Mapping[str, Any] | None, watch: Mapping[str, Any] | None,
          seen: Mapping[str, Sequence[str]] | None = None) -> list[str]:  # fmt: skip
    """Differences from the period's detector models: versions ``seen`` in this run's results, and the
    canaries re-scored now (uncached). An empty list means the conditions are the period's."""
    if not baseline:
        return []
    if baseline.get("_problem"):  # found when the period was loaded: nothing is compared until rescored
        return [str(baseline["_problem"])]
    problems: list[str] = []
    for name, b in baseline.items():
        got = sorted(set((seen or {}).get(name, ())))
        if b.get("versions") and got and got != b["versions"]:
            problems.append(f"{name}: model version {got} differs from the period's {b['versions']}")
        if b.get("canary") and watch is not None and name in watch:
            now = _score_texts(watch[name], b["canary"]["texts"])
            diff = max((abs(x - y) for x, y in zip(now, b["canary"]["scores"], strict=True)), default=0.0)
            if diff > canary_tol(name):
                problems.append(
                    f"{name}: canary scores moved by {diff:.4g} since the period's human texts were scored"
                )
    return problems
