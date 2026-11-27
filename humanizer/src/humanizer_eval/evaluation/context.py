"""What the orchestrator needs to know about a period, prepared once per period at the owner's cost.

* inputs with their facts (G1 v2 facts; a pilot period with v1 string facts gets v2 facts here, through
  the judge cache),
* the human reference set H with its crown-detector scores (D-54),
* the period seed (D-57): the harness seed and every crown bootstrap seed derive from it,
* the king's run on this period's inputs (the run every challenger is compared with, D-11).
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from humanizer_eval.detectors.base import Detector
from humanizer_eval.period.crypto import decrypt_at_rest, encrypt_at_rest
from humanizer_eval.period.seed import derive_seed, subseed
from humanizer_eval.scoring.metric import HumanReference, SubmissionRun

log = logging.getLogger(__name__)


def stratum(slot: Mapping[str, Any]) -> str:
    return f"{slot['genre']}|{slot['length_bin']}"


def human_id(scores: Any) -> str:
    """A short id of one set of H scores ({detector: [score per human text]}): records name the H they were
    scored against, so a reveal after a mid-period rescore can check each record against its own H."""
    import hashlib

    norm = {
        d: [None if x is None else round(float(x), 8) for x in v] for d, v in sorted(dict(scores).items())
    }
    return hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class PeriodContext:
    period_id: str
    seed: bytes
    items: list[dict[str, Any]]  # id, text, slot{genre, length_bin, ...}, facts
    human_ids: list[str]
    human_strata: list[str]
    human_scores: dict[str, list[float]]  # crown detector -> score per human text
    dir: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def detectors(self) -> tuple[str, ...]:
        return tuple(sorted(self.human_scores))

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def input_ids(self) -> list[str]:
        return [it["id"] for it in self.items]

    @property
    def strata(self) -> list[str]:
        return [stratum(it["slot"]) for it in self.items]

    def input_words(self) -> list[int]:
        return [len(it["text"].split()) for it in self.items]

    def harness_items(self) -> list[dict[str, Any]]:
        """What the miner's code receives (D-60): no facts, no slot metadata beyond the genre."""
        return [
            {"id": it["id"], "text": it["text"], "lang": "en", "genre": it["slot"]["genre"]}
            for it in self.items
        ]

    def human_reference(self) -> HumanReference:
        return HumanReference(self.human_scores, self.human_strata)

    @property
    def h_id(self) -> str:
        """Which scoring of H this context carries (changes when a detector is rescored mid-period)."""
        return human_id(self.human_scores)

    @property
    def harness_seed(self) -> int:
        """One seed per period for every submission: the same bundle gives the same outputs (no re-rolls)."""
        return subseed(self.seed, "harness")

    def crown_seed(self, submission_id: str) -> int:
        return subseed(self.seed, f"crown:{submission_id}")


def _facts_are_v2(facts: Any) -> bool:
    """Anchored facts as the period builder saves them ({"text", "sents"}); v1 periods have plain strings."""
    return isinstance(facts, list) and all(isinstance(f, dict) and "sents" in f for f in facts)


def load_period(
    pdir: Path | str,
    key: bytes,
    *,
    human_texts: Mapping[str, str],
    detectors: Mapping[str, Detector],
    facts_llm: Any = None,
    facts_extractor: str = "anthropic/claude-haiku-5.5",
    facts_validators: tuple[str, str] = ("anthropic/claude-sonnet-5.5", "openai/gpt-6.1-sol"),
    workers: int = 8,
    allow_missing_human: bool = False,
    human_retries: int = 3,
    retry_wait_s: float = 20.0,
    human_cache: Path | str | None = None,
    watch: Mapping[str, Detector] | None = None,
) -> PeriodContext:
    """Load an encrypted period and score H with the crown detectors (owner's cost, cached).

    A vendor's transient error on a few human texts (ZeroGPT answers "code 500, try again" now and then)
    is retried ``human_retries`` times with a growing pause; only texts that still fail stop the load.
    Scores already obtained are cached, so a retry pays only for the failed texts. With ``human_cache``
    the scores of H are kept there (encrypted) once complete: a restart, the reveal and the next load of the
    same period read them instead of asking the detectors again. The detector models H was scored by
    (versions; canary scores through ``watch``, the same detectors uncached, for vendors without a
    version) are kept with them: every later run of the period must be scored by the same models
    (evaluation/conditions.py)."""
    from humanizer_eval.evaluation import conditions
    from humanizer_eval.gates.judges import Judge
    from humanizer_eval.gates.meaning import prepare_facts

    pdir = Path(pdir)
    manifest = json.loads((pdir / "manifest.json").read_text())
    data = json.loads(decrypt_at_rest(key, (pdir / "dataset.enc").read_bytes()))
    secret = bytes.fromhex((pdir / "secret.hex").read_text().strip())
    seed = derive_seed(secret, manifest["block_hash"])
    items = [dict(it) for it in data["items"]]
    if facts_llm is not None and not all(_facts_are_v2(it.get("facts")) for it in items):
        # v1 pilot periods: anchored v2 facts from the same extractor/validators as score_outputs.py
        ext, vals = Judge(facts_extractor), (Judge(facts_validators[0]), Judge(facts_validators[1]))

        def v2(i: int) -> list[dict[str, Any]] | None:
            facts, _ = prepare_facts(facts_llm, ext, vals, items[i]["text"])
            return [f.to_dict() for f in facts] if facts else None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, f in enumerate(ex.map(v2, range(len(items)))):
                if f is not None:
                    items[i]["facts"] = f
    human = [h for h in data["human_ref"] if h["id"] in human_texts]
    missing = len(data["human_ref"]) - len(human)
    if missing and not allow_missing_human:  # pilots built from an older pool may miss a few
        raise ValueError(f"{missing} human reference texts are missing from the pool")
    texts = [human_texts[h["id"]] for h in human]
    ids = [h["id"] for h in human]
    cached = load_human_scores(human_cache, key) if human_cache else None
    scores: dict[str, list[float]] = {}
    same = bool(cached) and cached["ids"] == ids
    base_cached: dict[str, Any] = dict((cached or {}).get("baseline") or {}) if same else {}
    hvers: dict[str, list[str]] = {}
    if same:
        scores = {d: v for d, v in cached["scores"].items() if d in detectors}
    for name, det in detectors.items():
        if name in scores:
            continue
        res = list(det.score(texts))
        for attempt in range(human_retries):
            bad_i = [i for i, r in enumerate(res) if not r.ok]
            if not bad_i:
                break
            time.sleep(retry_wait_s * (attempt + 1))
            for i, r in zip(bad_i, det.score([texts[i] for i in bad_i]), strict=True):
                res[i] = r
        bad = [r for r in res if not r.ok]
        if bad:
            raise RuntimeError(f"{name}: {len(bad)} human texts could not be scored ({bad[0].error})")
        scores[name] = [r.score for r in res]
        hvers[name] = conditions.versions(res)
    for name, det in detectors.items():  # scores from the file of an older load without a baseline
        if name not in hvers and name not in base_cached:
            hvers[name] = conditions.versions(det.score(texts))  # the detector cache answers: free
    try:
        new = conditions.take_baseline(hvers, watch, texts, {d: scores[d] for d in hvers if d in scores})
    except conditions.CanaryUnavailable:  # an outage: the load fails and is tried again (never a hold)
        raise
    except RuntimeError as e:  # H's detector models are not the vendors' current ones: hold every evaluation
        log.error("period %s: %s", manifest["period_id"], e)
        new = {"_problem": str(e)}
    baseline = {k: v for k, v in {**base_cached, **new}.items() if k in detectors or k == "_problem"}
    changed = not same or set(scores) - set(cached["scores"]) or baseline != base_cached
    if (
        human_cache and changed and "_problem" not in baseline
    ):  # a problem is never stored: rescore looks again
        save_human_scores(human_cache, key, ids, {**(cached["scores"] if same else {}), **scores}, baseline)
    return PeriodContext(
        period_id=manifest["period_id"],
        seed=seed,
        items=items,
        human_ids=[h["id"] for h in human],
        human_strata=[stratum(h) for h in human],
        human_scores=scores,
        dir=pdir,
        meta={
            "merkle_root": manifest.get("merkle_root"),
            "n_human": len(human),
            "missing_human": missing,
            "detector_baseline": baseline,
        },
    )


def save_human_scores(
    path: Path | str, key: bytes, ids: list[str], scores: Mapping[str, list[float]],
    baseline: Mapping[str, Any] | None = None,
) -> None:  # fmt: skip
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    body = {"ids": ids, "scores": dict(scores), "baseline": dict(baseline or {})}
    tmp.write_bytes(encrypt_at_rest(key, json.dumps(body).encode()))
    tmp.replace(p)


def load_human_scores(path: Path | str, key: bytes) -> dict[str, Any] | None:
    """{"ids": [...], "scores": {detector: [...]}} of the human reference texts, or None."""
    p = Path(path)
    return json.loads(decrypt_at_rest(key, p.read_bytes())) if p.exists() else None


# --- the king's run on this period ---------------------------------------------------------------
def _enc(x: float) -> float | None:
    return None if not math.isfinite(x) else float(x)


def _dec(x: float | None) -> float:
    return math.inf if x is None else float(x)


@dataclass
class KingRun:
    """The current king's quality levels and detector scores on this period's inputs."""

    submission_id: str
    hotkey: str
    input_ids: list[str]
    quality: list[float]
    scores: dict[str, list[float]]  # +inf where the output was not sent (level 0)
    #: scoring fingerprint the run was scored under (evaluation/conditions.py); None: before 2026-11-21
    fingerprint: str | None = None
    #: the submission record in the period's store this run's scores come from (the reveal checks every
    #: crown decision against exactly that run); None: the king's own record, or the fixed baseline
    record_id: str | None = None
    #: the scoring of H it was made under (PeriodContext.h_id): a mid-period detector rescore makes it stale
    h_id: str | None = None

    def to_run(self, strata: Sequence[str]) -> SubmissionRun:
        return SubmissionRun(self.scores, self.quality, list(strata), self.input_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "submission_id": self.submission_id,
            "hotkey": self.hotkey,
            "input_ids": self.input_ids,
            "quality": self.quality,
            "scores": {d: [_enc(x) for x in v] for d, v in self.scores.items()},
            "fingerprint": self.fingerprint,
            "record_id": self.record_id,
            "h_id": self.h_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KingRun:
        return cls(
            d["submission_id"],
            d["hotkey"],
            list(d["input_ids"]),
            [float(q) for q in d["quality"]],
            {k: [_dec(x) for x in v] for k, v in d["scores"].items()},
            d.get("fingerprint"),
            d.get("record_id"),
            d.get("h_id"),
        )

    def save(self, path: Path | str, key: bytes) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")  # never a torn file: other threads and processes read it
        tmp.write_bytes(encrypt_at_rest(key, json.dumps(self.to_dict()).encode()))
        tmp.replace(p)

    @classmethod
    def load(cls, path: Path | str, key: bytes) -> KingRun | None:
        p = Path(path)
        if not p.exists():
            return None
        return cls.from_dict(json.loads(decrypt_at_rest(key, p.read_bytes())))
