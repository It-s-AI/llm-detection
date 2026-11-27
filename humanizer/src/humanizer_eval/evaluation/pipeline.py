"""One submission, end to end (D-11, D-20..D-24, D-40..D-44, D-58, D-60..D-63, D-80).

    quote -> hold                      (billing: the balance must cover the quote; the result waits for it)
    run the bundle                     (fresh Lium pod; outputs saved encrypted right away)
    G0 normalize + length cap          (D-40)
    G1-G3 -> level 1 / 0.75 / 0.5 / 0 (judge tokens metered per model; cache hits are free)
    crown detectors                    (outputs at level 0 count as caught anyway and are not sent)
    S, crown test against the king     (paired bootstrap, seed from the period seed)
    statement -> settle                (metered words, tokens, pod hours x the public card; <= hold)

Failure rules (docs/billing-policy.md):
* our side fails (no pod, setup, lost pod): the hold is released, nothing is charged, status ``failed``;
* the miner's bundle fails (bad bundle, crash, no outputs): the run is scored as it is (missing outputs
  are level 0) and the pod time used is charged;
* judges or detectors unavailable (ANY output still without a verdict or a score after
  ``retry_rounds`` retries, or a spend cap is hit): the evaluation **pauses** (D-90) with the hold kept
  and the outputs saved; calling again resumes from the saved outputs (cached verdicts and scores are
  not paid twice). An output is never given level 0 or a "caught" score because OUR judge or detector
  failed: the king and every challenger are judged and scored completely, or not compared at all;
* the scoring conditions changed (``evaluation/conditions.py``: a different scoring fingerprint than
  the king's run, or a detector model that is not the one H was scored by): paused as well, never
  compared.

What the miner sees before the period is disclosed (D-58): the stage, the decision, rounded aggregate
scores and the statement. Never text produced by its code at run time (an error message could carry
the hidden inputs out); the full record is encrypted and published with the period.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from humanizer_eval.billing.prices import statement as make_statement
from humanizer_eval.detectors.base import Detector, word_count
from humanizer_eval.evaluation.context import KingRun, PeriodContext
from humanizer_eval.evaluation.metering import MeteringLLM
from humanizer_eval.evaluation.runners import Runner, RunOutput, Submission
from humanizer_eval.gates.judges import GateResult
from humanizer_eval.gates.normalize import apply_length_limit, normalize
from humanizer_eval.period.crypto import decrypt_at_rest, encrypt_at_rest
from humanizer_eval.scoring.crown import KingState, crown_decision
from humanizer_eval.scoring.metric import SubmissionRun, submission_score

__all__ = [
    "EvalConfig",
    "EvalResult",
    "EvaluationPaused",
    "InsufficientBalance",
    "Submission",
    "SubmissionStore",
    "apply_crown",
    "evaluate_submission",
]


@dataclass
class EvalConfig:
    workers: int = 8
    #: judge / detector calls that failed are retried this many rounds (waiting retry_wait_s x round)
    #: before the evaluation pauses; no output is ever left without a verdict or a score
    retry_rounds: int = 3
    retry_wait_s: float = 15.0
    #: scoring fingerprint (evaluation/conditions.py), stamped on every record and king run; runs with
    #: different fingerprints are never compared
    fingerprint: str | None = None
    skip_severe: bool = True  # level-0 outputs count as caught by every detector: not sent, not paid
    public_digits: int = 4
    update_king: bool = True  # False: decide, but leave the king and the king state untouched
    #: True: a won crown waits (stage "archiving", crowned False, crown_pending True) until the caller
    #: has archived the bundle and calls ``apply_crown`` (the queue worker does, D-70)
    defer_crown: bool = False
    crown_kw: dict[str, Any] = field(default_factory=dict)  # alpha, delta_min, n_boot, ...


class EvaluationPaused(RuntimeError):
    def __init__(self, stage: str, reason: str):
        super().__init__(f"paused at {stage}: {reason}")
        self.stage, self.reason = stage, reason


class ConditionsChanged(EvaluationPaused):
    """The run would not be scored under the conditions of the period's H and the king's run."""


class InsufficientBalance(RuntimeError):
    def __init__(self, quote: dict[str, Any], amount_due_cents: int):
        super().__init__(f"balance does not cover quote {quote['id']}: pay {amount_due_cents} cents")
        self.quote, self.amount_due_cents = quote, amount_due_cents


@dataclass
class EvalResult:
    submission_id: str
    status: str  # "done" | "failed"
    public: dict[str, Any]
    private: dict[str, Any]
    statement: dict[str, Any] | None = None
    settlement: dict[str, Any] | None = None
    crowned: bool = False


# --- records ---------------------------------------------------------------------------------------
def _storable(meta: Any) -> Any:
    """The run's meta as it can be stored: it may carry what the miner's code wrote, and a nesting too
    deep to serialise must never fail the save (that looked like our failure)."""
    stack, seen = [(meta, 1)], 0
    while stack:  # iterative: the check itself never recurses
        node, depth = stack.pop()
        seen += 1
        if depth > 32 or seen > 100_000:
            return {"dropped": "meta too deep or too large to store"}
        if isinstance(node, dict):
            stack.extend((v, depth + 1) for v in node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend((v, depth + 1) for v in node)
    return meta


class SubmissionStore:
    """``<root>/<submission_id>/``: ``status.json`` and ``statement.json`` (miner-visible), ``outputs.enc``
    and ``record.enc`` (encrypted until the period is disclosed), ``run/`` (pod logs, private)."""

    def __init__(self, root: Path | str, key: bytes):
        self.root = Path(root)
        self.key = key

    def dir(self, sid: str) -> Path:
        d = self.root / sid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def status(self, sid: str) -> dict[str, Any] | None:
        p = self.root / sid / "status.json"
        return json.loads(p.read_text()) if p.exists() else None

    def set_stage(self, sid: str, hotkey: str, stage: str, **public: Any) -> dict[str, Any]:
        st = {"submission_id": sid, "hotkey": hotkey, "stage": stage, "updated_at": time.time(), **public}
        p = self.dir(sid) / "status.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1, default=str))
        tmp.replace(p)
        return st

    def _put(self, sid: str, name: str, obj: Any) -> None:
        p = self.dir(sid) / name
        tmp = p.with_name(p.name + ".tmp")  # never a torn file: a kill mid-write would block the row for good
        tmp.write_bytes(encrypt_at_rest(self.key, json.dumps(obj, default=str).encode()))
        tmp.replace(p)

    def _get(self, sid: str, name: str) -> Any:
        p = self.root / sid / name
        return json.loads(decrypt_at_rest(self.key, p.read_bytes())) if p.exists() else None

    def save_run(self, sid: str, run: RunOutput) -> None:
        self._put(
            sid,
            "outputs.enc",
            {
                "status": run.status,
                "outputs": run.outputs,
                "pod_seconds": run.pod_seconds,
                "reason": run.reason,
                "meta": _storable(run.meta),
                "cost_usd": run.cost_usd,
            },
        )

    def load_run(self, sid: str) -> RunOutput | None:
        d = self._get(sid, "outputs.enc")
        if d is None:
            return None
        return RunOutput(d["status"], d["outputs"], d["pod_seconds"], d["reason"], d["meta"], d["cost_usd"])

    def load_metered(self, sid: str) -> dict[str, Any]:
        """Judge tokens and detector words paid during earlier attempts that paused (charged at the end)."""
        return self._get(sid, "metered.enc") or {"judge_tokens": {}, "detector_words": {}}

    def drop_metered_detectors(self, sid: str, detectors: Any) -> None:
        """Forget the paid words of these detectors (their cache was dropped by a rescore: paid again)."""
        m = self.load_metered(sid)
        for d in detectors:
            m["detector_words"].pop(d, None)
            m.get("dw_by", {}).pop(d, None)
        self._put(sid, "metered.enc", m)

    def add_metered(
        self,
        sid: str,
        judge_tokens: dict[str, Any],
        detector_words: dict[str, list[int]],
        epochs: dict[str, str] | None = None,
        judge_by_output: dict[int, dict[str, tuple[int, int]]] | None = None,
        attempt: str = "legacy",
    ) -> None:
        """Usage is kept per cache epoch: the judges' prompts and config, each detector's config (what the
        cache keys are made of). Within one epoch purchases add up (attempts that paused: the resume buys
        only what the cache lacks). Across epochs (a deploy during a pause emptied that cache, so the same
        calls were bought again) each item is charged once, the largest purchase: the second one was caused
        by our deploy. A code-only deploy keeps the epochs, so nothing paid is lost to the owner."""
        epochs = epochs or {}
        m = self.load_metered(sid)
        jb = m.setdefault("jt_by", {})  # epoch -> model -> (in, out)
        db = m.setdefault("dw_by", {})  # detector -> epoch -> [words]
        if not jb and not db and (m["judge_tokens"] or m["detector_words"]):  # stored before epochs
            jb["legacy"] = {k: tuple(v) for k, v in m["judge_tokens"].items()}
            for d, w in m["detector_words"].items():
                db[d] = {"legacy": list(w)}
        je = epochs.get("judges", "legacy")
        for k, (a, c) in judge_tokens.items():
            x, y = jb.setdefault(je, {}).get(k, (0, 0))
            jb[je][k] = (x + a, y + c)
        for d, w in detector_words.items():
            de = epochs.get(d, "legacy")
            db.setdefault(d, {})[de] = list(db.get(d, {}).get(de, [])) + list(w)
        jt: dict[str, tuple[int, int]] = {}
        for by_model in jb.values():
            for k, (a, c) in by_model.items():
                x, y = jt.get(k, (0, 0))
                jt[k] = (max(x, a), max(y, c))
        # judges counted per output (exact): within an attempt they add up, across attempts each output's
        # judgement is charged once (the largest), so neither a cache-keeping nor a cache-emptying deploy
        # during a pause can make the miner pay twice or the owner pay for the miner
        jo = m.setdefault("jo", {})
        for i, by_m in (judge_by_output or {}).items():
            slot = jo.setdefault(attempt, {}).setdefault(str(i), {})
            for k, (a, c) in by_m.items():
                x, y = slot.get(k, (0, 0))
                slot[k] = (x + a, y + c)
        if jo:
            best: dict[tuple[str, str], tuple[int, int]] = {}
            for att in jo.values():
                for i, by_m in att.items():
                    for k, (a, c) in by_m.items():
                        x, y = best.get((i, k), (0, 0))
                        best[(i, k)] = (max(x, a), max(y, c))
            jt = {}
            for (_, k), (a, c) in best.items():
                x, y = jt.get(k, (0, 0))
                jt[k] = (x + a, y + c)
        dw = {d: max(eps.values(), key=sum) for d, eps in db.items() if eps}
        m.pop("by_fp", None)
        m["judge_tokens"], m["detector_words"] = jt, dw
        self._put(sid, "metered.enc", m)

    def save_record(self, sid: str, record: dict[str, Any]) -> None:
        self._put(sid, "record.enc", record)

    def load_record(self, sid: str) -> dict[str, Any] | None:
        return self._get(sid, "record.enc")

    def save_statement(self, sid: str, st: dict[str, Any]) -> None:
        p = self.dir(sid) / "statement.json"
        tmp = p.with_name(p.name + ".tmp")  # never a torn file (it is served and revealed)
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(p)


# --- steps -----------------------------------------------------------------------------------------
def _g0(outputs: Mapping[str, str], ctx: PeriodContext) -> tuple[list[str], list[dict[str, Any]]]:
    outs, notes = [], []
    for it in ctx.items:
        raw = outputs.get(it["id"]) or ""
        text, rep = normalize(raw)
        text, cut = apply_length_limit(text, it["text"])
        outs.append(text)
        notes.append({"cut": cut, "normalization": getattr(rep, "__dict__", {}) or str(rep)})
    return outs, notes


_OURS = (
    "budget",
    "cap $",
    "429",
    "HTTP 5",
    "code=5",
    "timeout",
    "Timeout",
    "rate limit",
    "Connect",
    "unavailable",
)


#: answers about the text itself: too long, unsupported, unprocessable (never our account or the load)
_TEXT_REFUSALS = ("400", "404", "413", "415", "422")


def _ours(reason: str) -> bool:
    """A judge or detector error that is our failure or an outage, never the output's doing."""
    return any(k in reason for k in _OURS)


def _refusal(err: str) -> bool:
    """A detector's answer that refuses this very text: an HTTP 4xx other than 408/429."""
    import re

    # an HTTP 4xx, or the same code inside a 200 body (ZeroGPT answers errors as 200 with "code=4xx")
    m = re.search(r"HTTP (4\d\d)", err) or re.search(r"code=(4\d\d)", err)
    if not m or m.group(1) not in _TEXT_REFUSALS:  # 401/402/403: our key or account, 408/429: load
        return False
    low = err.lower()
    return not any(k in low for k in ("credit", "quota", "key", "auth", "balance", "plan", "limit"))


def _gate_all(
    gates: Any,
    ctx: PeriodContext,
    outs: list[str],
    cfg: EvalConfig,
    retry_gates: Any = None,
    progress: Any = None,
    final_gates: Any = None,
    per_output: Any = None,  # i -> gates metered for output i (the first pass)
) -> list[GateResult]:
    """Every output judged; outputs whose judges failed are asked again, ``cfg.retry_rounds`` times."""

    def one(i: int) -> GateResult:
        g = per_output(i) if per_output is not None else gates
        return g.check(ctx.items[i]["text"], outs[i], ctx.items[i].get("facts") or [])

    def again(i: int, last: bool = False) -> GateResult:  # usable cached verdicts reused; the last round
        g = (final_gates if last else None) or retry_gates or gates  # reads no cache at all
        return g.check(ctx.items[i]["text"], outs[i], ctx.items[i].get("facts") or [])

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        res = []
        for r in ex.map(one, range(len(outs))):
            res.append(r)
            if progress is not None and len(res) % PROGRESS_EVERY == 0:
                progress()  # what was paid so far is stored: a kill loses at most one batch
        for k in range(cfg.retry_rounds):
            retry = [i for i, r in enumerate(res) if r.status == "error"]
            if not retry:
                break
            time.sleep(cfg.retry_wait_s * k)
            last = k == cfg.retry_rounds - 1
            for i, r in zip(retry, ex.map(lambda i, last=last: again(i, last), retry), strict=True):
                res[i] = r
    # a judge failure is never turned into a level here: "no answer" cannot tell a provider outage (ours)
    # from an unusable reply, so the run pauses; past the pauses it is parked for the owner to decide
    return res


def _score_detector(
    det: Detector, texts: list[str], cfg: EvalConfig, progress: Any = None
) -> tuple[list[float], list[int], list[str | None], bool, list[str]]:
    """Scores, metered words (sent and not cached, in the order they were paid), errors, budget hit, model
    versions seen. Failed texts are sent again ``cfg.retry_rounds`` times; the caller pauses if any is
    still unscored. ``progress(words)`` is called after each batch so a kill loses at most one batch."""
    words: list[int] = []

    def paid(batch: list[int], got: list[Any]) -> None:
        words.extend(word_count(texts[i]) for i, r in zip(batch, got, strict=True) if r.ok and not r.cached)
        if progress is not None:
            progress(list(words))

    res: list[Any] = [None] * len(texts)
    for start in range(0, len(texts), PROGRESS_EVERY):
        batch = list(range(start, min(len(texts), start + PROGRESS_EVERY)))
        got = det.score([texts[i] for i in batch])
        for i, r in zip(batch, got, strict=True):
            res[i] = r
        paid(batch, got)
    for k in range(cfg.retry_rounds):

        def again(r: Any) -> bool:  # a budget stop or a refusal of that very text is never sent again
            err = r.error or ""
            return not r.ok and not err.startswith("budget") and not _refusal(err)

        retry = [i for i, r in enumerate(res) if again(r)]
        if not retry:
            break
        time.sleep(cfg.retry_wait_s * k)
        got = det.score([texts[i] for i in retry])
        for i, r in zip(retry, got, strict=True):
            res[i] = r
        paid(retry, got)
    scores = [r.score if r.ok else math.nan for r in res]
    errors = [None if r.ok else (r.error or "error")[:200] for r in res]
    budget = any((e or "").startswith("budget") for e in errors)
    return scores, words, errors, budget, sorted({r.version for r in res if r.ok and r.version})


def _r(x: float, digits: int) -> float | None:
    x = float(x)
    return round(x, digits) if math.isfinite(x) else None


LEVELS = (1.0, 0.75, 0.5, 0.0)  # D-22 (0.75: G3 minor slips, owner decision 2026-11-20)


def _lv(level: float | str) -> str:
    """Level key for public counts: "1", "0.75", "0.5" or "0"."""
    x = float(level)
    return "1" if x >= 1.0 else "0.75" if x >= 0.75 else "0.5" if x >= 0.5 else "0"


def cache_epochs(detectors: Any) -> dict[str, str]:
    """What the caches' keys depend on, per kind of purchase: the judges' prompts and config (a change makes
    every judge call a miss) and each detector's config (part of its cache key)."""
    import hashlib

    from humanizer_eval.evaluation.conditions import PKG

    h = hashlib.sha256()
    for f in ("gates/judges.py", "gates/meaning.py"):
        with contextlib.suppress(OSError):
            h.update((PKG / f).read_bytes())
    with contextlib.suppress(OSError):
        h.update(Path("configs/gates.json").read_bytes())
    out = {"judges": h.hexdigest()[:12]}
    for n, d in (detectors or {}).items():
        out[n] = str(getattr(d, "config", None) or type(d).__name__)
    return out


#: texts per stored metering step (judges and detectors): a kill loses at most one such batch of paid calls
PROGRESS_EVERY = 25


class _Meterbook:
    """What this attempt paid for, written to the store as it happens (after the judges, after each
    detector): a crash or a kill then loses nothing, and the resume's cache hits are not paid twice."""

    def __init__(self, store: Any, sid: str, meter: Any, epochs: dict[str, str] | None = None):
        import uuid

        self.store, self.sid, self.meter, self.epochs = store, sid, meter, epochs
        self.attempt = uuid.uuid4().hex[:12]  # one attempt: within it judge purchases add up
        self.by_out: dict[int, dict[str, tuple[int, int]]] = {}
        self.judges: dict[str, tuple[int, int]] = {}
        self.det_n: dict[str, int] = {}  # detector -> texts of it already stored

    def flush(self, det_words: dict[str, list[int]]) -> None:
        cur = self.meter.judge_tokens()
        dj = {}
        for m, (a, b) in cur.items():
            x, y = self.judges.get(m, (0, 0))
            if a - x or b - y:
                dj[m] = (a - x, b - y)
        dd = {  # a detector seen for the first time is stored even with no words (its line on the statement)
            d: w[self.det_n.get(d, 0) :]
            for d, w in det_words.items()
            if d not in self.det_n or len(w) > self.det_n[d]
        }
        cur_o = self.meter.judge_tokens_by_output() if hasattr(self.meter, "judge_tokens_by_output") else {}
        do: dict[int, dict[str, tuple[int, int]]] = {}
        for i, by_m in cur_o.items():
            for m, (a, b) in by_m.items():
                x, y = self.by_out.get(i, {}).get(m, (0, 0))
                if a - x or b - y:
                    do.setdefault(i, {})[m] = (a - x, b - y)
        if dj or dd or do:
            self.store.add_metered(self.sid, dj, dd, self.epochs, judge_by_output=do, attempt=self.attempt)
        self.by_out = cur_o
        self.judges = dict(cur)
        for d, w in det_words.items():
            self.det_n[d] = max(self.det_n.get(d, 0), len(w))


def _gate_and_detect(
    ctx, outs, gate_factory, meter, detectors, watch, cfg, store, sid, hk, det_words, book=None
):
    """Steps 3-4: gates, canaries, crown detectors. Raises EvaluationPaused; ``det_words`` is filled as
    detectors finish, so that a pause can keep what was paid."""
    from humanizer_eval.evaluation import conditions
    from humanizer_eval.llm.cache import Refresh

    # retries of failed outputs are asked again uncached and are NOT metered: a failure there is ours, and
    # the miner never pays twice for the same judgement (the owner pays the retry)
    flush = (lambda: book.flush(det_words)) if book is not None else None
    per_out = (lambda i: gate_factory(meter.for_output(i))) if hasattr(meter, "for_output") else None
    gres = _gate_all(gate_factory(meter), ctx, outs, cfg, retry_gates=gate_factory(Refresh(meter.llm)),
                     final_gates=gate_factory(Refresh(meter.llm, strict=True)), per_output=per_out,
                     progress=flush)  # fmt: skip
    if book is not None:
        book.flush(det_words)
    n_err = sum(r.status == "error" for r in gres)
    if n_err:  # never a level for an output our judges did not finish: everyone is judged completely
        store.set_stage(sid, hk, "paused", reason="quality judges unavailable; the run resumes automatically")
        raise EvaluationPaused("gating", f"{n_err} of {ctx.n} outputs could not be judged")
    levels = np.array([r.level for r in gres])

    # 3b. the detector models are still the ones H and the king were scored by (canaries, uncached)
    try:
        problems = conditions.check(ctx.meta.get("detector_baseline"), watch)
    except RuntimeError as e:  # a canary got no answer: the detector is unavailable, not changed
        store.set_stage(sid, hk, "paused", reason="a detector is unavailable; the run resumes automatically")
        raise EvaluationPaused("detecting", str(e)) from e
    if problems:
        store.set_stage(
            sid, hk, "paused", reason="a detector changed during the period; the run waits for the owner"
        )
        raise ConditionsChanged("detecting", "; ".join(problems))

    # 4. crown detectors on the outputs that can still count
    store.set_stage(sid, hk, "detecting")
    send = [i for i in range(ctx.n) if outs[i].strip() and (levels[i] > 0 or not cfg.skip_severe)]
    scores: dict[str, list[float]] = {}
    det_errors: dict[str, dict[int, str]] = {}
    det_versions: dict[str, list[str]] = {}
    for name in ctx.detectors:

        def stored(w: list[int], name: str = name) -> None:
            if book is not None:
                det_words[name] = w
                book.flush(det_words)

        s, words, errs, budget, vers = _score_detector(detectors[name], [outs[i] for i in send], cfg, stored)
        det_versions[name] = vers
        # a vendor that refuses particular texts (a text-level 4xx, never credit/quota/auth): those texts are
        # the miner's doing, counted as caught however many, even all of them: the vendor is up (its
        # uncached canaries just answered, step 3b), so no count of refusals can make it look like our
        # outage; anything else (5xx, timeouts, budget) is ours
        refused = [k for k, e in enumerate(errs) if e is not None and _refusal(e)]
        if refused and not budget:
            for k in refused:
                s[k], errs[k] = math.inf, None
        bad = sum(e is not None for e in errs)
        det_words[name] = words  # paid even if this attempt pauses below
        if book is not None:
            book.flush(det_words)
        if budget or bad:  # never a "caught" score for a text our detector did not score
            store.set_stage(
                sid, hk, "paused", reason="a detector is unavailable; the run resumes automatically"
            )
            raise EvaluationPaused(
                "detecting", f"{name}: {bad} of {len(send)} texts failed (budget={budget})"
            )
        full = [math.inf] * ctx.n
        for i, x in zip(send, s, strict=True):
            full[i] = x
        scores[name] = full
        det_errors[name] = {send[k]: e for k, e in enumerate(errs) if e is not None}
    problems = conditions.check(ctx.meta.get("detector_baseline"), None, det_versions)
    if problems:
        store.set_stage(
            sid, hk, "paused", reason="a detector changed during the period; the run waits for the owner"
        )
        raise ConditionsChanged("detecting", "; ".join(problems))

    return gres, levels, send, scores, det_errors, det_versions


def evaluate_submission(
    sub: Submission,
    ctx: PeriodContext,
    *,
    runner: Runner,
    llm: Any,
    gate_factory: Callable[[Any], Any],
    detectors: Mapping[str, Detector],
    store: SubmissionStore,
    king: KingRun | None = None,
    billing: Any = None,  # billing.BillingService; None for owner runs (the king's re-run, baselines)
    king_state_path: Path | None = None,
    king_run_path: Path | None = None,
    cfg: EvalConfig | None = None,
    watch: Mapping[str, Detector] | None = None,  # the same detectors uncached: canaries (conditions.py)
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> EvalResult:

    cfg = cfg or EvalConfig()
    sid, hk = sub.submission_id, sub.hotkey
    if set(detectors) != set(ctx.detectors):
        raise ValueError(f"detectors {sorted(detectors)} differ from the period's H scores {ctx.detectors}")
    if king is not None and set(king.scores) != set(ctx.detectors):
        raise ValueError("the king run was scored with other detectors")
    if king is not None and king.fingerprint != cfg.fingerprint:
        raise ValueError(
            "the king's run was scored under other conditions; rescore it first (worker.ensure_king)"
        )
    t_start = time.time()

    # 0. quote and hold (the result is held until the payment is confirmed: no hold, no run)
    if billing is not None:
        hold = billing.accounts.hold(sid)
        if hold is None:
            dets = [d for d in ctx.detectors if d in billing.card.detectors]
            q = billing.make_quote(hk, ctx.input_words(), detectors=dets)
            due = billing.amount_due(q["id"])
            if due > 0:
                store.set_stage(
                    sid,
                    hk,
                    "awaiting_payment",
                    quote_id=q["id"],
                    amount_due_cents=due,
                    quote_total_cents=q["total_cents"],
                )
                raise InsufficientBalance(q, due)
            billing.start(q["id"], sid)
        elif hold.status != "open":
            raise ValueError(f"submission {sid} was already {hold.status}")

    # 1. run the bundle, or resume from the saved outputs
    run = store.load_run(sid)
    if run is None:
        store.set_stage(sid, hk, "running")
        run = runner.run(sub, ctx.harness_items(), ctx.harness_seed, store.dir(sid) / "run")
        if run.status == "infra_error":
            # ours (no pod, a lost pod, our docker/GPU): the same submission, the same pinned commit, is run
            # again later with its hold kept; the miner never has to resubmit (the first commitment stands)
            store.set_stage(sid, hk, "paused", reason="our infrastructure failed; the run is retried")
            raise EvaluationPaused("running", f"our side: {run.reason}")
        store.save_run(sid, run)

    # 2. G0, then 3. gates G1-G3 (metered)
    store.set_stage(sid, hk, "gating")
    outs, g0 = _g0(run.outputs, ctx)
    meter = MeteringLLM(llm)
    det_words: dict[str, list[int]] = {}
    book = _Meterbook(store, sid, meter, cache_epochs(detectors))
    try:
        gres, levels, send, scores, det_errors, det_versions = _gate_and_detect(
            ctx, outs, gate_factory, meter, detectors, watch, cfg, store, sid, hk, det_words, book
        )
    except BaseException as e:  # a pause, a crash, a kill signal turned into an exception
        # what this attempt paid for comes back as free cache hits on the resume: keep it for the statement.
        book.flush(det_words)
        if isinstance(e, ConditionsChanged):  # not the changed detectors' words: their rescore pays again
            changed = {p.split(":", 1)[0].strip() for p in e.reason.split("; ") if ":" in p}
            store.drop_metered_detectors(sid, changed & set(det_words))
        raise
    book.flush(det_words)
    n_err = 0  # a run with judge errors paused above
    metered = store.load_metered(sid)  # every attempt's paid usage, this one included
    # 5. score and the crown test
    store.set_stage(sid, hk, "scoring")
    srun = SubmissionRun(scores, levels, ctx.strata, ctx.input_ids)
    human = ctx.human_reference()
    sc = submission_score(srun, human)
    decision = None
    if king is not None and king.submission_id != sid:
        decision = crown_decision(
            king.to_run(ctx.strata), srun, human, seed=ctx.crown_seed(sid), **cfg.crown_kw
        )
    crowned = bool(decision and decision.crown)

    # 6. statement and settlement
    statement = settlement = to_settle = None
    if billing is not None:
        card = billing.card_for(sid) if hasattr(billing, "card_for") else billing.card
        judge_tokens = {k: tuple(v) for k, v in metered["judge_tokens"].items()}
        off_card = sorted(set(judge_tokens) - set(card.raw["judges"]))
        st = make_statement(
            card,
            detector_words={d: list(w) for d, w in metered["detector_words"].items() if d in card.detectors},
            judge_tokens={m: t for m, t in judge_tokens.items() if m in card.raw["judges"]},
            pod_hours=run.pod_seconds / 3600.0,
            meta={
                "submission_id": sid,
                "period_id": ctx.period_id,
                "n_outputs": ctx.n,
                "sent_to_detectors": len(send),
                "off_card_models": off_card,
            },
        )
        to_settle = st  # settled last, after the record: a crash in between re-runs from the saved outputs

    # 7. the king (now, or after the caller archived the bundle)
    king_entry = None
    pending = crowned and cfg.defer_crown
    if crowned and cfg.update_king and not pending:
        king_entry = apply_crown(
            KingRun(
                sid, hk, ctx.input_ids, levels.tolist(), scores, fingerprint=cfg.fingerprint, h_id=ctx.h_id
            ),
            decision.delta,
            king,
            reg_block=sub.reg_block,
            king_state_path=king_state_path,
            king_run_path=king_run_path,
            key=store.key,
            now=now,
        )

    # 8. records
    d = cfg.public_digits
    levels_count = {str(k): int((levels == k).sum()) for k in LEVELS}
    gate_counts: dict[str, dict[str, int]] = {}
    for r in gres:
        for g, lv in (r.levels or {}).items():
            c = gate_counts.setdefault(g, {"1": 0, "0.75": 0, "0.5": 0, "0": 0})
            c[_lv(lv)] += 1
    public: dict[str, Any] = {
        "period_id": ctx.period_id,
        "S": _r(sc.s, d),
        "crowned": crowned and not pending,
        "bundle_ran": run.status == "ok",
        # aggregates only (owner decision 2026-11-13): never judges' reasons, which quote the hidden texts
        "levels": {_lv(k): v for k, v in levels_count.items()},
        "gates": gate_counts,
        "judge_errors": int(n_err),
        "detectors": {
            det: {
                "E": _r(sc.evasion[det], d),
                "auroc": _r(sc.auroc[det], d),
                "p_human": _r(sc.p_human[det], d),
            }
            for det in sorted(sc.evasion)
        },
    }
    if run.status != "ok":
        public["bundle_error"] = run.reason  # our wording, set by the runner
    if decision is not None:
        public.update(
            {
                "S_king": _r(decision.s_king, d),
                "delta": _r(decision.delta, d),
                "lower_bound": _r(decision.lower, d),
                "alpha": decision.alpha,
            }
        )
    private = {
        "period_id": ctx.period_id,
        "submission": {"id": sid, "hotkey": hk, "name": sub.name, "bundle_sha256": sub.bundle_sha256},
        "run": {
            "status": run.status,
            "reason": run.reason,
            "pod_seconds": run.pod_seconds,
            "cost_usd": run.cost_usd,
            "meta": run.meta,
        },
        "g0": g0,
        "outputs": outs,
        "levels": levels.tolist(),
        "levels_count": levels_count,
        "gates": [
            {
                "status": r.status,
                "level": r.level,
                "levels": r.levels,
                "reasons": r.reasons,
                "detail": r.detail,
            }
            for r in gres
        ],
        "scores": {k: [None if not math.isfinite(x) else x for x in v] for k, v in scores.items()},
        "detector_errors": det_errors,
        "detector_versions": det_versions,
        "fingerprint": cfg.fingerprint,
        "h_id": ctx.h_id,
        "score": {
            "S": sc.s,
            "auroc": sc.auroc,
            "p_human": sc.p_human,
            "evasion": sc.evasion,
            "severe_rate": sc.severe_rate,
            "minor_rate": sc.minor_rate,
            "excluded_strata": list(sc.excluded_strata),
        },
        "crown": None
        if decision is None
        else {
            **decision.__dict__,
            "reasons": list(decision.reasons),
            "king": king.submission_id,
            "king_run": getattr(king, "record_id", None),
            "decided_at": time.time(),  # pending crowns are applied in the order they were decided
        },
        "king_entry": None if king_entry is None else king_entry.to_dict(),
        "judge_tokens": meter.judge_tokens(),
        "judge_cost_usd": meter.cost_usd,
        "started_at": t_start,
        "seconds": round(time.time() - t_start, 1),
    }
    if pending:
        public["crown_pending"] = True
    store.save_record(sid, private)
    stage = "archiving" if pending else "done"
    pub = store.set_stage(sid, hk, stage, **public)
    if to_settle is not None:  # the result and a pending crown are on disk first; then the charge
        settlement = billing.finish(sid, to_settle)
        statement = to_settle.to_dict()
        store.save_statement(sid, {**statement, "settlement": settlement})
        pub = store.set_stage(sid, hk, stage, **{**public, "charged_cents": settlement["charged_cents"]})
    return EvalResult(sid, "done", pub, private, statement, settlement, crowned)


def apply_crown(
    run: KingRun,
    delta: float,
    king: KingRun | None,
    *,
    reg_block: int | None,
    king_state_path: Path | None,
    king_run_path: Path | None,
    key: bytes,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> Any:
    """Make ``run`` the king: append it to the king state (created from ``king`` if missing) and save it
    as the king's run on this period. Returns the new ``KingEntry`` (None without a state path)."""
    entry = None
    if king_state_path is not None:
        kp = Path(king_state_path)
        if kp.exists():
            state = KingState.from_json(kp.read_text())
        elif king is not None:
            state = KingState.initial(king.submission_id, king.hotkey, now())
        else:
            state = KingState()
        entry = state.apply_crown(run.submission_id, run.hotkey, now(), delta, registered_at=reg_block)
        kp.parent.mkdir(parents=True, exist_ok=True)
        tmp = kp.with_name(kp.name + ".tmp")  # the crown ledger: a torn write would lose every crown
        tmp.write_text(state.to_json())
        tmp.replace(kp)
    if king_run_path is not None:
        run.save(king_run_path, key)
    return entry
