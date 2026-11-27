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
        (self.dir(sid) / name).write_bytes(encrypt_at_rest(self.key, json.dumps(obj, default=str).encode()))

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
                "meta": run.meta,
                "cost_usd": run.cost_usd,
            },
        )

    def load_run(self, sid: str) -> RunOutput | None:
        d = self._get(sid, "outputs.enc")
        if d is None:
            return None
        return RunOutput(d["status"], d["outputs"], d["pod_seconds"], d["reason"], d["meta"], d["cost_usd"])

    def save_record(self, sid: str, record: dict[str, Any]) -> None:
        self._put(sid, "record.enc", record)

    def load_record(self, sid: str) -> dict[str, Any] | None:
        return self._get(sid, "record.enc")

    def save_statement(self, sid: str, st: dict[str, Any]) -> None:
        (self.dir(sid) / "statement.json").write_text(json.dumps(st, indent=1))


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


def _gate_all(gates: Any, ctx: PeriodContext, outs: list[str], cfg: EvalConfig) -> list[GateResult]:
    """Every output judged; outputs whose judges failed are asked again, ``cfg.retry_rounds`` times."""

    def one(i: int) -> GateResult:
        return gates.check(ctx.items[i]["text"], outs[i], ctx.items[i].get("facts") or [])

    with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
        res = list(ex.map(one, range(len(outs))))
        for k in range(cfg.retry_rounds):
            retry = [i for i, r in enumerate(res) if r.status == "error"]
            if not retry:
                break
            time.sleep(cfg.retry_wait_s * k)
            for i, r in zip(retry, ex.map(one, retry), strict=True):
                res[i] = r
    return res


def _score_detector(
    det: Detector, texts: list[str], cfg: EvalConfig
) -> tuple[list[float], list[int], list[str | None], bool, list[str]]:
    """Scores, metered words (sent and not cached), errors, budget hit, model versions seen. Failed texts
    are sent again ``cfg.retry_rounds`` times; the caller pauses if any is still unscored."""
    res = det.score(texts)
    for k in range(cfg.retry_rounds):
        retry = [i for i, r in enumerate(res) if not r.ok and not (r.error or "").startswith("budget")]
        if not retry:
            break
        time.sleep(cfg.retry_wait_s * k)
        for i, r in zip(retry, det.score([texts[i] for i in retry]), strict=True):
            res[i] = r
    scores = [r.score if r.ok else math.nan for r in res]
    words = [word_count(t) for t, r in zip(texts, res, strict=True) if r.ok and not r.cached]
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
    from humanizer_eval.evaluation import conditions

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
            if billing is not None:
                billing.fail(sid)
            store.save_record(sid, {"run": {"status": run.status, "reason": run.reason, "meta": run.meta}})
            pub = store.set_stage(
                sid, hk, "failed", reason="evaluation infrastructure failed; nothing was charged"
            )
            return EvalResult(sid, "failed", pub, {"run_reason": run.reason})
        store.save_run(sid, run)

    # 2. G0, then 3. gates G1-G3 (metered)
    store.set_stage(sid, hk, "gating")
    outs, g0 = _g0(run.outputs, ctx)
    meter = MeteringLLM(llm)
    gres = _gate_all(gate_factory(meter), ctx, outs, cfg)
    n_err = sum(r.status == "error" for r in gres)
    if n_err:  # never a level for an output our judges did not finish: everyone is judged completely
        store.set_stage(sid, hk, "paused", reason="quality judges unavailable; the run resumes automatically")
        raise EvaluationPaused("gating", f"{n_err} of {ctx.n} outputs could not be judged")
    levels = np.array([r.level for r in gres])

    # 3b. the detector models are still the ones H and the king were scored by (canaries, uncached)
    problems = conditions.check(ctx.meta.get("detector_baseline"), watch)
    if problems:
        store.set_stage(
            sid, hk, "paused", reason="a detector changed during the period; the run waits for the owner"
        )
        raise ConditionsChanged("detecting", "; ".join(problems))

    # 4. crown detectors on the outputs that can still count
    store.set_stage(sid, hk, "detecting")
    send = [i for i in range(ctx.n) if outs[i].strip() and (levels[i] > 0 or not cfg.skip_severe)]
    scores: dict[str, list[float]] = {}
    det_words: dict[str, list[int]] = {}
    det_errors: dict[str, dict[int, str]] = {}
    det_versions: dict[str, list[str]] = {}
    for name in ctx.detectors:
        s, words, errs, budget, vers = _score_detector(detectors[name], [outs[i] for i in send], cfg)
        det_versions[name] = vers
        bad = sum(e is not None for e in errs)
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
        det_words[name] = words
        det_errors[name] = {send[k]: e for k, e in enumerate(errs) if e is not None}
    problems = conditions.check(ctx.meta.get("detector_baseline"), None, det_versions)
    if problems:
        store.set_stage(
            sid, hk, "paused", reason="a detector changed during the period; the run waits for the owner"
        )
        raise ConditionsChanged("detecting", "; ".join(problems))

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
    statement = settlement = None
    if billing is not None:
        card = billing.card
        judge_tokens = meter.judge_tokens()
        off_card = sorted(set(judge_tokens) - set(card.raw["judges"]))
        st = make_statement(
            card,
            detector_words={d: w for d, w in det_words.items() if d in card.detectors},
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
        settlement = billing.finish(sid, st)
        statement = st.to_dict()
        store.save_statement(sid, {**statement, "settlement": settlement})

    # 7. the king (now, or after the caller archived the bundle)
    king_entry = None
    pending = crowned and cfg.defer_crown
    if crowned and cfg.update_king and not pending:
        king_entry = apply_crown(
            KingRun(sid, hk, ctx.input_ids, levels.tolist(), scores, fingerprint=cfg.fingerprint),
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
    if statement is not None:
        public["charged_cents"] = settlement["charged_cents"]
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
        else {**decision.__dict__, "reasons": list(decision.reasons), "king": king.submission_id},
        "king_entry": None if king_entry is None else king_entry.to_dict(),
        "judge_tokens": meter.judge_tokens(),
        "judge_cost_usd": meter.cost_usd,
        "started_at": t_start,
        "seconds": round(time.time() - t_start, 1),
    }
    if pending:
        public["crown_pending"] = True
    store.save_record(sid, private)
    pub = store.set_stage(sid, hk, "archiving" if pending else "done", **public)
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
        kp.write_text(state.to_json())
    if king_run_path is not None:
        run.save(king_run_path, key)
    return entry
