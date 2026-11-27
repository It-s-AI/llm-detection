"""The queue worker: one submission at a time through ``evaluate_submission`` (evaluation/pipeline.py).

Before any challenger of a period runs, the king must have a run on that period's inputs (D-11):
* the owner's baseline (inputs unchanged) is scored by replay, no pod;
* a miner king's bundle is run again on the new inputs at the owner's cost (D-63: its S on D_{t+1} is
  the free signal published next to the coronation).

Outcomes of one evaluation:
* done / failed (``evaluate_submission`` decided; money settled or released there);
* paused (judges or detectors unavailable): retried later with backoff, the hold and outputs kept; after
  ``max_pauses`` the owner gives up, the hold is released and nothing is charged;
* our bug (any other exception): retried up to ``max_attempts``, then released like a pause give-up.
A restart puts a row left ``running`` back in the queue; saved outputs are reused, the pod is not rerun.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from humanizer_eval.evaluation.context import KingRun, PeriodContext
from humanizer_eval.evaluation.pipeline import (
    ConditionsChanged,
    EvalConfig,
    EvaluationPaused,
    InsufficientBalance,
    SubmissionStore,
    apply_crown,
    evaluate_submission,
)
from humanizer_eval.evaluation.runners import ReplayRunner, Runner, Submission
from humanizer_eval.intake.queue import Row, SubmissionQueue
from humanizer_eval.runner.bundle import MAX_BUNDLE_BYTES
from humanizer_eval.scoring.crown import KingState

log = logging.getLogger(__name__)

#: in a period's directory: a detector answered differently from when H was scored. Every evaluation of the
#: period is held, without calling anything, until scripts/rescore_period.py moves this file aside. A
#: different answer is not a failure that goes away: retrying only pays for the same answer again.
CONDITIONS_FILE = "CONDITIONS_CHANGED"
#: re-runs of a miner king that failed on our side before the owner is alerted; the retries go on with a
#: backoff up to KING_RETRY_MAX_S (the king is never scored as failed for our failures)
KING_MAX_FAILED_RUNS = 3
#: failures that rented no pod (a GPU shortage): they never use up a king re-run attempt
NO_POD = ("rent refused", "no suitable executor", "executor list failed")


def _no_pod(reason: str) -> bool:
    """No pod was had in this attempt: a refusal or a shortage, and no rent that may have been billed."""
    return any(k in reason for k in NO_POD) and "rent failed" not in reason


#: a submission whose evaluation kept failing on our side waits for the owner (scripts/parked.py)
PARKED_S = 10 * 365 * 86400.0
#: how long a copy waits for an earlier commitment of its bundle that has not paid yet
AUTHOR_WAIT_S = 2 * 3600.0
#: how long a just-paid row whose repo is still private waits before it is refused (the miner opens it)
PRIVATE_GRACE_S = 15 * 60.0
KING_RETRY_MAX_S = 2 * 3600.0
OWNER_HOTKEY = "owner-baseline"
BASELINE_ID = "baseline-identity"


@dataclass
class WorkerEnv:
    period: Callable[[], PeriodContext]
    store_for: Callable[[str], SubmissionStore]
    llm: Any
    gate_factory: Callable[[Any], Any]
    detectors: Mapping[str, Any]
    runner_for: Callable[[Submission], Runner]
    king_run_path: Callable[[PeriodContext], Path]
    king_state_path: Path
    key: bytes
    cfg: EvalConfig
    #: ``(hotkey, registered_at) -> bool`` from the metagraph snapshot; None: every king counts
    is_registered: Callable[[str, int | None], bool] | None = None
    hub: Any = None  # intake.hf.HubClient: download links for bundles pinned on Hugging Face
    archive: Any = None  # intake.archive.Archive: with it, a crown counts once the bundle is archived
    secret_key: bytes | None = None  # decrypts miners' read tokens for private repos
    max_bundle_bytes: int = MAX_BUNDLE_BYTES
    #: the crown detectors again WITHOUT a cache: canaries for vendors that report no model version
    #: (evaluation/conditions.py); passed to every evaluation, the king's and the challengers' alike
    detector_watch: Mapping[str, Any] | None = None


class KingUnavailable(RuntimeError):
    """The king's run on the current period could not be made yet (retry later)."""


class QueueWorker:
    def __init__(
        self,
        queue: SubmissionQueue,
        service: Any,  # intake.service.IntakeService (enqueue_ready)
        billing: Any,
        env: WorkerEnv,
        *,
        max_attempts: int = 3,
        # local retries only: 15 min, 30, 1 h, 2 h (about 4 h), then the submission is parked for the owner,
        # who is alerted at the first pause and looks within hours
        max_pauses: int = 4,
        pause_base_s: float = 900.0,
        pause_max_s: float = 2 * 3600.0,
        retry_s: float = 600.0,
        concurrency: int = 1,
        alerter: Any = None,  # ops.alerts.Alerter: failures, crowns, a held queue
        now: Callable[[], float] = time.time,
    ):
        self.queue = queue
        self.service = service
        self.billing = billing
        self.env = env
        self.max_attempts = max_attempts
        self.max_pauses = max_pauses
        self.pause_base_s = pause_base_s
        self.pause_max_s = pause_max_s
        self.retry_s = retry_s
        self.concurrency = max(1, concurrency)
        self.alerter = alerter
        self._now = now
        #: one king re-run at a time per period (the next period's pod run must not stop the current queue)
        self._king_locks: dict[str, threading.Lock] = {}
        self._king_locks_guard = threading.Lock()
        #: period id -> (retry at, wait, reason): a failed king run is not tried again before that time.
        #: Every attempt pays the detectors for the canaries, so a loop without it burns money.
        self._king_wait: dict[str, tuple[float, float, str]] = {}
        self._king_pod_tried: dict[str, bool] = {}  # period -> a paid attempt was made (ensure_king)
        self._conditions: dict[str, dict[str, Any]] = {}  # periods without a directory (tests, --period)
        self._king_fails: dict[tuple[str, str], int] = {}  # (period, king) -> re-runs failed on our side
        self._crown_lock = (
            threading.Lock()
        )  # crowns are decided one at a time, against the king of that moment

    def _alert(self, text: str, dedupe: str | None = None) -> None:
        if self.alerter is not None:
            try:  # an alert that fails must never stop a step (or end the worker thread)
                self.alerter.send(text, dedupe=dedupe)
            except Exception as e:
                log.warning("alert not sent (%s): %s", type(e).__name__, text)

    # --- the king's run on this period -------------------------------------------------------------
    def ensure_king(self, ctx: PeriodContext) -> KingRun:
        """The current king's run on this period: the stored one if it is that king's, else a new run.

        The current king is the latest coronation whose hotkey still holds the registration it
        submitted from; if a king lost its registration, the crown returns to the previous one and
        that king is run on this period (D-66).

        The stored run is used only if it was scored under the current scoring fingerprint
        (evaluation/conditions.py). Otherwise the king is rescored from his saved outputs (no pod, the
        same outputs) before anyone is compared with him: king and challenger always under identical
        conditions.
        """
        env = self.env
        path = env.king_run_path(ctx)
        sp = Path(env.king_state_path)
        cur = KingState.from_json(sp.read_text()).current_active(env.is_registered) if sp.exists() else None
        want = cur.submission_id if cur else BASELINE_ID
        kr = KingRun.load(path, env.key)
        if (
            kr is not None
            and kr.submission_id == want
            and kr.fingerprint == env.cfg.fingerprint
            and kr.h_id == ctx.h_id
        ):
            return kr
        if kr is not None and kr.submission_id == want:
            log.warning("king %s was scored under other conditions (%s != %s); rescoring his saved outputs",
                        want, kr.fingerprint, env.cfg.fingerprint)  # fmt: skip
        elif kr is not None:
            log.warning("king %s lost its registration; the crown returns to %s", kr.submission_id, want)
        store = env.store_for(ctx.period_id)
        cfg = replace(env.cfg, update_king=False)
        common = {
            "llm": env.llm,
            "gate_factory": env.gate_factory,
            "detectors": env.detectors,
            "store": store,
            "cfg": cfg,
            "watch": env.detector_watch,
        }
        if cur is None or cur.is_baseline:
            # the owner's baseline is not run: it is fixed as "every output caught by every detector, every
            # gate passed" (owner decision 2026-11-26), what the unchanged AI inputs essentially score. No
            # judge or detector is called, so it is ready for any period at once.
            sid = cur.submission_id if cur else BASELINE_ID
            hk = cur.hotkey if cur else OWNER_HOTKEY
            n = len(ctx.input_ids)
            kr = KingRun(sid, hk, ctx.input_ids, [1.0] * n, {d: [1.0] * n for d in ctx.detectors},
                         fingerprint=env.cfg.fingerprint, h_id=ctx.h_id)  # fmt: skip
            kr.save(path, env.key)
            log.info("baseline king fixed on %s (S = 0: every output caught)", ctx.period_id)
            return kr
        row = self.queue.get(cur.submission_id)
        if row is None:
            raise KingUnavailable(f"the king's bundle ({cur.submission_id}) is not in the queue records")
        sid, hk = cur.submission_id, cur.hotkey
        # its own record in this period stays as the miner saw it; the owner's run has its own id, one per
        # scoring fingerprint: a rescore after a deploy never overwrites the run earlier decisions used
        fp = re.sub(r"[^A-Za-z0-9]", "", str(env.cfg.fingerprint or ""))[:12]
        # and one per scoring of H (a mid-period detector rescore): earlier decisions keep the run they used
        run_id = f"{sid}-kingrun-{fp}-{ctx.h_id}" if fp else f"{sid}-kingrun-{ctx.h_id}"
        sub = Submission(run_id, hk, bundle_url=row.bundle_url, bundle_sha256=row.bundle_sha256,
                         name=row.name or "", reg_block=row.reg_block)  # fmt: skip
        # evaluated in this very period (or rescored): same seed, same outputs, no pod
        saved = store.load_run(run_id) or store.load_run(f"{sid}-kingrun") or store.load_run(sid)
        if saved is None:  # his owner run of this period under an earlier fingerprint: rescored, no pod
            if kr is not None and kr.submission_id == sid and kr.record_id:
                saved = store.load_run(kr.record_id)
            for old in sorted(Path(store.root).glob(f"{sid}-kingrun*")) if saved is None else []:
                saved = store.load_run(old.name)
                if saved is not None:
                    break
        if saved is None and env.archive is not None and env.archive.has(sid):
            sub = replace(sub, bundle=env.archive.path(sid), bundle_url=None, bundle_sha256=None)
        elif saved is None and row.secret is None and row.hf_repo and env.archive is not None:
            raise KingUnavailable(f"the king's bundle ({sid}) is not in the archive")
        # every paid attempt (a pod, or a rescore of saved outputs that pays judges and detectors) is counted
        # on disk first: a kill loop or a re-run that keeps pausing stops after KING_MAX_FAILED_RUNS
        tries = path.with_name(f"{path.name}.{sid}.attempts")  # per king: another king is not blocked
        n = int(tries.read_text() or 0) if tries.exists() else 0
        if n >= KING_MAX_FAILED_RUNS:
            with contextlib.suppress(Exception):
                self._alert(f"king {sid}: {n} re-runs on {ctx.period_id} never finished; stopped (nothing "
                            f"more is paid). Delete {tries} to try again, or revoke the crown",
                            dedupe=f"king-tries-{ctx.period_id}")  # fmt: skip
            raise KingUnavailable(f"king re-run stopped after {n} attempts (delete {tries.name} to retry)")
        tries.parent.mkdir(parents=True, exist_ok=True)
        tries.write_text(str(n + 1))
        self._king_pod_tried[ctx.period_id] = True  # a paid attempt: its failure counts (_king)
        runner = ReplayRunner(saved.outputs) if saved is not None else env.runner_for(sub)
        try:
            r = evaluate_submission(sub, ctx, runner=runner, **common)
        except EvaluationPaused as e:
            if _no_pod(str(e)):  # no pod was had: nothing paid, the attempt is given back
                tries.write_text(str(n))
                self._king_pod_tried[ctx.period_id] = False
            raise KingUnavailable(f"king re-run paused: {e}") from e
        why0 = str(r.private.get("run_reason") or "") if r.status != "done" else ""
        if _no_pod(why0):
            tries.write_text(str(n))  # no pod was had: nothing paid, the attempt is given back
            self._king_pod_tried[ctx.period_id] = False
        if r.status != "done":
            # an infra error is ours (a bundle's own failure is scored, it never lands here): the king is
            # never zeroed for it; retries back off (see _king) and the owner is told
            key = (ctx.period_id, sid)
            n = self._king_fails.get(key, 0) + 1
            self._king_fails[key] = n
            why = r.private.get("run_reason")
            if n == 1 or n >= KING_MAX_FAILED_RUNS:  # at once, and again if it keeps failing
                self._alert(f"king {sid}: {n} re-runs on {ctx.period_id} failed on our side ({why}); "
                            "the queue waits, look at it", dedupe=f"king-failed-{ctx.period_id}")  # fmt: skip
            raise KingUnavailable(f"king re-run failed on our side ({n}): {why}")
        scores = {d: [float("inf") if x is None else x for x in v] for d, v in r.private["scores"].items()}
        kr = KingRun(sid, hk, ctx.input_ids, r.private["levels"], scores, fingerprint=env.cfg.fingerprint,
                     record_id=run_id, h_id=ctx.h_id)  # fmt: skip
        kr.save(path, env.key)
        path.with_name(f"{path.name}.{sid}.attempts").unlink(missing_ok=True)
        log.info("king %s scored on %s: S=%s", sid, ctx.period_id, r.public.get("S"))
        return kr

    # --- scoring conditions changed: held until the owner rescores -----------------------------------
    def conditions_changed(self, ctx: PeriodContext) -> dict[str, Any] | None:
        """The hold of this period (``CONDITIONS_FILE``), or None."""
        if ctx.dir is None:
            return self._conditions.get(ctx.period_id)
        f = Path(ctx.dir) / CONDITIONS_FILE
        return json.loads(f.read_text()) if f.exists() else None

    def _hold(self, ctx: PeriodContext, e: ConditionsChanged) -> None:
        dets = sorted({p.split(":", 1)[0].strip() for p in e.reason.split("; ") if ":" in p})
        info = {"reason": e.reason, "detectors": dets, "at": self._now()}
        if ctx.dir is None:
            self._conditions[ctx.period_id] = info
        else:
            (Path(ctx.dir) / CONDITIONS_FILE).write_text(json.dumps(info))
        log.error("scoring conditions of %s changed, every evaluation held: %s", ctx.period_id, e.reason)
        self._alert_held(ctx, info)

    def _alert_held(self, ctx: PeriodContext, info: dict[str, Any]) -> None:
        flags = " ".join(f"--detector {d}" for d in info.get("detectors") or []) or "--detector <name>"
        self._alert(
            f"{ctx.period_id}: a detector answers differently than when H was scored ({info['reason']}). "
            f"Every evaluation of {ctx.period_id} is held; no detector is called. If its model changed: "
            f"stop humeval-intake, rescore_period.py --period {ctx.period_id} {flags} --yes, start it",
            dedupe=f"conditions-{ctx.period_id}",
        )

    def _king(self, ctx: PeriodContext) -> KingRun:
        """``ensure_king`` with backoff: after a failure the same period is retried in ``retry_s``, then
        twice as long each time up to an hour; meanwhile the failure is raised again without any call.
        A change of the scoring conditions is not retried at all: the period is held (``CONDITIONS_FILE``)."""
        with self._king_locks_guard:
            lock = self._king_locks.setdefault(ctx.period_id, threading.Lock())
        with lock:
            if (info := self.conditions_changed(ctx)) is not None:
                self._alert_held(ctx, info)  # a reminder, at most hourly
                raise KingUnavailable(f"held, scoring conditions changed: {info['reason']}")
            held = self._king_wait.get(ctx.period_id)
            if held is not None and self._now() < held[0]:
                raise KingUnavailable(held[2])
            self._king_pod_tried[ctx.period_id] = False
            try:
                kr = self.ensure_king(ctx)
            except Exception as e:  # a detector error at the canary check counts too
                if isinstance(e.__cause__, ConditionsChanged):
                    self._hold(ctx, e.__cause__)
                    raise
                if not isinstance(e, KingUnavailable):
                    log.exception("king run on %s failed", ctx.period_id)
                wait = min(2 * held[1], KING_RETRY_MAX_S) if held is not None else self.retry_s
                # only tries that rented a pod count towards the stop: a stale metagraph, a disk or state read
                # error costs nothing and keeps the capped backoff (it must never hold the queue for good)
                fails = self._king_fails.get((ctx.period_id, "any"), 0) + int(
                    self._king_pod_tried.get(ctx.period_id, False)
                )
                self._king_fails[(ctx.period_id, "any")] = fails
                if (
                    self._king_pod_tried.get(ctx.period_id) and fails >= KING_MAX_FAILED_RUNS
                ):  # each try rented a pod: stop
                    wait = PARKED_S
                why = e if isinstance(e, KingUnavailable) else f"king run failed: {type(e).__name__}: {e}"
                msg = f"{why} (next try in {wait / 60:.0f} min)"
                # the backoff is stored before any alert: an alert that fails must never skip it (a pod rent)
                self._king_wait[ctx.period_id] = (self._now() + wait, wait, msg)
                with contextlib.suppress(Exception):
                    if wait == PARKED_S:
                        stop = (
                            f"king run on {ctx.period_id} failed {fails} times ({e}); stopped, no more pods"
                        )
                        self._alert(f"{stop}; restart humeval-intake to try again (or revoke the crown)",
                                    dedupe=f"king-stop-{ctx.period_id}")  # fmt: skip
                    if held is None:  # the owner hears of it at once
                        first = f"king run on {ctx.period_id} failed: {e}; retried with backoff (queue waits)"
                        self._alert(first, dedupe=f"king-fail-{ctx.period_id}")
                raise KingUnavailable(msg) from e
            self._king_wait.pop(ctx.period_id, None)
            self._king_fails.pop((ctx.period_id, "any"), None)
            return kr

    def prepare_next(self, ctx: PeriodContext) -> KingRun | None:
        """Get the upcoming period ready before the switch: its context is loaded (the caller did that) and
        the king's run on its inputs exists. ``<period dir>/READY`` tells the rotation it can switch."""
        try:
            kr = self._king(ctx)
        except KingUnavailable as e:
            log.warning("next period %s not ready: %s", ctx.period_id, e)
            self._alert(f"next period {ctx.period_id} not ready: {e}", dedupe=f"prepare-{ctx.period_id}")
            return None
        if ctx.dir is not None:
            marker = ctx.dir / "READY"
            info = {"king": kr.submission_id, "at": self._now()}
            if not marker.exists() or json.loads(marker.read_text()).get("king") != kr.submission_id:
                marker.write_text(json.dumps(info))
                log.info("next period %s ready (king %s)", ctx.period_id, kr.submission_id)
        return kr

    # --- one step ------------------------------------------------------------------------------------
    def _read_bundle(self, row: Row, store: SubmissionStore) -> dict[str, Any] | None:
        """The bundle at the pinned commit, read when the evaluation starts. The repo may be private while
        the miner pays (minutes); it must be public by now, else the submission is refused at once: nothing
        is charged and the registration counts as used (private repos are not encouraged)."""
        from humanizer_eval.intake.hf import HubError, HubNotReadable

        try:
            info = self.env.hub.bundle_info(row.hf_repo, row.hf_revision)
            public = not info.private and not info.gated
        except HubNotReadable:
            info, public = None, False
        except HubError as e:  # the commit itself is unusable (no bundle.tar, ambiguous ref, too big...)
            return self._refuse(row, store, f"the pinned commit cannot be evaluated: {e}")
        except Exception as e:  # Hugging Face unavailable: a pause, like any outage (counted, bounded)
            log.warning("bundle of %s not read: %s", row.id, e)
            pauses = row.pauses + 1
            if pauses == 1:
                self._alert(
                    f"{row.id}: Hugging Face unavailable ({e}); retried for ~4 h", dedupe=f"hf-{row.id}"
                )
            if pauses > self.max_pauses:
                return self._park(row, store, "Hugging Face stayed unavailable")
            wait = min(self.pause_max_s, self.pause_base_s * 2 ** (pauses - 1))
            # (the pause is credited once: the attempt guards already subtract pauses from attempts)
            self.queue.update(row.id, state="paused", pauses=pauses, retry_at=self._now() + wait)
            return {"submission_id": row.id, "state": "paused"}
        if not public and self._now() - float(row.enqueued_ts or 0) < PRIVATE_GRACE_S:
            # just paid: the miner (or miner_submit.py) opens the repo right after paying; it may not have
            # yet. Waits a little, not a pause, nothing charged; refused only after the grace
            self.queue.update(row.id, state="paused", retry_at=self._now() + 60.0,
                              attempts=max(0, row.attempts - 1))  # fmt: skip
            self.queue.note(row.id, paused="the model repo is not public yet")
            return {"submission_id": row.id, "state": "paused"}
        if not public:
            return self._refuse(row, store, "the model repo was not public when the evaluation started")
        if row.bundle_sha256 and info.sha256 != row.bundle_sha256:
            return self._refuse(row, store, "the bundle at the pinned commit differs from the one submitted")
        if not 0 < info.size <= self.env.max_bundle_bytes:
            lim = self.env.max_bundle_bytes
            return self._refuse(row, store, f"bundle size {info.size} outside 1..{lim} bytes")
        if not row.bundle_sha256:
            self.queue.update(row.id, bundle_sha256=info.sha256, bundle_bytes=info.size)
        return None

    def _refuse(self, row: Row, store: SubmissionStore, reason: str) -> dict[str, Any]:
        """The miner's bundle cannot be evaluated: nothing charged, the registration is used (state done)."""
        h = self.billing.accounts.hold(row.id) if self.billing is not None else None
        if h is not None and h.status == "open":
            self.billing.fail(row.id)
        store.set_stage(row.id, row.hotkey, "failed", reason=f"{reason}; nothing was charged")
        self._close(row, "done", note=reason)
        return {"submission_id": row.id, "state": "failed"}

    def _park(self, row: Row, store: SubmissionStore, reason: str) -> dict[str, Any]:
        h = self.billing.accounts.hold(row.id) if self.billing is not None else None
        if h is not None and h.status == "settled":  # charged: its result stands; the owner finishes it
            return self._park_settled(row, store, reason)
        return self._park_open(row, store, reason)

    def _author(self, row: Row) -> str | None:
        """The earlier committer of this bundle (this row is the copy), None, or "wait": an unpaid earlier
        commitment with a live quote is waited for, at most AUTHOR_WAIT_S from the first wait (stored on
        the row), then only paid commitments count."""
        author = self.queue.committed_earlier(row)
        if author is None or not self.queue.author_unpaid(author):
            return author
        note = {}
        with contextlib.suppress(Exception):
            note = json.loads(row.note or "{}") or {}
        since = float(note.get("author_wait_since") or 0) or self._now()
        if not note.get("author_wait_since"):
            self.queue.note(row.id, author_wait_since=since,
                            paused="an earlier commitment of this bundle has not paid yet")  # fmt: skip
        if self._now() - since < AUTHOR_WAIT_S:
            return "wait"
        return self.queue.committed_earlier(row, unpaid=False)

    def _crown_wait(self, row: Row, store: SubmissionStore, why: str) -> dict[str, Any]:
        """A run that cannot continue now for a reason of ours (a charged run whose crown cannot be settled, a
        period or king run that cannot be loaded): the same bounded backoff as any pause of ours
        (15/30/60/120 min, an alert at the first), then parked for the owner. Each try may rescore (paid
        calls), so it is never retried on a fixed short loop."""
        # waiting for an earlier decided winner's crown is not a failure of this row (that one is bounded)
        with contextlib.suppress(Exception):
            other = self._earlier_pending_row(row, row.period_id, store) if row.period_id else None
            if other is not None:  # after that one (claimed after it), checked again at least every 2 h:
                # never copy a parked row's far retry time (it would hide this one for years)
                now = self._now()
                after = now + min(self.pause_max_s, max(0.0, float(other.retry_at or 0) - now)) + 60.0
                if float(other.retry_at or 0) - now > self.pause_max_s:
                    msg = f"{row.id} (charged) waits for the earlier crown of {other.id}, which is parked"
                    self._alert(f"{msg}: resume {other.id} (scripts/parked.py) and this one follows",
                                dedupe=f"waits-{row.id}")  # fmt: skip
                self.queue.update(row.id, state="paused", retry_at=after)
                self.queue.note(row.id, paused="an earlier decided crown is settled first")
                return {"submission_id": row.id, "state": "paused"}
        # its own count (an evaluation that used its pauses still gets the full backoff here)
        note = {}
        with contextlib.suppress(Exception):
            note = json.loads(row.note or "{}") or {}
        waits = int(note.get("crown_waits") or 0) + 1
        if waits == 1:
            self._alert(f"{row.id}: waits for us ({why}); retried with backoff",
                        dedupe=f"crownwait-{row.id}")  # fmt: skip
        if waits > self.max_pauses:
            return self._park(row, store, f"could not continue ({why})")
        wait = min(self.pause_max_s, self.pause_base_s * 2 ** (waits - 1))
        self.queue.update(row.id, state="paused", retry_at=self._now() + wait, pauses=row.pauses + 1)
        self.queue.note(row.id, paused=f"crown: {why}", crown_waits=waits)
        return {"submission_id": row.id, "state": "paused"}

    def _park_settled(self, row: Row, store: SubmissionStore, reason: str) -> dict[str, Any]:
        """Charged (its result stands), but finishing it keeps failing: parked for the owner, no more tries
        on its own (scripts/parked.py resume runs it again)."""
        self.queue.note(row.id, parked=reason)  # first: a kill between the two never hides the row
        self.queue.update(row.id, state="paused", retry_at=self._now() + PARKED_S, attempts=0,
                          pauses=self.max_pauses)  # fmt: skip
        how = f"scripts/parked.py resume {row.id}"
        self._alert(f"{row.id} (charged) parked: {reason}. Decide: {how}", dedupe=f"parked-{row.id}")
        return {"submission_id": row.id, "state": "paused"}

    def _park_open(self, row: Row, store: SubmissionStore, reason: str) -> dict[str, Any]:
        """Our side failed past every retry: the submission is never closed for it (the miner would have to
        resubmit and its first commitment would lose its place) and its money is not given back on its own
        (part of the work may be done). It keeps its hold and its place and waits for the owner, who decides:
        scripts/parked.py resume <id> (run it again) or release <id> (close it, the hold given back)."""
        self.queue.note(row.id, parked=reason)  # first: a kill between the two never hides the row
        self.queue.update(row.id, state="paused", retry_at=self._now() + PARKED_S, attempts=0,
                          pauses=self.max_pauses)  # fmt: skip
        store.set_stage(row.id, row.hotkey, "paused",
                        reason=f"waiting for us ({reason}); we look at it and continue it")  # fmt: skip
        how = f"scripts/parked.py resume {row.id} | release {row.id}"
        self._alert(f"{row.id} parked: {reason}. Decide: {how}", dedupe=f"parked-{row.id}")
        return {"submission_id": row.id, "state": "paused"}

    def _give_up(self, row: Row, store: SubmissionStore, reason: str) -> None:
        h = self.billing.accounts.hold(row.id) if self.billing is not None else None
        if (
            h is not None and h.status == "settled"
        ):  # charged already: the run stands, never 'nothing charged'
            store.set_stage(row.id, row.hotkey, "done", note=f"closed after settlement: {reason}")
            self._close(row, "done", note=reason)
            self._alert(
                f"evaluation {row.id} closed after it was charged: {reason}", dedupe=f"failed-{row.id}"
            )
            return
        if h is not None and h.status == "open":
            self.billing.fail(row.id)
        store.set_stage(row.id, row.hotkey, "failed", reason=reason)
        self._close(row, "failed")
        self._alert(f"evaluation {row.id} failed: {reason}", dedupe=f"failed-{row.id}")

    def run_once(self) -> dict[str, Any] | None:
        """Process at most one submission; None when there was nothing to do."""
        self.service.enqueue_ready()
        ctx = self.env.period()
        try:
            king = self._king(ctx)
        except KingUnavailable as e:
            log.warning("queue held: %s", e)
            self._alert(f"queue held: {e}", dedupe="king-unavailable")
            return None
        row = self.queue.claim(max_running=self.concurrency)
        if row is None:
            return None
        try:
            return self._run_claimed(row, ctx, king)
        except Exception as e:  # never leave a claimed row 'running' (it would hold a slot and the hold)
            log.exception("step on %s failed", row.id)
            self._alert(f"step on {row.id} failed ({type(e).__name__}: {e})", dedupe=f"step-{row.id}")
            cur = self.queue.get(row.id)
            if cur is not None and cur.state == "running":
                self.queue.note(row.id, error=type(e).__name__)
                h = self.billing.accounts.hold(row.id) if self.billing is not None else None
                if h is not None and h.status == "settled":  # charged: the attempts guard never parks it
                    with contextlib.suppress(Exception):
                        return self._crown_wait(cur, self.env.store_for(cur.period_id or row.period_id),
                                                f"step failed: {type(e).__name__}")  # fmt: skip
                self.queue.update(row.id, state="paused", retry_at=self._now() + self.retry_s)
            return {"submission_id": row.id, "state": "paused"}

    def _run_claimed(self, row: Row, ctx: PeriodContext, king: KingRun) -> dict[str, Any] | None:
        if self._current_period_id() not in (None, ctx.period_id):
            # the period switched while this step waited (e.g. for the king lock): never run on stale inputs
            self.queue.update(row.id, state="queued", attempts=max(0, row.attempts - 1))
            return None
        h = self.billing.accounts.hold(row.id) if self.billing is not None else None
        settled = h is not None and h.status == "settled"
        ran = bool(row.period_id) and self.env.store_for(row.period_id).load_run(row.id) is not None
        if (settled or ran) and row.period_id and row.period_id != ctx.period_id:
            # charged, or its bundle already ran, on its own period, which has switched since: it is finished
            # there (its outputs, its record, its king run, its inputs), never re-run on the new period
            get = getattr(self.env.period, "_get", None)
            try:
                if not callable(get):
                    raise RuntimeError("no period loader")
                ctx = get(row.period_id)
                # and that period's king: the run is compared with the king's run on its own inputs
                # (under the current scoring conditions: after a deploy that changed them, _king rescores
                # the king's saved outputs on that period, with its own backoff and stop)
                own_king = KingRun.load(self.env.king_run_path(ctx), self.env.key)
                if (
                    own_king is None
                    or own_king.fingerprint != self.env.cfg.fingerprint
                    or own_king.h_id != ctx.h_id
                    or own_king.submission_id != self._current_king_id()  # dethroned since: the current one
                ):
                    own_king = self._king(ctx)
                king = own_king
            except Exception as e:  # never settle it on another period's inputs: wait and try again
                log.warning("period %s of %s not loadable: %s", row.period_id, row.id, e)
                own_store = self.env.store_for(row.period_id)
                why = f"period {row.period_id} or its king run not ready: {e}"
                return self._crown_wait(row, own_store, why)
        if self.conditions_changed(ctx) is not None:  # its period is held (a detector changed): no paid call
            self.queue.update(row.id, state="paused", retry_at=self._now() + self.retry_s,
                              attempts=max(0, row.attempts - 1))  # fmt: skip
            return None
        store = self.env.store_for(ctx.period_id)
        if row.period_id != ctx.period_id:
            self.queue.update(row.id, period_id=ctx.period_id)
        # every claim counts an attempt; one that never came back (the process was killed mid-run, e.g. out
        # of memory) is seen only here: without this check a crash loop would rent a pod on every restart
        if row.attempts > self.max_attempts + row.pauses and not (h is not None and h.status == "settled"):
            log.error("%s: %d attempts (%d pauses), parked", row.id, row.attempts, row.pauses)
            return self._park(row, store, "our evaluation kept failing")
        if h is not None and h.status == "settled":  # settled before a crash: the result stands
            if not (store.dir(row.id) / "statement.json").exists():  # killed between the charge and it
                with contextlib.suppress(Exception):
                    st = self.billing.accounts.hold_statement(row.id)
                    if st is not None:
                        out = {"submission_id": row.id, "hold_cents": h.amount_cents,
                               "charged_cents": h.charged_cents,
                               "returned_cents": h.amount_cents - h.charged_cents}  # fmt: skip
                        store.save_statement(row.id, {**st, "settlement": out})
                        cur = store.status(row.id) or {}
                        skip = ("submission_id", "hotkey", "stage", "updated_at")
                        store.set_stage(row.id, row.hotkey, cur.get("stage") or "scoring",
                                        **{k: v for k, v in cur.items() if k not in skip},
                                        charged_cents=h.charged_cents)  # fmt: skip
            if store.load_record(row.id) is None:  # charged, but the crash came before the record was saved
                self._alert(f"{row.id}: charged but no result was recorded (crash during settlement); "
                            "look at it by hand", dedupe=f"norecord-{row.id}")  # fmt: skip
            crowned = None
            if store.load_record(row.id) is not None:  # a winner waiting, or a re-decision left undone
                if ctx.period_id != self.env.period().period_id:
                    king = KingRun.load(self.env.king_run_path(ctx), self.env.key) or king
                crowned = self._settle_crown(row, ctx, king, store)
                if crowned is None:  # the current king's run is not there yet: settle again later
                    return self._crown_wait(row, store, "the current king's run is not ready")
            self._close(row, "done", note="recovered after settlement")
            return {"submission_id": row.id, "state": "done", "crowned": crowned}
        # the hotkey must still hold the registration it submitted from: after a long queue or long pauses
        # (up to ~26 h of backoff) a challenger past its 24 h immunity may have been pruned, and a crown
        # would not count; nothing is run or charged then
        # read when the evaluation starts, not on a resume: the bundle already ran (and was paid for)
        if row.hf_repo and not row.owner_job and self.env.hub is not None and store.load_run(row.id) is None:
            waiting = self._read_bundle(row, store)
            if waiting is not None:
                return waiting
            row = self.queue.get(row.id) or row
        if self.env.hub is not None:  # earlier rows whose sha is unknown: a re-hosted copy has another repo
            for other in self.queue.unknown_earlier(row):
                try:
                    info = self.env.hub.bundle_info(other.hf_repo, other.hf_revision)
                    self.queue.update(other.id, bundle_sha256=info.sha256, bundle_bytes=info.size)
                except Exception as e:  # still private (it will be refused when due) or Hub trouble
                    log.info("sha of earlier %s not read: %s", other.id, e)
        # before any work only: once the bundle ran (paid), a resume finishes it and charges what was used;
        # the crown itself is refused later to a copy or a deregistered hotkey (_settle_crown / KingState)
        fresh = store.load_run(row.id) is None
        author = self._author(row) if fresh else None
        if author == "wait":  # an earlier commitment may still pay: wait (bounded), no charge
            self.queue.update(row.id, state="paused", retry_at=self._now() + self.retry_s,
                              attempts=max(0, row.attempts - 1))  # fmt: skip
            return {"submission_id": row.id, "state": "paused"}
        if author is not None:
            self._give_up(row, store, f"this exact bundle was committed on chain earlier by another hotkey "
                          f"({author}); it counts once; nothing was charged")  # fmt: skip
            return {"submission_id": row.id, "state": "failed"}
        reg = self.env.is_registered
        if (
            fresh
            and reg is not None
            and not row.owner_job
            and row.reg_block is not None
            and not reg(row.hotkey, row.reg_block)
        ):
            self._give_up(
                row, store, "the hotkey lost its registration before the evaluation; nothing was charged"
            )
            return {"submission_id": row.id, "state": "failed"}
        env = self.env
        cfg = replace(env.cfg, defer_crown=True)  # the crown is settled below, under the crown lock
        try:
            url = self._pod_url(row)  # a private repo without a signed link yet: a pause, nothing rented
            sub = Submission(row.id, row.hotkey, bundle_url=url, bundle_sha256=row.bundle_sha256,
                             name=row.name or "", reg_block=row.reg_block)  # fmt: skip
            res = evaluate_submission(
                sub,
                ctx,
                runner=env.runner_for(sub),
                llm=env.llm,
                gate_factory=env.gate_factory,
                detectors=env.detectors,
                store=store,
                king=king,
                billing=self.billing,
                king_state_path=env.king_state_path,
                king_run_path=env.king_run_path(ctx),
                cfg=cfg,
                watch=env.detector_watch,
            )
        except ConditionsChanged as e:  # held with the period: not a pause, not an attempt; nothing charged
            self._hold(ctx, e)
            # due again in a while (never 'now': claim() takes a NULL retry_at as due, a tight loop); the hold
            # check above returns it without any call until the owner has rescored
            self.queue.update(row.id, state="paused", retry_at=self._now() + self.retry_s,
                              attempts=max(0, row.attempts - 1))  # fmt: skip
            self.queue.note(row.id, paused=str(e))
            return {"submission_id": row.id, "state": "paused"}
        except EvaluationPaused as e:
            pauses = row.pauses + 1
            if pauses > self.max_pauses:
                return self._park(row, store, f"the evaluation services stayed unavailable ({e})")
            wait = min(self.pause_max_s, self.pause_base_s * 2 ** (pauses - 1))
            self.queue.update(row.id, state="paused", pauses=pauses, retry_at=self._now() + wait)
            self.queue.note(row.id, paused=str(e))
            if pauses == 1:  # the owner hears of it at once, not after the retries
                self._alert(f"evaluation {row.id} paused: {e}; retried for ~4 h, then parked for you",
                            dedupe=f"paused-{row.id}")  # fmt: skip
            return {"submission_id": row.id, "state": "paused"}
        except InsufficientBalance:  # no hold: should not happen for a queued row
            self._give_up(row, store, "payment not found; nothing was charged")
            return {"submission_id": row.id, "state": "failed"}
        except Exception as e:
            log.exception("evaluation of %s crashed", row.id)
            self._alert(f"evaluation {row.id} crashed ({type(e).__name__}), attempt {row.attempts}",
                        dedupe=f"crash-{row.id}-{row.attempts}")  # fmt: skip
            if row.attempts - row.pauses >= self.max_attempts:  # claims that ended in pauses do not count
                return self._park(row, store, f"our evaluation kept failing ({type(e).__name__})")
            self.queue.update(row.id, state="paused", retry_at=self._now() + self.retry_s)
            self.queue.note(row.id, error=type(e).__name__)
            return {"submission_id": row.id, "state": "paused"}
        crowned = self._settle_crown(row, ctx, king, store) if res.status == "done" else False
        if crowned is None:  # charged; the crown is settled once the current king's run exists
            return self._crown_wait(row, store, "the current king's run is not ready")
        state = "done" if res.status == "done" else "failed"
        self._close(row, state)
        return {"submission_id": row.id, "state": state, "crowned": crowned}

    # --- bundles on Hugging Face, archive, crown -----------------------------------------------------
    def _token(self, row: Row) -> str | None:
        if not row.secret or self.env.secret_key is None:
            return None
        from humanizer_eval.period.crypto import decrypt_at_rest

        return decrypt_at_rest(self.env.secret_key, bytes.fromhex(row.secret)).decode()

    def _pod_url(self, row: Row) -> str:
        """The pinned resolve URL, or a fresh signed link for a private repo (the token stays here)."""
        if not row.hf_repo or self.env.hub is None:
            return row.bundle_url
        token = self._token(row)
        try:
            return self.env.hub.download_url(row.hf_repo, row.hf_revision, token)
        except Exception as e:
            log.warning("no download link for %s: %s", row.id, e)
            if token is not None:  # the plain URL needs the token: the pod would fail, billed to the miner
                why = f"no signed download link yet ({type(e).__name__})"
                raise EvaluationPaused("downloading", why) from e
            return row.bundle_url  # a public repo: the pinned resolve URL works as it is

    def _close(self, row: Row, state: str, note: str | None = None) -> None:
        cols: dict[str, Any] = {"state": state, "finished_ts": self._now(), "secret": None}
        if note:
            cols["note"] = note
        self.queue.update(row.id, **cols)

    def _settle_crown(
        self, row: Row, ctx: PeriodContext, king_at_start: KingRun, store: SubmissionStore
    ) -> bool | None:
        """Decide the crown against the king of this moment (one decision at a time). The stored decision
        was made against ``rec['crown']['king']``; if the current king is someone else (another challenger
        was crowned meanwhile, or that king lost his registration and the crown went back, D-66), the
        decision is made again against the current king from the saved per-text scores (same texts, no new
        pod, D-11), for a loser too when the king it lost to dropped out. The run's own period counts, even
        if the period switched while it ran: the inputs, the king's run and the seed are those of the run.
        None: the current king's run on this period is not available yet; the row waits and is settled
        again later (its charge already stands)."""
        with self._crown_lock:
            rec = store.load_record(row.id) or {}
            st = store.status(row.id) or {}
            if self._already_crowned(row.id):  # crowned before a crash, the bookkeeping was not finished
                return self._close_crown(row, store, st)
            cr = rec.get("crown") or {}
            # the record is written first and is the source of truth: a crash between it and the status
            # (a re-decision that won) must resolve towards the record, never lose an earned crown
            won = not st.get("crown_revoked") and (
                (bool(st.get("crown_pending")) and bool(cr.get("crown", True)))
                or (bool(cr.get("crown")) and cr.get("king") == self._current_king_id())
            )
            reg = self.env.is_registered
            if (
                won
                and reg is not None
                and not row.owner_job
                and row.reg_block is not None
                and not reg(row.hotkey, row.reg_block)
            ):  # pruned while it waited: charged for its run (rules.md 3), no crown
                skip = ("submission_id", "hotkey", "stage", "updated_at", "crown_pending")
                pub = {k: v for k, v in st.items() if k not in skip}
                why = "the hotkey lost its registration before the crown: no crown"
                store.set_stage(row.id, row.hotkey, "done", **{**pub, "crowned": False, "crown_revoked": why})
                return False
            author = self._author(row) if won else None
            if author == "wait":
                return None  # an earlier commitment may still pay: the crown waits (bounded, AUTHOR_WAIT_S)
            if author is not None:  # a copy: charged for its run, no crown
                why = "this exact bundle was committed on chain earlier by another hotkey: no crown"
                skip = ("submission_id", "hotkey", "stage", "updated_at", "crown_pending")
                pub = {k: v for k, v in st.items() if k not in skip}
                store.set_stage(row.id, row.hotkey, "done", **{**pub, "crowned": False, "crown_revoked": why})
                return False
            decided_against = (rec.get("crown") or {}).get("king") or king_at_start.submission_id
            cur_id = self._current_king_id()
            if won and self._earlier_pending_crown(row, ctx.period_id, store):
                return None  # an earlier winner's crown waits (archiving): it is settled first, then this one
            if cur_id == decided_against:
                return self._finish_crown(row, ctx, king_at_start, store) if won else False
            if not won and not self._dropped_out(decided_against):
                return False  # it lost to a king who was then dethroned, not removed: the loss stands
            if rec.get("scores") is None:
                return False
            try:
                current = self._current_king_run(ctx)
            except KingUnavailable as e:
                log.warning("crown of %s waits for the current king's run: %s", row.id, e)
                return None
            try:
                if not self._redecide(row, ctx, current, rec, store):
                    return False
            except (KingUnavailable, EvaluationPaused) as e:  # the rescore could not finish now: later
                log.warning("crown of %s waits: %s", row.id, e)
                return None
            # a re-decision that now wins passes the same checks as any winner: its registration, a copy
            st = store.status(row.id) or {}
            skip = ("submission_id", "hotkey", "stage", "updated_at", "crown_pending")
            pub = {k: v for k, v in st.items() if k not in skip}
            reg = self.env.is_registered
            if (
                reg is not None
                and not row.owner_job
                and row.reg_block is not None
                and not reg(row.hotkey, row.reg_block)
            ):
                why = "the hotkey lost its registration before the crown: no crown"
                store.set_stage(row.id, row.hotkey, "done", **{**pub, "crowned": False, "crown_revoked": why})
                return False
            author = self._author(row)
            if author == "wait":
                return None
            if author is not None:
                why = "this exact bundle was committed on chain earlier by another hotkey: no crown"
                store.set_stage(row.id, row.hotkey, "done", **{**pub, "crowned": False, "crown_revoked": why})
                return False
            return self._finish_crown(row, ctx, current, store)

    def _earlier_pending_crown(self, row: Row, period_id: str, store: SubmissionStore) -> bool:
        return self._earlier_pending_row(row, period_id, store) is not None

    def _earlier_pending_row(self, row: Row, period_id: str, store: SubmissionStore) -> Row | None:
        """An earlier-queued winner of the same period whose crown is still pending (its archive failed for
        now): crowns follow the order the decisions were made in, not the order the archives succeeded."""
        for other in self.queue.by_state("paused", "queued"):  # a recovered row is queued again
            if other.id == row.id:
                continue
            other_store = self.env.store_for(other.period_id) if other.period_id else store
            # a strict order by when each crown was decided (ties: enqueue, id): never wait for each other
            mine = (self._decided_at(store, row.id), row.enqueued_ts or 0, row.id)
            theirs = (self._decided_at(other_store, other.id), other.enqueued_ts or 0, other.id)
            if theirs >= mine:
                continue
            st = other_store.status(other.id) or {}
            if st.get("stage") == "archiving" and st.get("crown_pending"):
                return other
        return None

    @staticmethod
    def _decided_at(store: SubmissionStore, sid: str) -> float:
        with contextlib.suppress(Exception):
            return float(((store.load_record(sid) or {}).get("crown") or {}).get("decided_at") or math.inf)
        return math.inf

    def _current_king_id(self) -> str:
        sp = Path(self.env.king_state_path)
        cur = (
            KingState.from_json(sp.read_text()).current_active(self.env.is_registered)
            if sp.exists()
            else None
        )
        return cur.submission_id if cur else BASELINE_ID

    def _dropped_out(self, sid: str) -> bool:
        """``sid`` was crowned but no longer counts (its hotkey lost the registration it submitted from)."""
        sp = Path(self.env.king_state_path)
        if not sp.exists() or sid == BASELINE_ID:
            return False
        st = KingState.from_json(sp.read_text())
        crowned = any(e.submission_id == sid for e in st.history)
        return crowned and not any(e.submission_id == sid for e in st.active_history(self.env.is_registered))

    def _current_king_run(self, ctx: PeriodContext) -> KingRun:
        """The current king (KingState, registered kings only) run on this period: the stored run if it is
        his, else ``_king`` makes it."""
        sp = Path(self.env.king_state_path)
        cur = (
            KingState.from_json(sp.read_text()).current_active(self.env.is_registered)
            if sp.exists()
            else None
        )
        want = cur.submission_id if cur else BASELINE_ID
        kr = KingRun.load(self.env.king_run_path(ctx), self.env.key)
        if (
            kr is not None
            and kr.submission_id == want
            and kr.fingerprint == self.env.cfg.fingerprint
            and kr.h_id == ctx.h_id
        ):
            return kr
        return self._king(ctx)

    def _current_period_id(self) -> str | None:
        name = getattr(self.env.period, "_name", None)  # CurrentPeriod: the CURRENT file
        return name("CURRENT") if callable(name) else None

    def _already_crowned(self, sid: str) -> bool:
        sp = Path(self.env.king_state_path)
        hist = KingState.from_json(sp.read_text()).history if sp.exists() else []
        return any(e.submission_id == sid for e in hist)
        return sp.exists() and any(
            e.submission_id == sid for e in KingState.from_json(sp.read_text()).history
        )

    def _close_crown(self, row: Row, store: SubmissionStore, st: dict) -> bool:
        pub = {k: v for k, v in st.items() if k not in ("submission_id", "hotkey", "stage", "updated_at")}
        pub.pop("crown_pending", None)
        archived = {"archived": True} if self.env.archive is not None else {}
        self.queue.update(row.id, archived=1 if self.env.archive is not None else 0)
        store.set_stage(row.id, row.hotkey, "done", **{**pub, "crowned": True, **archived})
        return True

    def _redecide(
        self, row: Row, ctx: PeriodContext, king: KingRun, rec: dict, store: SubmissionStore
    ) -> bool:
        import numpy as np

        from humanizer_eval.scoring.crown import crown_decision
        from humanizer_eval.scoring.metric import SubmissionRun

        if king.fingerprint != rec.get("fingerprint") or rec.get("h_id") not in (None, ctx.h_id):
            # never compare runs scored under other conditions (or against another scoring of H)
            rec = self._rescore_record(row, ctx, rec, store)  # same outputs, today's conditions (no pod)
        scores = {d: [float("inf") if x is None else x for x in v] for d, v in rec["scores"].items()}
        run = SubmissionRun(scores, np.asarray(rec["levels"], dtype=float), ctx.strata, ctx.input_ids)
        dec = crown_decision(king.to_run(ctx.strata), run, ctx.human_reference(), seed=ctx.crown_seed(row.id),
                             **self.env.cfg.crown_kw)  # fmt: skip
        first = (rec.get("crown") or {}).get("decided_at")  # a re-decision keeps its place in the order
        rec["crown"] = {**dec.__dict__, "reasons": list(dec.reasons), "king": king.submission_id,
                        "king_run": king.record_id, "decided_at": first or time.time()}  # fmt: skip
        store.save_record(row.id, rec)
        d = self.env.cfg.public_digits
        pub = {k: v for k, v in (store.status(row.id) or {}).items()
               if k not in ("submission_id", "hotkey", "stage", "updated_at", "crown_pending")}  # fmt: skip
        pub.update(S_king=round(dec.s_king, d), delta=round(dec.delta, d), lower_bound=round(dec.lower, d),
                   crowned=False, king_changed_during_run=True)  # fmt: skip
        if dec.crown:
            pub["crown_pending"] = True
        store.set_stage(row.id, row.hotkey, "archiving" if dec.crown else "done", **pub)
        log.info("%s re-decided against the new king %s: crown=%s", row.id, king.submission_id, dec.crown)
        return bool(dec.crown)

    def _rescore_record(self, row: Row, ctx: PeriodContext, rec: dict, store: SubmissionStore) -> dict:
        """The challenger scored again from its saved outputs under the current scoring conditions (a deploy
        changed them while its crown was pending), like the king is rescored. Its own record keeps what the
        miner saw; the rescore is stored under a separate id. Judges and detectors mostly hit the caches."""
        saved = store.load_run(row.id)
        if saved is None:
            raise KingUnavailable(f"{row.id}: no saved outputs to rescore under the current conditions")
        env = self.env
        sub = Submission(f"{row.id}-rescore", row.hotkey, name=row.name or "")
        cfg = replace(env.cfg, update_king=False)
        r = evaluate_submission(sub, ctx, runner=ReplayRunner(saved.outputs), llm=env.llm,
                                gate_factory=env.gate_factory, detectors=env.detectors, store=store, cfg=cfg,
                                watch=env.detector_watch)  # fmt: skip
        if r.status != "done":
            raise KingUnavailable(f"{row.id}: rescore under the current conditions did not finish")
        rs = (
            store.load_record(sub.submission_id) or {}
        )  # the rescore's own record: its S goes with its scores
        return {**rec, "scores": r.private["scores"], "levels": r.private["levels"],
                "score": rs.get("score", rec.get("score")), "gates": rs.get("gates", rec.get("gates")),
                "fingerprint": env.cfg.fingerprint, "h_id": ctx.h_id}  # fmt: skip

    def _finish_crown(
        self, row: Row, ctx: PeriodContext, king: KingRun, store: SubmissionStore
    ) -> bool | None:
        """Archive the winner's bundle from the pinned commit; the crown counts only if that works."""
        env = self.env
        pub = {k: v for k, v in (store.status(row.id) or {}).items()
               if k not in ("submission_id", "hotkey", "stage", "updated_at")}  # fmt: skip
        pub.pop("crown_pending", None)
        rec = store.load_record(row.id) or {}
        try:
            won = bool((rec.get("crown") or {}).get("crown"))  # the record's decision, written first
            if rec.get("period_id", ctx.period_id) != ctx.period_id or not won:
                raise RuntimeError("the period changed before the crown was confirmed")
            if env.archive is not None:
                env.archive.store(row.id, self._pod_url(row), row.bundle_sha256, env.max_bundle_bytes)
        except EvaluationPaused as e:  # no signed link for a private repo yet: archived on a later try
            log.warning("crown of %s waits: %s", row.id, e)
            return None
        except Exception as e:
            # only the miner's own doing revokes an earned, paid crown: the file changed or is gone, too big;
            # a download error, a 5xx or our disk: the crown waits and is tried again
            miners = ("sha256", "larger than allowed", "(404)", "(401)", "(403)", "(410)", "period changed")
            if not any(k in str(e) for k in miners):
                log.warning("crown of %s waits (archive: %s)", row.id, e)
                self._alert(f"crown of {row.id} waits: archiving failed ({e})", dedupe=f"archive-{row.id}")
                return None
            log.warning("crown of %s not awarded: %s", row.id, e)
            why = "the bundle could not be archived from the pinned commit; the crown was not awarded"
            store.set_stage(row.id, row.hotkey, "done", **{**pub, "crowned": False, "crown_revoked": why})
            self._alert(f"crown of {row.id} not awarded: {e}")
            return False
        scores = {d: [float("inf") if x is None else x for x in v] for d, v in rec["scores"].items()}
        # the fingerprint it was scored under: without it the next ensure_king would rescore the new
        # king (paid) and a parallel challenger's re-decision would refuse to compare
        # and the H it was scored against: after a mid-period detector rescore the next ensure_king rescores
        # the new king from his saved outputs (no pod) before anyone is compared with him
        run = KingRun(row.id, row.hotkey, ctx.input_ids, rec["levels"], scores,
                      fingerprint=rec.get("fingerprint"), h_id=rec.get("h_id"))  # fmt: skip
        apply_crown(
            run,
            rec["crown"]["delta"],
            king,
            reg_block=row.reg_block,
            king_state_path=env.king_state_path,
            king_run_path=env.king_run_path(ctx),
            key=env.key,
        )
        self._close_crown(row, store, {**pub})
        log.info("%s crowned; bundle archived", row.id)
        self._alert(f"new king {row.id} (hotkey {row.hotkey[:10]}...), delta S {rec['crown']['delta']:.4f}")
        return True

    def serve(self, stop: threading.Event, idle_s: float = 15.0) -> None:
        """Run up to ``concurrency`` evaluations at once, each on its own pod."""
        for sid in self.queue.recover():
            log.warning("resuming %s after a restart", sid)

        def loop() -> None:
            while not stop.is_set():
                try:
                    did = self.run_once()
                except Exception as e:
                    log.exception("worker step failed")
                    self._alert(f"worker step failed: {type(e).__name__}: {e}", dedupe="worker-step")
                    did = None
                if not did:
                    stop.wait(idle_s)

        if self.concurrency == 1:
            loop()
            return
        threads = [
            threading.Thread(target=loop, daemon=True, name=f"eval-{k}") for k in range(self.concurrency)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


class CurrentPeriod:
    """``<periods_dir>/CURRENT`` names the period being evaluated, ``NEXT`` the one being prepared (written by
    the rotation, period/lifecycle.py). Each context is loaded once, in this process only: H is scored here
    and nowhere else. Loading the next period does not block callers of the current one."""

    def __init__(
        self,
        periods_dir: Path,
        loader: Callable[[Path], PeriodContext],
        *,
        retry_s: float = 600.0,
        retry_max_s: float = 3600.0,
        now: Callable[[], float] = time.monotonic,
    ):
        self.periods_dir = Path(periods_dir)
        self.loader = loader
        self._cache: dict[str, PeriodContext] = {}
        self._loading: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        #: period id -> (retry at, wait, error): a failed load is not repeated before that time. A load
        #: scores H and the canaries (paid); the worker and every API request ask for the period.
        self._failed: dict[str, tuple[float, float, str]] = {}
        self.retry_s, self.retry_max_s, self._now = retry_s, retry_max_s, now

    def _name(self, f: str) -> str | None:
        p = self.periods_dir / f
        return (p.read_text().strip() or None) if p.exists() else None

    def _get(self, pid: str) -> PeriodContext:
        with self._lock:
            if pid in self._cache:
                return self._cache[pid]
            lk = self._loading.setdefault(pid, threading.Lock())
        with lk:
            with self._lock:
                if pid in self._cache:
                    return self._cache[pid]
                held = self._failed.get(pid)
            if held is not None and self._now() < held[0]:
                raise RuntimeError(held[2])
            try:
                ctx = self.loader(self.periods_dir / pid)
            except Exception as e:
                wait = min(2 * held[1], self.retry_max_s) if held is not None else self.retry_s
                msg = f"loading period {pid} failed: {type(e).__name__}: {e} (next try in {wait / 60:.0f}m)"
                with self._lock:
                    self._failed[pid] = (self._now() + wait, wait, msg)
                raise RuntimeError(msg) from e
            with self._lock:
                self._failed.pop(pid, None)
                self._cache[pid] = ctx
                keep = {pid, self._name("CURRENT"), self._name("NEXT")}
                for k in [k for k in self._cache if k not in keep]:
                    del self._cache[k]
            return ctx

    def __call__(self) -> PeriodContext:
        pid = self._name("CURRENT")
        if pid is None:
            raise RuntimeError(f"no current period in {self.periods_dir}")
        return self._get(pid)

    def upcoming(self) -> PeriodContext | None:
        nxt = self._name("NEXT")
        if nxt is None or nxt == self._name("CURRENT") or not (self.periods_dir / nxt).is_dir():
            return None
        return self._get(nxt)
