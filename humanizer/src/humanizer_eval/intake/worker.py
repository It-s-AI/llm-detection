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

import json
import logging
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
        max_pauses: int = 8,
        pause_base_s: float = 900.0,
        pause_max_s: float = 6 * 3600.0,
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
        self._king_lock = threading.Lock()  # one king re-run at a time
        self._crown_lock = (
            threading.Lock()
        )  # crowns are decided one at a time, against the king of that moment

    def _alert(self, text: str, dedupe: str | None = None) -> None:
        if self.alerter is not None:
            self.alerter.send(text, dedupe=dedupe)

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
        if kr is not None and kr.submission_id == want and kr.fingerprint == env.cfg.fingerprint:
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
            sid = cur.submission_id if cur else BASELINE_ID
            hk = cur.hotkey if cur else OWNER_HOTKEY
            runner: Runner = ReplayRunner([it["text"] for it in ctx.items])
            sub = Submission(sid, hk, name="identity")
        else:
            row = self.queue.get(cur.submission_id)
            if row is None:
                raise KingUnavailable(f"the king's bundle ({cur.submission_id}) is not in the queue records")
            sid, hk = cur.submission_id, cur.hotkey
            # its own record in this period stays as the miner saw it; the owner's run has its own id
            sub = Submission(f"{sid}-kingrun", hk, bundle_url=row.bundle_url, bundle_sha256=row.bundle_sha256,
                             name=row.name or "", reg_block=row.reg_block)  # fmt: skip
            # evaluated in this very period (or rescored): same seed, same outputs, no pod
            saved = store.load_run(f"{sid}-kingrun") or store.load_run(sid)
            if saved is None and env.archive is not None and env.archive.has(sid):
                sub = replace(sub, bundle=env.archive.path(sid), bundle_url=None, bundle_sha256=None)
            elif saved is None and row.secret is None and row.hf_repo and env.archive is not None:
                raise KingUnavailable(f"the king's bundle ({sid}) is not in the archive")
            runner = ReplayRunner(saved.outputs) if saved is not None else env.runner_for(sub)
        try:
            r = evaluate_submission(sub, ctx, runner=runner, **common)
        except EvaluationPaused as e:
            raise KingUnavailable(f"king re-run paused: {e}") from e
        if r.status != "done":
            raise KingUnavailable(f"king re-run failed: {r.private.get('run_reason')}")
        scores = {d: [float("inf") if x is None else x for x in v] for d, v in r.private["scores"].items()}
        kr = KingRun(sid, hk, ctx.input_ids, r.private["levels"], scores, fingerprint=env.cfg.fingerprint)
        kr.save(path, env.key)
        log.info("king %s scored on %s: S=%s", sid, ctx.period_id, r.public.get("S"))
        return kr

    def prepare_next(self, ctx: PeriodContext) -> KingRun | None:
        """Get the upcoming period ready before the switch: its context is loaded (the caller did that) and
        the king's run on its inputs exists. ``<period dir>/READY`` tells the rotation it can switch."""
        try:
            with self._king_lock:
                kr = self.ensure_king(ctx)
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
    def _give_up(self, row: Row, store: SubmissionStore, reason: str) -> None:
        h = self.billing.accounts.hold(row.id) if self.billing is not None else None
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
            with self._king_lock:
                king = self.ensure_king(ctx)
        except KingUnavailable as e:
            log.warning("queue held: %s", e)
            self._alert(f"queue held: {e}", dedupe="king-unavailable")
            return None
        row = self.queue.claim(max_running=self.concurrency)
        if row is None:
            return None
        store = self.env.store_for(ctx.period_id)
        self.queue.update(row.id, period_id=ctx.period_id)
        h = self.billing.accounts.hold(row.id) if self.billing is not None else None
        if h is not None and h.status == "settled":  # settled before a crash: the result stands
            crowned = None
            if (store.status(row.id) or {}).get("stage") == "archiving":
                crowned = self._settle_crown(row, ctx, king, store)
            self._close(row, "done", note="recovered after settlement")
            return {"submission_id": row.id, "state": "done", "crowned": crowned}
        # the hotkey must still hold the registration it submitted from: after a long queue or long pauses
        # (up to ~26 h of backoff) a challenger past its 24 h immunity may have been pruned, and a crown
        # would not count; nothing is run or charged then
        reg = self.env.is_registered
        if (
            reg is not None
            and not row.owner_job
            and row.reg_block is not None
            and not reg(row.hotkey, row.reg_block)
        ):
            self._give_up(
                row, store, "the hotkey lost its registration before the evaluation; nothing was charged"
            )
            return {"submission_id": row.id, "state": "failed"}
        sub = Submission(row.id, row.hotkey, bundle_url=self._pod_url(row), bundle_sha256=row.bundle_sha256,
                         name=row.name or "", reg_block=row.reg_block)  # fmt: skip
        env = self.env
        cfg = replace(env.cfg, defer_crown=True)  # the crown is settled below, under the crown lock
        try:
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
        except EvaluationPaused as e:
            pauses = row.pauses + 1
            if pauses > self.max_pauses:
                self._give_up(
                    row, store, "evaluation services were unavailable too long; nothing was charged"
                )
                return {"submission_id": row.id, "state": "failed"}
            wait = min(self.pause_max_s, self.pause_base_s * 2 ** (pauses - 1))
            self.queue.update(row.id, state="paused", pauses=pauses, retry_at=self._now() + wait)
            self.queue.note(row.id, paused=str(e))
            if isinstance(
                e, ConditionsChanged
            ):  # needs the owner: rescore H and the king (rescore_period.py)
                self._alert(f"scoring conditions changed, evaluations wait: {e.reason}", dedupe="conditions")
            elif pauses >= 3:
                self._alert(f"evaluation {row.id} paused {pauses} times: {e}", dedupe=f"paused-{row.id}")
            return {"submission_id": row.id, "state": "paused"}
        except InsufficientBalance:  # no hold: should not happen for a queued row
            self._give_up(row, store, "payment not found; nothing was charged")
            return {"submission_id": row.id, "state": "failed"}
        except Exception as e:
            log.exception("evaluation of %s crashed", row.id)
            self._alert(f"evaluation {row.id} crashed ({type(e).__name__}), attempt {row.attempts}",
                        dedupe=f"crash-{row.id}-{row.attempts}")  # fmt: skip
            if row.attempts >= self.max_attempts:
                self._give_up(row, store, "evaluation failed on our side; nothing was charged")
                return {"submission_id": row.id, "state": "failed"}
            self.queue.update(row.id, state="paused", retry_at=self._now() + self.retry_s)
            self.queue.note(row.id, error=type(e).__name__)
            return {"submission_id": row.id, "state": "paused"}
        crowned = self._settle_crown(row, ctx, king, store) if res.status == "done" else False
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
        try:
            return self.env.hub.download_url(row.hf_repo, row.hf_revision, self._token(row))
        except Exception as e:  # the pod then fails on the download: a bundle error (miner side)
            log.warning("no download link for %s: %s", row.id, e)
            return row.bundle_url

    def _close(self, row: Row, state: str, note: str | None = None) -> None:
        cols: dict[str, Any] = {"state": state, "finished_ts": self._now(), "secret": None}
        if note:
            cols["note"] = note
        self.queue.update(row.id, **cols)

    def _settle_crown(
        self, row: Row, ctx: PeriodContext, king_at_start: KingRun, store: SubmissionStore
    ) -> bool:
        """Decide the crown against the king of this moment (one decision at a time). With evaluations in
        parallel another challenger may have been crowned while this one ran: the decision is then made
        again against the new king from the saved per-text scores (same texts, no new pod, D-11)."""
        with self._crown_lock:
            rec = store.load_record(row.id) or {}
            current = KingRun.load(self.env.king_run_path(ctx), self.env.key) or king_at_start
            won = bool((store.status(row.id) or {}).get("crown_pending"))
            if current.submission_id != king_at_start.submission_id and rec.get("scores") is not None:
                won = self._redecide(row, ctx, current, rec, store)
            if not won:
                return False
            return self._finish_crown(row, ctx, current, store)

    def _redecide(
        self, row: Row, ctx: PeriodContext, king: KingRun, rec: dict, store: SubmissionStore
    ) -> bool:
        import numpy as np

        from humanizer_eval.scoring.crown import crown_decision
        from humanizer_eval.scoring.metric import SubmissionRun

        if king.fingerprint != rec.get("fingerprint"):  # never compare runs scored under other conditions
            raise RuntimeError(f"{row.id}: the king's run and the record differ in scoring conditions")
        scores = {d: [float("inf") if x is None else x for x in v] for d, v in rec["scores"].items()}
        run = SubmissionRun(scores, np.asarray(rec["levels"], dtype=float), ctx.strata, ctx.input_ids)
        dec = crown_decision(king.to_run(ctx.strata), run, ctx.human_reference(), seed=ctx.crown_seed(row.id),
                             **self.env.cfg.crown_kw)  # fmt: skip
        rec["crown"] = {**dec.__dict__, "reasons": list(dec.reasons), "king": king.submission_id}
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

    def _finish_crown(self, row: Row, ctx: PeriodContext, king: KingRun, store: SubmissionStore) -> bool:
        """Archive the winner's bundle from the pinned commit; the crown counts only if that works."""
        env = self.env
        pub = {k: v for k, v in (store.status(row.id) or {}).items()
               if k not in ("submission_id", "hotkey", "stage", "updated_at")}  # fmt: skip
        pub.pop("crown_pending", None)
        rec = store.load_record(row.id) or {}
        try:
            if rec.get("period_id", ctx.period_id) != ctx.period_id or not rec.get("crown"):
                raise RuntimeError("the period changed before the crown was confirmed")
            if env.archive is not None:
                env.archive.store(row.id, self._pod_url(row), row.bundle_sha256, env.max_bundle_bytes)
        except Exception as e:
            log.warning("crown of %s not awarded: %s", row.id, e)
            why = "the bundle could not be archived from the pinned commit; the crown was not awarded"
            store.set_stage(row.id, row.hotkey, "done", **{**pub, "crowned": False, "crown_revoked": why})
            self._alert(f"crown of {row.id} not awarded: {e}")
            return False
        scores = {d: [float("inf") if x is None else x for x in v] for d, v in rec["scores"].items()}
        run = KingRun(row.id, row.hotkey, ctx.input_ids, rec["levels"], scores)
        apply_crown(
            run,
            rec["crown"]["delta"],
            king,
            reg_block=row.reg_block,
            king_state_path=env.king_state_path,
            king_run_path=env.king_run_path(ctx),
            key=env.key,
        )
        self.queue.update(row.id, archived=1 if env.archive is not None else 0)
        archived = {"archived": True} if env.archive is not None else {}
        store.set_stage(row.id, row.hotkey, "done", **{**pub, "crowned": True, **archived})
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

    def __init__(self, periods_dir: Path, loader: Callable[[Path], PeriodContext]):
        self.periods_dir = Path(periods_dir)
        self.loader = loader
        self._cache: dict[str, PeriodContext] = {}
        self._loading: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

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
            ctx = self.loader(self.periods_dir / pid)
            with self._lock:
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
