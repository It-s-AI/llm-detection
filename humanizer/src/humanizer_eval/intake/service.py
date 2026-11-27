"""What a miner can do (HTTP-free; ``intake/api.py`` maps it to routes).

    deposit()                           -> the hotkey's TAO deposit address (billing/tao.py): TAO sent
                                          there is bridged to Base, swapped to USDC, credited to the balance
    submit(...)                         -> quote for the current period in USDC; queued at once when the
                                          balance covers the hold, else the TAO to send for what is missing
    paid(sid) / status poll             -> the deposit is checked now; once the balance covers the hold, the
                                          hold opens and the submission joins the queue
    status(sid)                         -> state, place in the queue, the public result (D-58)
    cancel(sid)                         -> before it runs: the hold is released
    balance(), refund(key, to)          -> the hotkey's USDC; a refund goes back in TAO at the current rate to
                                          any address the miner names (refunds wait while a hold is open)

Rules that keep the queue fair and cheap to run:
* only hotkeys registered on the subnet (the share of a king that cannot receive weights would be lost);
* one submission per hotkey registration (D-66: a new submission needs a new registration), one active
  submission per hotkey, a cap on the waiting queue (a flood costs the flooder the holds);
* the bundle is a https URL with its sha256 and size (<= 50 GB, D-61); the pod downloads and checks it,
  our server never touches the bytes;
* order = payment time.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from humanizer_eval.billing.accounts import BillingError
from humanizer_eval.billing.prices import quote as price_quote
from humanizer_eval.billing.tao import SlippageTooHigh, usdc
from humanizer_eval.intake.queue import QueueError, Row, SubmissionQueue
from humanizer_eval.runner.bundle import MAX_BUNDLE_BYTES

_SHA = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9 ._-]{0,64}$")
#: what anyone may see of a finished submission (the public leaderboard): scores and aggregates only;
#: texts, judges' reasons and money stay private (D-58)
PUBLIC_RESULT = ("S", "crowned", "S_king", "delta", "lower_bound", "alpha", "levels", "gates", "judge_errors",
                 "detectors")  # fmt: skip
_PERIOD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$")
#: TAO sent within the quote's hour takes a few minutes to arrive as USDC (bridge, swap): a submission waits
#: this long past its window before it expires (the money stays on the balance either way)
LATE_GIVE_UP_S = 3600
log = logging.getLogger(__name__)


class NotFound(LookupError):
    pass


class Forbidden(PermissionError):
    pass


class Unavailable(RuntimeError):
    """A dependency we need to accept the request is down (HTTP 503)."""


class RateLimited(RuntimeError):
    """The evaluator's daily capacity is used up (HTTP 429)."""


@dataclass
class IntakeLimits:
    max_bundle_bytes: int = MAX_BUNDLE_BYTES  # D-61, the same limit the pod enforces
    max_waiting: int = 50
    max_url_len: int = 2048
    payment_poll_interval_s: float = 15.0  # status checks of an unpaid submission's deposit address
    paid_check_interval_s: float = 3.0  # the miner's "paid" command checks at most this often
    #: at most this many evaluations opened per 24 h, all miners together (None: no cap)
    max_evals_per_day: int | None = None
    #: False: the API is read-only (info, leaderboard, results, kings)
    submissions_open: bool = True
    closed_message: str = "submissions are closed during the test launch; they open at the production launch"
    allow_raw_urls: bool = False  # development only: a bundle at any https URL instead of HF + commitment


def check_bundle_url(url: str, max_len: int = 2048) -> str:
    """https, a DNS host name (no IP literals, no localhost, no credentials), default port."""
    if not isinstance(url, str) or len(url) > max_len or any(c.isspace() or ord(c) < 32 for c in url):
        raise ValueError("bundle_url: a https URL without spaces, at most 2048 characters")
    u = urlsplit(url)
    if u.scheme != "https" or not u.hostname:
        raise ValueError("bundle_url must be https")
    if u.username or u.password or "@" in u.netloc:
        raise ValueError("bundle_url must not carry credentials")
    if u.port not in (None, 443):
        raise ValueError("bundle_url must use the default https port")
    host = u.hostname.rstrip(".").lower()
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False
    if is_ip:
        raise ValueError("bundle_url must use a host name, not an IP address")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")) or "." not in host:
        raise ValueError("bundle_url must point to a public host")
    return url


SUBMITS_PER_HOUR = 20
PRIVATE_REFUSED = "no tokens: the repo may be private while you pay, but must be public when evaluated"


class IntakeService:
    def __init__(
        self,
        queue: SubmissionQueue,
        billing: Any,  # billing.BillingService
        period: Callable[[], Any],  # -> PeriodContext of the current period
        store_for: Callable[[str], Any],  # period_id -> SubmissionStore
        registration: Callable[[str], int | None],  # hotkey -> its registration block, None if not registered
        *,
        limits: IntakeLimits | None = None,
        now: Callable[[], float] = time.time,
        commitment: Callable[[str], tuple[str, int] | None] | None = None,  # hotkey -> (data, block)
        hub: Any = None,  # intake.hf.HubClient
        secret_key: bytes | None = None,  # encrypts miners' read tokens at rest
        about: dict[str, Any]
        | None = None,  # shown in /v1/info: {"mode": "test", "network": ..., "netuid": ...}
        uid: Callable[[str], int | None] | None = None,  # hotkey -> UID (the public queue shows it)
    ):
        self.uid = uid
        self.queue = queue
        self.billing = billing
        self.period = period
        self.store_for = store_for
        self.registration = registration
        self.limits = limits or IntakeLimits()
        self._now = now
        self._enqueue_lock = threading.Lock()
        self._submits: dict[str, list[float]] = {}  # hotkey -> recent submit requests
        self._bundle_infos: dict[tuple[str, str], tuple[float, Any]] = {}  # (repo, commit) -> (at, HFBundle)
        self._polled: dict[str, float] = {}
        self.commitment = commitment
        self.hub = hub
        self.secret_key = secret_key
        self.about = dict(about or {})

    # --- public ------------------------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        ctx = self.period()
        dets = self._card_detectors(ctx)
        est = price_quote(self.billing.card, ctx.input_words(), dets)
        counts = self.queue.counts()
        return {
            "period_id": ctx.period_id,
            "n_inputs": ctx.n,
            "crown_detectors": list(ctx.detectors),
            "price_card": {"version": self.billing.card.version, "sha256": self.billing.card.sha256},
            "payment_methods": ["tao"],
            "currency": "usdc",
            "quote_total_usdc": usdc(est.total_cents),
            "queue": {
                "waiting": counts.get("queued", 0) + counts.get("paused", 0),
                "running": counts.get("running", 0),
            },
            "limits": {
                "max_bundle_bytes": self.limits.max_bundle_bytes,
                "one_active_submission_per_hotkey": True,
                "submissions_open": self.limits.submissions_open,
                "max_evals_per_day": self.limits.max_evals_per_day,
                "evals_last_24h": self._evals_last_24h(),
            },
            **self.about,
        }

    _COUNTED = ("queued", "running", "paused", "done")  # paid work only: unpaid quotes cost nothing to make

    def _evals_last_24h(self) -> int:
        return self.queue.created_since(self._now() - 86400, self._COUNTED)

    def _check_capacity(self) -> None:
        cap = self.limits.max_evals_per_day
        if cap is not None and self._evals_last_24h() >= cap:
            raise RateLimited(f"the evaluator's daily capacity ({cap} evaluations per 24 h) is used up; "
                              "try again later (an unpaid link frees its place when it expires)")  # fmt: skip

    # --- submit ------------------------------------------------------------------------------------
    def _card_detectors(self, ctx: Any) -> list[str]:
        return [d for d in ctx.detectors if d in self.billing.card.detectors]

    def _registration(self, hotkey: str) -> int:
        try:
            reg_block = self.registration(hotkey)
        except Exception as e:
            raise Unavailable("subnet registration check unavailable; try again later") from e
        if reg_block is None:
            raise Forbidden("hotkey is not registered on the subnet")
        return int(reg_block)

    @staticmethod
    def _check_name(name: str) -> None:
        if not isinstance(name, str) or not _NAME.match(name):
            raise ValueError("name: up to 64 of [A-Za-z0-9 ._-]")

    def submit_hf(
        self, hotkey: str, *, commitment: str | None = None, hf_token: str | None = None, name: str = ""
    ) -> dict[str, Any]:
        """A bundle pinned by the hotkey's on-chain commitment ``<repo>@<commit>`` (intake/hf.py).

        ``commitment``, when given, must equal what the chain shows (it tells a stale snapshot from a
        typo). The repo may still be private when the miner submits and pays (so nobody can copy it in
        the meantime); it must be public when the evaluation starts: the bundle is read then (worker.py).
        ``hf_token`` is refused (owner decision 2026-11-26).
        """
        from humanizer_eval.intake.hf import HubNotReadable, parse_commitment

        self._check_name(name)
        if hf_token:  # owner decision 2026-11-26: public repos only (the pod downloads without a token)
            raise ValueError(PRIVATE_REFUSED)
        if self.commitment is None or self.hub is None:
            raise Unavailable("Hugging Face submissions are not configured here")
        reg_block = self._registration(hotkey)
        # cheap refusals first: what follows calls Hugging Face (up to a 50 MB read) and writes a quote
        self._check_open()
        self._check_capacity()
        self.queue.precheck(hotkey, reg_block, self.limits.max_waiting)
        now = self._now()
        recent = [t for t in self._submits.get(hotkey, []) if now - t < 3600]
        if len(recent) >= SUBMITS_PER_HOUR:  # submit + cancel in a loop: each may read Hugging Face
            raise QueueError(f"at most {SUBMITS_PER_HOUR} submission requests per hour per hotkey")
        self._submits[hotkey] = [*recent, now]
        try:
            c = self.commitment(hotkey)
        except Exception as e:
            raise Unavailable("chain commitments unavailable; try again later") from e
        if c is None:
            raise QueueError("no commitment of this hotkey on chain yet (it may take a few minutes to show)")
        data, block = c
        if commitment is not None and commitment != data:
            raise QueueError(f"the chain shows {data!r}; wait a few minutes if you have just committed")
        if block < reg_block:
            raise QueueError("the commitment predates this registration; commit again from it")
        repo, rev = parse_commitment(data)
        try:
            hit = self._bundle_infos.get((repo, rev))
            info = hit[1] if hit and now - hit[0] < 3600 else self.hub.bundle_info(repo, rev)
            self._bundle_infos[(repo, rev)] = (now, info)  # a pinned commit's bundle never changes
        except HubNotReadable:
            info = None  # private for now: read when the evaluation starts
        if info is not None and not 0 < info.size <= self.limits.max_bundle_bytes:
            raise ValueError(f"bundle size {info.size} outside 1..{self.limits.max_bundle_bytes} bytes")
        sha, size = (info.sha256, info.size) if info is not None else ("", None)
        bundle: dict[str, Any] = {"repo": repo, "commit": rev, "sha256": sha or None, "bytes": size}
        if info is None or info.private or info.gated:
            bundle["note"] = "make the repo public before the evaluation starts: a private one is refused"
        return self._open(
            hotkey,
            reg_block,
            bundle_url=self.hub.resolve_url(repo, rev),
            bundle_sha256=sha,
            bundle_bytes=size,
            name=name,
            hf_repo=repo,
            hf_revision=rev,
            commit_block=block,
            extra={"bundle": bundle},
        )

    def submit(
        self, hotkey: str, *, bundle_url: str, bundle_sha256: str, bundle_bytes: int, name: str = ""
    ) -> dict[str, Any]:
        """Development only (``allow_raw_urls``): a bundle at any https URL with its sha256."""
        if not self.limits.allow_raw_urls:
            raise ValueError("submit a Hugging Face commitment (bundle_url is for development only)")
        check_bundle_url(bundle_url, self.limits.max_url_len)
        if not isinstance(bundle_sha256, str) or not _SHA.match(bundle_sha256):
            raise ValueError("bundle_sha256: 64 lowercase hex characters")
        if not isinstance(bundle_bytes, int) or not 0 < bundle_bytes <= self.limits.max_bundle_bytes:
            raise ValueError(f"bundle_bytes: 1..{self.limits.max_bundle_bytes}")
        self._check_name(name)
        reg_block = self._registration(hotkey)
        return self._open(hotkey, reg_block, bundle_url=bundle_url, bundle_sha256=bundle_sha256,
                          bundle_bytes=bundle_bytes, name=name)  # fmt: skip

    def _check_open(self) -> None:
        if not self.limits.submissions_open:
            raise Forbidden(self.limits.closed_message)

    def _open(self, hotkey: str, reg_block: int, *, extra: dict[str, Any] | None = None, **cols: Any) -> dict:
        self._check_open()
        self._check_capacity()
        ctx = self.period()
        q = self.billing.make_quote(hotkey, ctx.input_words(), detectors=self._card_detectors(ctx))
        sid = f"sub-{uuid.uuid4().hex[:16]}"
        self.queue.create(
            sid,
            hotkey,
            period_id=ctx.period_id,
            quote_id=q["id"],
            hold_cents=q["hold_cents"],
            expires_ts=q["expires_ts"],
            max_waiting=self.limits.max_waiting,
            reg_block=reg_block,
            **cols,
        )
        out: dict[str, Any] = {
            "submission_id": sid,
            "period_id": ctx.period_id,
            "quote": {
                "id": q["id"],
                "total_usdc": usdc(q["total_cents"]),
                "hold_usdc": usdc(q["hold_cents"]),
                "items": q["items"],
                "expires_ts": q["expires_ts"],
            },
            **(extra or {}),
        }
        row = self._try_enqueue(self.queue.get(sid))  # type: ignore[arg-type]
        if row.state == "queued":
            out.update(state="queued", queue_position=self.queue.position(sid), payment=None)
            return out
        try:
            pay = self.billing.payment(q["id"])
        except Exception:
            try:  # only if still unpaid: the payments loop may have queued it meanwhile (then it stands)
                now = self._now()
                self.queue.update(sid, expect=("awaiting_payment",), state="cancelled", finished_ts=now,
                                  note="no payment address")  # fmt: skip
            except QueueError:
                return {**out, "state": self.queue.get(sid).state, "payment": None}  # type: ignore[union-attr]
            raise
        out.update(state="awaiting_payment", payment={**pay, "then": f"POST /v1/submissions/{sid}/paid"})
        return out

    def _try_enqueue(self, row: Row) -> Row:
        """awaiting_payment -> queued once the balance covers the hold; -> expired past the window (plus
        ``LATE_GIVE_UP_S`` while TAO sent in time is still on its way)."""
        with self._enqueue_lock:  # the status poll and the worker loop must not both open and release it
            row = self.queue.get(row.id) or row
            return self._try_enqueue_locked(row)

    def _try_enqueue_locked(self, row: Row) -> Row:
        if row.state != "awaiting_payment":
            return row
        h = self.billing.accounts.hold(row.id)
        if (
            h is not None and h.status == "open"
        ):  # paid and held, the enqueue was cut off (a crash): finish it
            try:
                return self.queue.enqueue(row.id)
            except QueueError:
                return self.queue.get(row.id)  # type: ignore[return-value]
        now = self._now()
        late = row.expires_ts is not None and now > row.expires_ts
        if self.billing.amount_due(row.quote_id) > 0:
            # never while the miner's TAO is on its way (a slow or paused bridge is not the miner's doing)
            if late and not self._converting(row.hotkey):
                return self._expire(row)
            return row
        try:
            self.billing.start(row.quote_id, row.id, paid_in_time=late)
        except BillingError:
            return row  # spent elsewhere in the meantime (e.g. a refund)
        try:
            return self.queue.enqueue(row.id)
        except QueueError:  # cancelled or expired concurrently: give the money back (never for a live row)
            cur = self.queue.get(row.id)
            if cur is not None and cur.state in ("cancelled", "expired"):
                self.billing.fail(row.id)
            return cur  # type: ignore[return-value]

    def _release_orphan_holds(self) -> None:
        """A hold still open for a row that is over without a run (cancelled, expired, failed): a crash
        between two steps left it; it would lock the miner's money and block every refund."""
        try:
            ids = self.billing.accounts.open_hold_ids()
        except Exception as e:
            log.warning("open holds not listed: %s", e)
            return
        for sid in ids:
            row = self.queue.get(sid)
            if row is not None and row.state in ("cancelled", "expired", "failed"):
                log.warning("releasing the orphan hold of %s (%s)", sid, row.state)
                self.billing.fail(sid)

    def _converting(self, hotkey: str) -> bool:
        tao = getattr(self.billing, "tao", None)
        return tao is not None and tao.incoming(hotkey)

    def _expire(self, row: Row) -> Row:
        try:
            out = self.queue.update(row.id, expect=("awaiting_payment",), state="expired", payment_url=None)
        except QueueError:
            return self.queue.get(row.id)  # type: ignore[return-value]
        h = self.billing.accounts.hold(row.id)  # opened, but the enqueue never happened: give it back
        if h is not None and h.status == "open":
            self.billing.fail(row.id)
        return out

    def enqueue_ready(self) -> list[str]:
        """Called by the worker loop: deposits converted and refunds moved on (billing housekeeping), then
        every submission whose balance now covers its hold joins the queue, in payment order."""
        try:
            self.billing.housekeeping()
        except Exception as e:
            log.warning("payments housekeeping failed: %s", e)
        self._release_orphan_holds()
        moved = []
        for row in self.queue.by_state("awaiting_payment"):
            try:  # one row's trouble (an RPC error) must not stall everyone's queue
                if self._try_enqueue(row).state != "awaiting_payment":
                    moved.append(row.id)
            except Exception as e:
                log.warning("submission %s not moved on: %s", row.id, e)
        return moved

    def _collect(self, row: Row, min_interval: float) -> dict[str, Any] | None:
        """Check the hotkey's deposit address now (rate-limited); None if not asked or it failed."""
        if row.state != "awaiting_payment":
            return None
        now = self._now()
        if now - self._polled.get(row.hotkey, 0.0) < min_interval:
            return None
        self._polled[row.hotkey] = now
        try:
            return {"credited_cents": self.billing.collect(row.hotkey)}
        except SlippageTooHigh as e:
            return {"credited_cents": 0, "waiting": str(e)}
        except Exception as e:
            log.warning("deposit of %s not checked: %s", row.hotkey, e)
            return None

    def public_queue(self) -> dict[str, Any]:
        """The paid submissions waiting or running, in queue order: who (UID, hotkey, name) and when. Never
        the bundle (a repo stays unknown until its evaluation is over) nor money."""
        out = []
        for r in self.queue.by_state("running", "queued", "paused"):
            if r.owner_job:
                continue
            try:
                uid = self.uid(r.hotkey) if self.uid else None
            except Exception:  # a stale metagraph: the queue is still shown
                uid = None
            out.append({"submission_id": r.id, "uid": uid, "hotkey": r.hotkey, "name": r.name or "",
                        "state": "running" if r.state == "running" else "waiting", "period_id": r.period_id,
                        "enqueued_ts": r.enqueued_ts, "started_ts": r.started_ts})  # fmt: skip
        out.sort(key=lambda x: (x["state"] != "running", x["enqueued_ts"] or 0.0))
        return {"queue": out}

    def results(self, period_id: str | None = None, limit: int = 500) -> dict[str, Any]:
        """Finished submissions of a period (default: the current one), newest first: public scores only."""
        pid = period_id or self.period().period_id
        if not _PERIOD_ID.match(pid):
            raise ValueError("bad period id")
        rows = [r for r in self.queue.by_state("done") if r.period_id == pid and not r.owner_job]
        rows.sort(key=lambda r: r.finished_ts or 0.0, reverse=True)
        store = self.store_for(pid)
        out = []
        for r in rows[: max(1, min(limit, 500))]:
            pub = store.status(r.id) or {}
            scores = {k: pub[k] for k in PUBLIC_RESULT if k in pub}
            try:
                uid = self.uid(r.hotkey) if self.uid else None
            except Exception:
                uid = None
            out.append({"submission_id": r.id, "uid": uid, "hotkey": r.hotkey, "name": r.name or "",
                        "finished_ts": r.finished_ts, **scores})  # fmt: skip
        return {"period_id": pid, "results": out}

    # --- per submission ------------------------------------------------------------------------------
    def _own(self, hotkey: str, sid: str) -> Row:
        row = self.queue.get(sid)
        if row is None or row.hotkey != hotkey or row.owner_job:
            raise NotFound("no such submission for this hotkey")
        return row

    def status(self, hotkey: str, sid: str) -> dict[str, Any]:
        row = self._own(hotkey, sid)
        if row.state == "awaiting_payment":
            self._collect(row, self.limits.payment_poll_interval_s)
            row = self._try_enqueue(self.queue.get(sid))  # type: ignore[arg-type]
        out: dict[str, Any] = {
            "submission_id": sid,
            "state": row.state,
            "period_id": row.period_id,
            "created_ts": row.created_ts,
            "enqueued_ts": row.enqueued_ts,
            "started_ts": row.started_ts,
            "finished_ts": row.finished_ts,
            "hold_usdc": None if row.hold_cents is None else usdc(row.hold_cents),
        }
        if row.hf_repo:
            out["bundle"] = {
                "repo": row.hf_repo,
                "commit": row.hf_revision,
                "sha256": row.bundle_sha256,
                "commitment_block": row.commit_block,
                "archived": bool(row.archived),
            }
        if row.state in ("queued", "paused"):
            out["queue_position"] = self.queue.position(sid)
        if row.state == "paused":
            out["retry_at"] = row.retry_at
        if row.state == "awaiting_payment":
            out["quote_expires_ts"] = row.expires_ts
            try:
                out["payment"] = self.billing.payment(row.quote_id)
                out["payment"]["arriving_usdc"] = usdc(self.billing.tao.pending_cents(hotkey))
            except Exception as e:  # no rate now: the address still works
                log.debug("payment info: %s", e)
                out["payment"] = {"due_usdc": usdc(self.billing.amount_due(row.quote_id))}
        if row.period_id:
            pub = self.store_for(row.period_id).status(sid)
            if pub is not None:
                out["evaluation"] = {k: v for k, v in pub.items() if k not in ("hotkey", "submission_id")}
        return out

    def paid(self, hotkey: str, sid: str) -> dict[str, Any]:
        """The miner says the TAO is sent: check the deposit now and, once covered, join the queue."""
        row = self._own(hotkey, sid)
        chk = None
        if row.state == "awaiting_payment":
            chk = self._collect(row, self.limits.paid_check_interval_s)
            self._try_enqueue(self.queue.get(sid))  # type: ignore[arg-type]
        out = self.status(hotkey, sid)
        st = out["state"]
        pay = out.get("payment") or {}
        if st == "awaiting_payment":
            if chk is not None and chk.get("waiting"):
                out["message"] = "your TAO arrived; the swap waits for better prices and runs by itself"
            elif float(pay.get("arriving_usdc") or 0) > 0:
                out["message"] = ("your TAO is on its way to USDC (bridge and swap, a few minutes); "
                                  "the submission joins the queue by itself once it is credited")  # fmt: skip
            else:
                out["message"] = (f"no TAO yet: send {pay.get('amount_tao')} TAO to {pay.get('address')} "
                                  "before "
                                  "quote_expires_ts, then run paid")  # fmt: skip
        elif st in ("queued", "paused"):
            out["message"] = f"payment confirmed; in the queue at position {out.get('queue_position')}"
        elif st == "expired":
            out["message"] = ("the payment window closed; whatever arrived is on your balance: submit again "
                              "(the registration is not used) or ask for a refund")  # fmt: skip
        else:
            out["message"] = f"state {st}"
        return out

    def statement(self, hotkey: str, sid: str) -> dict[str, Any]:
        row = self._own(hotkey, sid)
        st = None
        if row.period_id:
            p = self.store_for(row.period_id).root / sid / "statement.json"
            if p.exists():
                st = json.loads(p.read_text())
        if st is None:
            raise NotFound("no statement yet")
        return st

    def cancel(self, hotkey: str, sid: str) -> dict[str, Any]:
        self._own(hotkey, sid)
        row = self.queue.get(sid)
        if row is not None and (row.attempts > 0 or row.started_ts is not None):
            raise QueueError(
                "this evaluation has already started (it resumes after a restart); it cannot be cancelled"
            )
        self.queue.update(
            sid, expect=("awaiting_payment", "queued"), state="cancelled", finished_ts=self._now()
        )
        released = 0
        h = self.billing.accounts.hold(sid)
        if h is not None and h.status == "open":
            self.billing.fail(sid)
            released = h.amount_cents
        return {"submission_id": sid, "state": "cancelled", "released_usdc": usdc(released)}

    # --- money ---------------------------------------------------------------------------------------
    @staticmethod
    def _dest(to: Any) -> str:
        from humanizer_eval.billing.tao import ss58_pubkey

        try:
            ss58_pubkey(to if isinstance(to, str) else "")
        except Exception as e:
            raise ValueError("to: the ss58 address the TAO goes to (any wallet)") from e
        return to

    def deposit(self, hotkey: str) -> dict[str, Any]:
        """The hotkey's TAO deposit address: TAO sent there becomes USDC on the balance."""
        if self.registration(hotkey) is None and self.billing.accounts.balance(hotkey) <= 0:
            raise Forbidden("hotkey is not registered on the subnet")
        dep = self.billing.deposit(hotkey)
        price = None
        try:
            price = round(self.billing.tao.route.price(), 2)
        except Exception as e:
            log.debug("no TAO price: %s", e)
        return {**dep, "price_usdc_per_tao": price, **self.balance(hotkey)}

    def balance(self, hotkey: str) -> dict[str, Any]:
        acc = self.billing.accounts
        out = {"hotkey": hotkey, "currency": "usdc", "balance_usdc": usdc(acc.balance(hotkey)),
               "available_usdc": usdc(acc.available(hotkey))}  # fmt: skip
        tao = getattr(self.billing, "tao", None)
        if tao is not None:
            out["arriving_usdc"] = usdc(tao.pending_cents(hotkey))
            out["deposits"] = tao.inbound(hotkey, limit=5)
        return out

    @staticmethod
    def _cents(amount_usdc: Any) -> int | None:
        if amount_usdc is None:
            return None
        try:
            cents = round(float(amount_usdc) * 100)
        except (TypeError, ValueError) as e:
            raise ValueError("amount_usdc: a positive number of USDC, e.g. 25.50") from e
        if cents <= 0:
            raise ValueError("amount_usdc must be positive")
        return cents

    def refund_quote(self, hotkey: str, to: Any, amount_usdc: Any = None) -> dict[str, Any]:
        """What a refund would send now: the TAO estimate after the cost of the way back."""
        return self.billing.refund_quote(hotkey, self._dest(to), self._cents(amount_usdc))

    def refund(self, hotkey: str, key: str, to: Any, amount_usdc: Any = None) -> dict[str, Any]:
        """USDC of the balance (default: all available) back in TAO, at the current rate, to ``to``: any ss58
        address the miner names (it also works after the hotkey was deregistered). The bridge and swap cost
        come out of the amount; the TAO sent is reported."""
        if not re.match(r"^[A-Za-z0-9_-]{8,64}$", key or ""):
            raise ValueError("refund key: 8-64 of [A-Za-z0-9_-] (reuse it to retry safely)")
        st = self.billing.refund(hotkey, f"{hotkey}:{key}", self._dest(to), self._cents(amount_usdc))
        return {"refund": st, **self.balance(hotkey)}

    def refund_status(self, hotkey: str, key: str) -> dict[str, Any]:
        st = self.billing.tao.refund_state(f"{hotkey}:{key}") if getattr(self.billing, "tao", None) else None
        if st is None:
            raise NotFound("no such refund")
        return st
