"""What a miner can do (HTTP-free; ``intake/api.py`` maps it to routes).

    submit(bundle_url, sha256, bytes)   -> quote for the current period; Stripe Checkout for what the
                                          balance lacks; queued at once when the balance covers the hold
    paid(sid) / status poll / webhook   -> Stripe says the Checkout link was paid: the hold opens and the
                                          submission joins the queue (no web site needed: the miner gets
                                          the link from the submit command and confirms with "paid")
    status(sid)                         -> state, place in the queue, the public result (D-58)
    cancel(sid)                         -> before it runs: the hold is released
    balance(), refund(key)              -> the hotkey's money (refunds wait while a hold is open)

Rules that keep the queue fair and cheap to run:
* only hotkeys registered on the subnet (the share of a king that cannot receive weights would be lost);
* one submission per hotkey registration (D-66: a new submission needs a new registration), one active
  submission per hotkey, a cap on the waiting queue (a flood costs the flooder the holds);
* the bundle is a https URL with its sha256 and size (<= 50 GB, D-61); the pod downloads and checks it,
  our server never touches the bytes;
* order = payment time; one evaluation at a time (one pod, D-65).
"""

from __future__ import annotations

import ipaddress
import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from humanizer_eval.billing.accounts import BillingError
from humanizer_eval.billing.prices import quote as price_quote
from humanizer_eval.intake.queue import QueueError, Row, SubmissionQueue
from humanizer_eval.runner.bundle import MAX_BUNDLE_BYTES

_SHA = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[A-Za-z0-9 ._-]{0,64}$")
#: what anyone may see of a finished submission (the public leaderboard): scores and aggregates only;
#: texts, judges' reasons and money stay private (D-58)
PUBLIC_RESULT = ("S", "crowned", "S_king", "delta", "lower_bound", "alpha", "levels", "gates", "judge_errors",
                 "detectors")  # fmt: skip
_PERIOD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,40}$")
LATE_GIVE_UP_S = 24 * 3600  # a Checkout link lives at most 24 h: past that, nothing can still be paid


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
    stripe_poll_interval_s: float = 15.0  # background and status checks of an unpaid Checkout link
    paid_check_interval_s: float = 3.0  # the miner's "paid" command asks Stripe at most this often
    late_check_interval_s: float = 60.0  # past the payment window: last looks at the link before expiring
    #: at most this many evaluations opened per 24 h, all miners together (None: no cap). The test
    #: launch sets it: test payments cost miners nothing, but judges and pods cost us money.
    max_evals_per_day: int | None = None
    #: False: the API is read-only (info, leaderboard, results, kings): the test launch shows the code and
    #: the rules, while an evaluation would cost as much as in production and give nothing in return
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


class IntakeService:
    def __init__(
        self,
        queue: SubmissionQueue,
        billing: Any,  # billing.BillingService
        period: Callable[[], Any],  # -> PeriodContext of the current period
        store_for: Callable[[str], Any],  # period_id -> SubmissionStore
        registration: Callable[[str], int | None],  # hotkey -> its registration block, None if not registered
        *,
        success_url: str,
        cancel_url: str,
        limits: IntakeLimits | None = None,
        now: Callable[[], float] = time.time,
        commitment: Callable[[str], tuple[str, int] | None] | None = None,  # hotkey -> (data, block)
        hub: Any = None,  # intake.hf.HubClient
        secret_key: bytes | None = None,  # encrypts miners' read tokens at rest
        about: dict[str, Any]
        | None = None,  # shown in /v1/info: {"mode": "test", "network": ..., "netuid": ...}
    ):
        self.queue = queue
        self.billing = billing
        self.period = period
        self.store_for = store_for
        self.registration = registration
        self.success_url = success_url
        self.cancel_url = cancel_url
        self.limits = limits or IntakeLimits()
        self._now = now
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
            "quote_total_cents": est.total_cents,
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

    _COUNTED = ("awaiting_payment", "queued", "running", "paused", "done")

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
        typo); ``hf_token`` is a read-only token for a private repo.
        """
        from humanizer_eval.intake.hf import parse_commitment

        self._check_name(name)
        if hf_token is not None and (not isinstance(hf_token, str) or not re.match(r"^hf_[A-Za-z0-9]{20,80}$",
                                                                                   hf_token)):  # fmt: skip
            raise ValueError("hf_token: a Hugging Face token (hf_...) that can read the repo")
        if self.commitment is None or self.hub is None:
            raise Unavailable("Hugging Face submissions are not configured here")
        reg_block = self._registration(hotkey)
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
        info = self.hub.bundle_info(repo, rev, hf_token)
        if not 0 < info.size <= self.limits.max_bundle_bytes:
            raise ValueError(f"bundle size {info.size} outside 1..{self.limits.max_bundle_bytes} bytes")
        if info.private and not hf_token:
            raise ValueError("the repo is private: give a read-only hf_token")
        secret = None
        if hf_token:
            if self.secret_key is None:
                raise Unavailable("private repos are not accepted here")
            from humanizer_eval.period.crypto import encrypt_at_rest

            secret = encrypt_at_rest(self.secret_key, hf_token.encode()).hex()
        return self._open(
            hotkey,
            reg_block,
            bundle_url=self.hub.resolve_url(repo, rev),
            bundle_sha256=info.sha256,
            bundle_bytes=info.size,
            name=name,
            hf_repo=repo,
            hf_revision=rev,
            commit_block=block,
            secret=secret,
            extra={"bundle": {"repo": repo, "commit": rev, "sha256": info.sha256, "bytes": info.size}},
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
            "quote": {k: q[k] for k in ("id", "total_cents", "hold_cents", "items", "expires_ts")},
            **(extra or {}),
        }
        row = self._try_enqueue(self.queue.get(sid))  # type: ignore[arg-type]
        if row.state == "queued":
            out.update(state="queued", queue_position=self.queue.position(sid), payment=None)
            return out
        try:
            pay = self.billing.checkout(
                q["id"], f"{self.success_url}?sid={sid}", f"{self.cancel_url}?sid={sid}"
            )
        except Exception:
            self.queue.update(sid, state="cancelled", finished_ts=self._now(), note="checkout failed")
            raise
        self.queue.update(sid, checkout_session=pay.get("session_id"), payment_url=pay.get("url"))
        out.update(state="awaiting_payment", payment={**pay, "then": f"POST /v1/submissions/{sid}/paid"})
        return out

    def _try_enqueue(self, row: Row, chk: dict[str, Any] | None = None) -> Row:
        """awaiting_payment -> queued if the balance covers the hold; -> expired past the payment window.

        A Checkout link paid before it expired counts even when the payment is confirmed later: past the
        window we ask Stripe once more (at most every ``late_check_interval_s``) and expire only on a
        definite "not paid"; while Stripe does not answer, nothing is decided."""
        if row.state != "awaiting_payment":
            return row
        late = row.expires_ts is not None and self._now() > row.expires_ts
        if late and self._now() > row.expires_ts + LATE_GIVE_UP_S:  # type: ignore[operator]
            return self._expire(row)  # a payment confirmed after this still lands on the balance
        if late and row.checkout_session:
            chk = chk or self._check_payment(row, self.limits.late_check_interval_s)
            if chk is None or (chk["paid"] and not chk["credited"]) or chk.get("status") == "open":
                return row  # no answer, fee not known yet, or the link is still open: decide later
            if not chk["paid"]:
                return self._expire(row)
        elif late:
            return self._expire(row)
        if self.billing.amount_due(row.quote_id) > 0:
            return row
        try:
            self.billing.start(row.quote_id, row.id, paid_in_time=late)
        except BillingError:
            return row  # spent elsewhere in the meantime (e.g. a refund)
        try:
            return self.queue.enqueue(row.id)
        except QueueError:  # cancelled concurrently: give the money back
            self.billing.fail(row.id)
            return self.queue.get(row.id)  # type: ignore[return-value]

    def _expire(self, row: Row) -> Row:
        try:
            return self.queue.update(row.id, expect=("awaiting_payment",), state="expired", payment_url=None)
        except QueueError:
            return self.queue.get(row.id)  # type: ignore[return-value]

    def enqueue_ready(self) -> list[str]:
        """Called after a credited payment and by the worker loop: also notices paid links by itself
        (at most every ``stripe_poll_interval_s`` per submission), so the queue follows payment order
        even if the miner never runs "paid"."""
        moved = []
        for row in self.queue.by_state("awaiting_payment"):
            if self._poll_payment(self._try_enqueue(row)).state != "awaiting_payment":
                moved.append(row.id)
        return moved

    def stripe_webhook(self, payload: bytes, sig_header: str, secret: str) -> str:
        r = self.billing.handle_webhook(payload, sig_header, secret)
        if r == "credited":
            self.enqueue_ready()
        return r

    def _check_payment(self, row: Row, min_interval: float) -> dict[str, Any] | None:
        """Ask Stripe about the row's Checkout link and credit it if paid; None if not asked (too soon,
        no link) or Stripe did not answer."""
        if row.state != "awaiting_payment" or not row.checkout_session:
            return None
        now = self._now()
        if now - self._polled.get(row.id, 0.0) < min_interval:  # inf: never ask
            return None
        self._polled[row.id] = now
        try:
            return self.billing.check_checkout(row.checkout_session)
        except BillingError as e:  # paid, but the processor's fee is not known yet
            return {"status": "complete", "paid": True, "credited": False, "pending": str(e)}
        except Exception:
            return None

    def _poll_payment(self, row: Row) -> Row:
        """Webhooks are optional: ask Stripe whether the link was paid (rate-limited)."""
        chk = self._check_payment(row, self.limits.stripe_poll_interval_s)
        if chk is not None and chk["credited"]:
            return self._try_enqueue(self.queue.get(row.id), chk)  # type: ignore[arg-type]
        return row

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
            out.append({"submission_id": r.id, "hotkey": r.hotkey, "name": r.name or "",
                        "finished_ts": r.finished_ts, **scores})  # fmt: skip
        return {"period_id": pid, "results": out}

    # --- per submission ------------------------------------------------------------------------------
    def _own(self, hotkey: str, sid: str) -> Row:
        row = self.queue.get(sid)
        if row is None or row.hotkey != hotkey or row.owner_job:
            raise NotFound("no such submission for this hotkey")
        return row

    def status(self, hotkey: str, sid: str) -> dict[str, Any]:
        row = self._poll_payment(self._try_enqueue(self._own(hotkey, sid)))
        out: dict[str, Any] = {
            "submission_id": sid,
            "state": row.state,
            "period_id": row.period_id,
            "created_ts": row.created_ts,
            "enqueued_ts": row.enqueued_ts,
            "started_ts": row.started_ts,
            "finished_ts": row.finished_ts,
            "hold_cents": row.hold_cents,
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
            out["amount_due_cents"] = self.billing.amount_due(row.quote_id)
            out["quote_expires_ts"] = row.expires_ts
            out["payment"] = {"url": row.payment_url, "amount_cents": out["amount_due_cents"],
                              "expires_ts": row.expires_ts}  # fmt: skip
        if row.period_id:
            pub = self.store_for(row.period_id).status(sid)
            if pub is not None:
                out["evaluation"] = {k: v for k, v in pub.items() if k not in ("hotkey", "submission_id")}
        return out

    def paid(self, hotkey: str, sid: str) -> dict[str, Any]:
        """The miner says the Checkout link is paid: ask Stripe now and, if so, join the queue."""
        row = self._own(hotkey, sid)
        chk = None
        if row.state == "awaiting_payment":
            chk = self._check_payment(row, self.limits.paid_check_interval_s)
            self._try_enqueue(self.queue.get(sid), chk)  # type: ignore[arg-type]
        out = self.status(hotkey, sid)
        st = out["state"]
        if st == "awaiting_payment":
            if chk is not None and chk["paid"]:
                out["message"] = "payment received; Stripe is still confirming it: run paid again in a minute"
            elif chk is None and row.checkout_session:
                out["message"] = "could not reach Stripe (or asked too often): run paid again shortly"
            else:
                out["message"] = "no payment yet: pay at payment.url before payment.expires_ts, then run paid"
        elif st in ("queued", "paused"):
            out["message"] = f"payment confirmed; in the queue at position {out.get('queue_position')}"
        elif st == "expired":
            out["message"] = "the payment window closed unpaid; submit again (the registration is not used)"
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
        self.queue.update(
            sid, expect=("awaiting_payment", "queued"), state="cancelled", finished_ts=self._now()
        )
        released = 0
        h = self.billing.accounts.hold(sid)
        if h is not None and h.status == "open":
            self.billing.fail(sid)
            released = h.amount_cents
        return {"submission_id": sid, "state": "cancelled", "released_cents": released}

    # --- money ---------------------------------------------------------------------------------------
    def balance(self, hotkey: str) -> dict[str, Any]:
        acc = self.billing.accounts
        return {
            "hotkey": hotkey,
            "balance_cents": acc.balance(hotkey),
            "available_cents": acc.available(hotkey),
        }

    def refund(self, hotkey: str, key: str, amount_cents: int | None = None) -> dict[str, Any]:
        if not re.match(r"^[A-Za-z0-9_-]{8,64}$", key or ""):
            raise ValueError("refund key: 8-64 of [A-Za-z0-9_-] (reuse it to retry safely)")
        if amount_cents is not None and (not isinstance(amount_cents, int) or amount_cents <= 0):
            raise ValueError("amount_cents must be a positive integer")
        rows = self.billing.refund(hotkey, f"{hotkey}:{key}", amount_cents)
        return {"refunds": rows, **self.balance(hotkey)}
