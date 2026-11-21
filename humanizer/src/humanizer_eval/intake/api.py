"""HTTP API for miners (FastAPI; the `api` extra). Every route except ``/v1/info``, the Stripe webhook and
the Checkout landing pages is signed with the miner's hotkey (``intake/auth.py``).

    GET    /v1/info                         period, crown detectors, quote estimate, queue length
    GET    /v1/leaderboard                  signed payout weights for validators (mechanism 1, D-92)
    GET    /v1/results?period=<id>          finished submissions of a period: scores and aggregates (public)
    GET    /v1/kings                        the kings, newest first, with the public link to each bundle
                                            48 h after its coronation (D-70; publish/kings.py)
    GET    /v1/periods                      commitments, seed blocks, Merkle roots, reveal state (D-57)
    GET    /v1/periods/{id}/{file}          the reveal of a closed period: files.json, dataset.json, ...
    POST   /v1/submissions                  {commitment?, hf_token?, name?}: the bundle pinned by the hotkey's
                                            on-chain commitment <repo>@<commit> -> quote, payment
                                            (development: {bundle_url, bundle_sha256, bundle_bytes})
    GET    /v1/submissions/{id}             state, place in the queue, public result (D-58)
    GET    /v1/submissions/{id}/statement   line items after settlement
    POST   /v1/submissions/{id}/paid        "I have paid the Checkout link": asks Stripe now, joins the queue
    DELETE /v1/submissions/{id}             cancel before it runs (the hold is released)
    GET    /v1/balance                      balance and available (not held) cents
    POST   /v1/refunds                      {key, amount_cents?}: to the original payments, idempotent per key
    POST   /v1/stripe/webhook               Stripe events (signature checked with the webhook secret)
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool as call  # sqlite and Stripe calls block

from humanizer_eval.billing.accounts import BillingError
from humanizer_eval.billing.stripe_api import StripeError
from humanizer_eval.intake.auth import AuthError, Verifier
from humanizer_eval.intake.queue import QueueError
from humanizer_eval.intake.service import Forbidden, IntakeService, NotFound, RateLimited, Unavailable

MAX_BODY = 16 * 1024
_PID = re.compile(r"[a-z0-9][a-z0-9-]{0,40}")
#: public read-only routes a browser page (the leaderboard Space) may fetch from any origin
PUBLIC = {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET"}


def create_app(
    svc: IntakeService,
    verifier: Verifier,
    *,
    webhook_secret: str | None = None,
    leaderboard: Callable[[], dict[str, Any]] | None = None,
    kings: Callable[[], dict[str, Any]] | None = None,
    periods_dir: Path | None = None,  # schedule.json: commitments, seed blocks, Merkle roots (lifecycle)
    reveal_dir: Path | None = None,  # <id>/...: the reveal of each closed period (period/reveal.py)
):
    app = FastAPI(title="SN32 humanizer evaluation", docs_url=None, redoc_url=None, openapi_url=None)

    def err(status: int, msg: str) -> JSONResponse:
        return JSONResponse({"error": msg}, status_code=status)

    async def body_of(request: Request) -> bytes:
        cl = request.headers.get("content-length")
        if cl is not None and (not cl.isdigit() or int(cl) > MAX_BODY):
            raise ValueError("request body too large")
        body = await request.body()
        if len(body) > MAX_BODY:
            raise ValueError("request body too large")
        return body

    async def signed(request: Request) -> tuple[str, dict[str, Any]]:
        body = await body_of(request)
        hotkey = await call(verifier.verify, request.method, request.url.path, dict(request.headers), body)
        data: dict[str, Any] = {}
        if body:
            try:
                data = json.loads(body)
            except ValueError as e:
                raise ValueError("body must be JSON") from e
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
        return hotkey, data

    async def guarded(fn, request: Request):
        try:
            return JSONResponse(await fn(request))
        except AuthError as e:
            return err(401, str(e))
        except Forbidden as e:
            return err(403, str(e))
        except NotFound as e:
            return err(404, str(e))
        except (QueueError, BillingError) as e:
            return err(409, str(e))
        except ValueError as e:
            return err(422, str(e))
        except RateLimited as e:
            return err(429, str(e))
        except Unavailable as e:
            return err(503, str(e))

    @app.get("/v1/info")
    async def info(request: Request):
        async def f(_):
            return await call(svc.info)

        r = await guarded(f, request)
        r.headers.update(PUBLIC)
        return r

    @app.get("/v1/results")
    async def results(request: Request):
        async def f(r):
            return await call(svc.results, r.query_params.get("period") or None)

        r = await guarded(f, request)
        r.headers.update({**PUBLIC, "Cache-Control": "public, max-age=30"})
        return r

    @app.get("/v1/leaderboard")
    async def board():
        """Signed payout weights for validators (publish/leaderboard.py); public."""
        if leaderboard is None:
            return err(404, "no leaderboard here")
        return JSONResponse(await call(leaderboard), headers={"Cache-Control": "no-store", **PUBLIC})

    @app.get("/v1/kings")
    async def king_list():
        """The kings and where each one's bundle is public (publish/kings.py); public."""
        if kings is None:
            return err(404, "no king list here")
        return JSONResponse(await call(kings), headers={"Cache-Control": "public, max-age=30", **PUBLIC})

    @app.get("/v1/periods")
    async def period_list():
        """Commitments, seed blocks, Merkle roots and reveal state of every period (D-57); public."""
        p = periods_dir / "schedule.json" if periods_dir else None
        if p is None or not p.exists():
            return err(404, "no period schedule here")
        return JSONResponse(
            json.loads(p.read_text()), headers={"Cache-Control": "public, max-age=30", **PUBLIC}
        )

    @app.get("/v1/periods/{pid}/{rel:path}")
    async def period_file(pid: str, rel: str):
        """A file of a revealed period (files.json lists them with their sha256); public."""
        if reveal_dir is None or not _PID.fullmatch(pid):
            return err(404, "not found")
        base = (reveal_dir / pid).resolve()
        f = (base / rel).resolve()
        if not f.is_relative_to(base) or not f.is_file():
            return err(404, "not found (a period is revealed after it closes)")
        return FileResponse(f, headers={"Cache-Control": "public, max-age=3600", **PUBLIC})

    @app.post("/v1/submissions")
    async def submit(request: Request):
        async def f(r):
            hk, d = await signed(r)
            if "bundle_url" not in d:  # the normal path: the bundle pinned by the on-chain commitment
                unknown = set(d) - {"commitment", "hf_token", "name"}
                if unknown:
                    raise ValueError(f"unknown fields {sorted(unknown)}")
                return await call(svc.submit_hf, hk, commitment=d.get("commitment"),
                                  hf_token=d.get("hf_token"), name=d.get("name", ""))  # fmt: skip
            unknown = set(d) - {"bundle_url", "bundle_sha256", "bundle_bytes", "name"}
            if unknown:
                raise ValueError(f"unknown fields {sorted(unknown)}")
            return await call(
                svc.submit,
                hk,
                bundle_url=d.get("bundle_url", ""),
                bundle_sha256=d.get("bundle_sha256", ""),
                bundle_bytes=d.get("bundle_bytes", 0),
                name=d.get("name", ""),
            )

        return await guarded(f, request)

    @app.get("/v1/submissions/{sid}")
    async def status(sid: str, request: Request):
        async def f(r):
            hk, _ = await signed(r)
            return await call(svc.status, hk, sid)

        return await guarded(f, request)

    @app.get("/v1/submissions/{sid}/statement")
    async def statement(sid: str, request: Request):
        async def f(r):
            hk, _ = await signed(r)
            return await call(svc.statement, hk, sid)

        return await guarded(f, request)

    @app.post("/v1/submissions/{sid}/paid")
    async def paid(sid: str, request: Request):
        async def f(r):
            hk, _ = await signed(r)
            return await call(svc.paid, hk, sid)

        return await guarded(f, request)

    @app.delete("/v1/submissions/{sid}")
    async def cancel(sid: str, request: Request):
        async def f(r):
            hk, _ = await signed(r)
            return await call(svc.cancel, hk, sid)

        return await guarded(f, request)

    @app.get("/v1/balance")
    async def balance(request: Request):
        async def f(r):
            hk, _ = await signed(r)
            return await call(svc.balance, hk)

        return await guarded(f, request)

    @app.post("/v1/refunds")
    async def refund(request: Request):
        async def f(r):
            hk, d = await signed(r)
            return await call(svc.refund, hk, d.get("key", ""), d.get("amount_cents"))

        return await guarded(f, request)

    @app.post("/v1/stripe/webhook")
    async def webhook(request: Request):
        if not webhook_secret:
            return err(404, "webhooks are not configured")
        payload = await request.body()
        if len(payload) > 1024 * 1024:
            return err(413, "payload too large")
        try:
            r = await call(
                svc.stripe_webhook, payload, request.headers.get("stripe-signature", ""), webhook_secret
            )
        except BillingError as e:
            # Stripe retries non-2xx; a fee not yet available is worth a retry, a bad signature is not
            return err(503 if "fee" in str(e) else 400, str(e))
        except (StripeError, ValueError) as e:
            return err(400, str(e))
        return JSONResponse({"result": r})

    def _sid(request: Request) -> str:
        sid = request.query_params.get("sid", "")
        return sid if re.fullmatch(r"sub-[0-9a-f]{16}", sid) else "<submission_id>"

    @app.get("/v1/checkout/done")
    async def done(request: Request):
        """Where Stripe sends the payer (Checkout requires a return URL): plain text, what to run next."""
        return PlainTextResponse(
            "Payment received. Confirm it from your terminal:\n\n"
            f"    python scripts/miner_submit.py --api <this server> --wallet <coldkey> --hotkey <hotkey> "
            f"paid {_sid(request)}\n\n"
            "(it is also picked up automatically within a minute).\n"
        )

    @app.get("/v1/checkout/cancelled")
    async def cancelled(request: Request):
        return PlainTextResponse(
            f"Payment cancelled. Submission {_sid(request)} waits for payment until the link expires; "
            "open the same link again to pay, or cancel the submission.\n"
        )

    return app
