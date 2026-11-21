import hashlib
import hmac
import json
import time
from dataclasses import replace

import httpx
import pytest

from billing.test_billing import CARD, FakeStripe
from evaluation.test_pipeline import KEY, FakeGates, FakeLLM, N, _ctx, _detectors, _outputs
from humanizer_eval.billing import Accounts, BillingService, StripeAPI
from humanizer_eval.detectors.base import DetectorScore
from humanizer_eval.evaluation import EvalConfig, KingRun, ReplayRunner, SubmissionStore
from humanizer_eval.intake import (
    IntakeLimits,
    IntakeService,
    NonceStore,
    QueueError,
    QueueWorker,
    SubmissionQueue,
    Verifier,
    WorkerEnv,
    check_bundle_url,
)
from humanizer_eval.intake.api import create_app
from humanizer_eval.intake.client import MinerClient, SeedSigner
from humanizer_eval.scoring.crown import KingState

URL = "https://huggingface.co/miner/bundle/resolve/main/b.tar"
SHA = "ab" * 32
WHSEC = "whsec_test"


def _sha(hotkey: str) -> str:
    return hashlib.sha256(hotkey.encode()).hexdigest()


class CountingReplay(ReplayRunner):
    calls = 0

    def run(self, *a, **kw):
        CountingReplay.calls += 1
        return super().run(*a, **kw)


class World:
    """Intake service, queue, billing (fake Stripe), worker on the fake period from the pipeline tests."""

    def __init__(self, tmp_path, *, registered=None, limits=None):
        self.t = [time.time()]
        now = lambda: self.t[0]  # noqa: E731
        self.fake = FakeStripe()
        api = StripeAPI("sk_test_x", transport=httpx.MockTransport(self.fake.handler))
        self.billing = BillingService(
            CARD, Accounts(tmp_path / "b.sqlite"), api, now=now, sleep=lambda s: None
        )
        self.queue = SubmissionQueue(tmp_path / "q.sqlite", now=now)
        self.ctx = _ctx()
        self.stores = {}
        self.registered = registered
        self.regs, self.gone = {}, set()
        self.svc = IntakeService(
            self.queue,
            self.billing,
            lambda: self.ctx,
            self.store_for,
            self.registration,
            success_url="https://eval.test/v1/checkout/done",
            cancel_url="https://eval.test/v1/checkout/cancelled",
            limits=limits or IntakeLimits(stripe_poll_interval_s=0, allow_raw_urls=True),
            now=now,
        )
        self.bundles = {URL: _outputs(self.ctx)}
        self.dets = _detectors()
        self.env = WorkerEnv(
            period=lambda: self.ctx,
            store_for=self.store_for,
            llm=FakeLLM(),
            gate_factory=FakeGates,
            detectors=self.dets,
            runner_for=lambda sub: CountingReplay(self.bundles[sub.bundle_url], pod_seconds=600),
            king_run_path=lambda ctx: tmp_path / f"king_run.{ctx.period_id}.enc",
            king_state_path=tmp_path / "king_state.json",
            key=KEY,
            cfg=EvalConfig(workers=4, crown_kw={"n_boot": 300}),
            is_registered=self.is_registered,
        )
        self.worker = QueueWorker(self.queue, self.svc, self.billing, self.env, now=now, pause_base_s=60)
        self.tmp = tmp_path
        self.verifier = Verifier(NonceStore(tmp_path / "n.sqlite", ttl=600), max_skew=300)

    regs: dict  # hotkey -> registration block (default 7)
    gone: set  # deregistered hotkeys

    def registration(self, hk):
        if hk in self.gone or (self.registered is not None and hk not in self.registered):
            return None
        return self.regs.get(hk, 7)

    def is_registered(self, hk, reg):
        b = self.registration(hk)
        return b is not None and (reg is None or b == reg)

    def store_for(self, pid):
        return self.stores.setdefault(pid, SubmissionStore(self.tmp / "periods" / pid / "subs", KEY))

    def client(self, signer):
        from fastapi.testclient import TestClient

        app = create_app(self.svc, self.verifier, webhook_secret=WHSEC)
        return MinerClient("http://testserver", signer, http=TestClient(app))

    def pay(self, hotkey, cents=100_000):
        self.billing.accounts.record_payment(
            f"pi_{hotkey[:6]}_{cents}_{len(self.fake.pis)}", hotkey, cents, 0
        )


def _webhook(http, session_id):
    payload = json.dumps(
        {
            "type": "checkout.session.completed",
            "data": {"object": {"id": session_id, "payment_status": "paid"}},
        }
    ).encode()
    ts = int(time.time())
    sig = hmac.new(WHSEC.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return http.post("/v1/stripe/webhook", content=payload, headers={"stripe-signature": f"t={ts},v1={sig}"})


def test_full_flow_through_the_api(tmp_path):
    w = World(
        tmp_path, limits=IntakeLimits(stripe_poll_interval_s=float("inf"), allow_raw_urls=True)
    )  # webhook, no polling
    miner = SeedSigner(b"\x05" * 32)
    c = w.client(miner)
    info = c.info()
    assert info["period_id"] == "p-test" and info["n_inputs"] == N and info["queue"]["waiting"] == 0
    sub = c.submit(URL, SHA, 10**9, name="v1")
    sid = sub["submission_id"]
    assert sub["state"] == "awaiting_payment" and sub["payment"]["url"].startswith(
        "https://checkout.stripe.test/"
    )
    st = c.status(sid)
    assert st["state"] == "awaiting_payment" and st["amount_due_cents"] == sub["payment"]["amount_cents"]
    # Stripe calls the webhook: the payment is credited and the submission joins the queue
    r = _webhook(c.http, sub["payment"]["session_id"])
    assert r.status_code == 200 and r.json()["result"] == "credited"
    st = c.status(sid)
    assert st["state"] == "queued" and st["queue_position"] == 1
    assert w.billing.accounts.hold(sid).status == "open"
    with pytest.raises(Exception, match="409"):
        c.refund("refund-key-1")  # money is held while the submission waits or runs
    # the worker: the baseline king is scored on this period first, then the challenger
    out = w.worker.run_once()
    assert out["state"] == "done" and out["crowned"]
    st = c.status(sid)
    ev = st["evaluation"]
    assert st["state"] == "done" and ev["stage"] == "done" and ev["crowned"] is True
    assert ev["S"] > ev["S_king"] and ev["charged_cents"] > 0
    assert set(ev) <= {"stage", "updated_at", "period_id", "S", "crowned", "bundle_ran", "S_king", "delta",
                       "lower_bound", "alpha", "charged_cents", "levels", "gates", "judge_errors",
                       "detectors"}  # fmt: skip
    assert ev["levels"] == {"1": N, "0.75": 0, "0.5": 0, "0": 0} and set(ev["detectors"]) == set(w.dets)
    assert all(set(v) == {"E", "auroc", "p_human"} for v in ev["detectors"].values())
    assert "reasons" not in json.dumps(ev)  # judges' reasons quote the hidden texts
    assert c.statement(sid)["settlement"]["charged_cents"] == ev["charged_cents"]
    state = KingState.from_json((tmp_path / "king_state.json").read_text())
    assert state.current.submission_id == sid and state.history[0].is_baseline
    bal = c.balance()
    assert bal["available_cents"] == bal["balance_cents"] > 0
    ref = c.refund("refund-key-1")
    assert ref["refunds"][0]["status"] == "succeeded" and ref["balance_cents"] == 0


def test_bad_requests(tmp_path):
    miner, stranger = SeedSigner(b"\x05" * 32), SeedSigner(b"\x06" * 32)
    w = World(tmp_path, registered={miner.ss58_address})
    c, cs = w.client(miner), w.client(stranger)
    with pytest.raises(Exception, match="403"):
        cs.submit(URL, SHA, 10)
    for url in (
        "http://huggingface.co/x.tar",
        "https://10.0.0.1/x.tar",
        "https://[::1]/x.tar",
        "https://user:pw@huggingface.co/x.tar",
        "https://localhost/x.tar",
        "https://metadata.internal/x",
        "https://huggingface.co:8443/x.tar",
        "https://huggingface.co/x .tar",
    ):
        with pytest.raises(Exception, match="422"):
            c.submit(url, SHA, 10)
    with pytest.raises(Exception, match="422"):
        c.submit(URL, "AB" * 32, 10)
    with pytest.raises(Exception, match="422"):
        c.submit(URL, SHA, 50 * 1024**3 + 1)  # over the 50 GB bundle limit (D-61)
    sid = c.submit(URL, SHA, 10)["submission_id"]
    with pytest.raises(Exception, match="409"):
        c.submit(URL, SHA, 10)  # one active submission per hotkey
    with pytest.raises(Exception, match="404"):
        cs.status(sid)  # another hotkey cannot see it
    # unsigned / badly signed requests
    r = c.http.get("/v1/balance")
    assert r.status_code == 401
    assert check_bundle_url(URL) == URL


def test_queue_cap_and_payment_order(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(max_waiting=2, stripe_poll_interval_s=0, allow_raw_urls=True))
    signers = [SeedSigner(bytes([i]) * 32) for i in (1, 2, 3)]
    for s in signers:
        w.pay(s.ss58_address)
    a = w.svc.submit(
        signers[0].ss58_address, bundle_url=URL, bundle_sha256=_sha(signers[0].ss58_address), bundle_bytes=10
    )
    w.t[0] += 1
    b = w.svc.submit(
        signers[1].ss58_address, bundle_url=URL, bundle_sha256=_sha(signers[1].ss58_address), bundle_bytes=10
    )
    assert a["state"] == b["state"] == "queued" and (a["queue_position"], b["queue_position"]) == (1, 2)
    with pytest.raises(QueueError, match="full"):
        w.svc.submit(
            signers[2].ss58_address,
            bundle_url=URL,
            bundle_sha256=_sha(signers[2].ss58_address),
            bundle_bytes=10,
        )
    first = w.queue.claim()
    assert first.id == a["submission_id"] and w.queue.claim() is None  # one evaluation at a time


def test_cancel_releases_the_hold_and_expired_quote(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sub = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)
    sid = sub["submission_id"]
    assert w.billing.accounts.available(m) < 100_000
    out = w.svc.cancel(m, sid)
    assert out["released_cents"] > 0 and w.billing.accounts.available(m) == 100_000
    assert w.queue.claim() is None
    # an unpaid quote expires; the miner submits again later
    m2 = SeedSigner(b"\x06" * 32).ss58_address
    sub2 = w.svc.submit(m2, bundle_url=URL, bundle_sha256=_sha(m2), bundle_bytes=10)
    w.fake.paid = False
    w.fake.sessions[sub2["payment"]["session_id"]]["expired"] = True  # Stripe closed the link
    w.t[0] += 3601
    assert w.svc.status(m2, sub2["submission_id"])["state"] == "expired"


def test_status_poll_credits_a_paid_checkout_without_webhook(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    sub = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)
    assert sub["state"] == "awaiting_payment"
    st = w.svc.status(m, sub["submission_id"])  # FakeStripe reports the session paid
    assert st["state"] == "queued"


def _failing_detectors(dets, name):
    class Down:
        def __init__(self, inner):
            self.inner, self.name = inner, inner.name

        def score(self, texts):
            return [DetectorScore(self.name, "h", None, error="budget: cap reached") for _ in texts]

    return {**dets, name: Down(dets[name])}


def test_pause_backoff_then_resume_without_rerunning_the_pod(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    w.worker.ensure_king(w.ctx)  # king first, with the detectors up
    CountingReplay.calls = 0
    ok_dets = w.env.detectors
    w.env.detectors = _failing_detectors(ok_dets, "pangram")
    assert w.worker.run_once()["state"] == "paused"
    row = w.queue.get(sid)
    assert row.state == "paused" and row.retry_at == pytest.approx(w.t[0] + 60)
    assert w.svc.status(m, sid)["evaluation"]["stage"] == "paused"
    assert w.billing.accounts.hold(sid).status == "open"
    assert w.queue.claim() is None  # not due yet
    w.env.detectors = ok_dets
    w.t[0] += 61
    assert w.worker.run_once()["state"] == "done"
    assert CountingReplay.calls == 1  # resumed from the saved outputs


def test_give_up_after_max_pauses_charges_nothing(tmp_path):
    w = World(tmp_path)
    w.worker.max_pauses = 1
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    w.worker.ensure_king(w.ctx)
    w.env.detectors = _failing_detectors(w.env.detectors, "gptzero")
    assert w.worker.run_once()["state"] == "paused"
    w.t[0] += 10_000
    assert w.worker.run_once()["state"] == "failed"
    assert w.billing.accounts.hold(sid).status == "released" and w.billing.accounts.available(m) == 100_000
    assert w.svc.status(m, sid)["evaluation"]["stage"] == "failed"


def test_restart_requeues_a_running_row(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    assert w.queue.claim().id == sid  # the process dies here
    assert w.queue.recover() == [sid]
    assert w.worker.run_once()["state"] == "done"
    assert w.queue.get(sid).attempts == 2


def test_new_period_reruns_the_king_bundle_at_owner_cost(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    assert w.worker.run_once()["crowned"]
    charged = w.billing.accounts.balance(m)
    # period rotates: the king's bundle runs on the new inputs before any challenger
    w.ctx = replace(w.ctx, period_id="p-next")
    CountingReplay.calls = 0
    assert w.worker.run_once() is None  # nothing queued, but the king was re-run
    assert CountingReplay.calls == 1
    kr = KingRun.load(tmp_path / "king_run.p-next.enc", KEY)
    assert kr.submission_id == sid and kr.hotkey == m
    assert w.billing.accounts.balance(m) == charged  # the owner pays the king's re-run
    assert w.store_for("p-next").status(f"{sid}-kingrun")["stage"] == "done"


def test_worker_crash_is_retried_then_released(tmp_path):
    w = World(tmp_path)
    w.worker.max_attempts = 2
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    w.worker.ensure_king(w.ctx)

    def boom(sub):
        raise RuntimeError("bug")

    w.env.runner_for = boom
    assert w.worker.run_once()["state"] == "paused"
    w.t[0] += 601
    assert w.worker.run_once()["state"] == "failed"
    assert w.billing.accounts.hold(sid).status == "released"


def test_leaderboard_endpoint_serves_a_verifiable_envelope(tmp_path):
    from fastapi.testclient import TestClient
    from nacl.signing import SigningKey

    from humanizer_eval.publish import build_payload, sign, verify

    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)
    assert w.worker.run_once()["crowned"]
    key = SigningKey(b"\x02" * 32)

    def board():
        return sign(build_payload(KingState.from_json((tmp_path / "king_state.json").read_text())), key)

    http = TestClient(create_app(w.svc, w.verifier, leaderboard=board))
    p = verify(http.get("/v1/leaderboard").json(), key.verify_key.encode().hex(), netuid=32)
    assert p["weights"] == {m: 1.0} and p["king"]["hotkey"] == m
    assert TestClient(create_app(w.svc, w.verifier)).get("/v1/leaderboard").status_code == 404


def test_one_submission_per_registration(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m, 300_000)
    # a submission cancelled before it ran does not use the registration up
    first = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)
    w.svc.cancel(m, first["submission_id"])
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    assert w.queue.get(sid).reg_block == 7
    assert w.worker.run_once()["state"] == "done"
    with pytest.raises(QueueError, match="registration already made its submission"):
        w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)
    w.regs[m] = 99  # deregistered and registered again: a new registration may submit once
    assert w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["state"] == "queued"


def test_our_failure_leaves_the_registration_free(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    w.worker.ensure_king(w.ctx)
    w.worker.max_attempts = 1

    def boom(sub):
        raise RuntimeError("bug")

    w.env.runner_for = boom
    assert w.worker.run_once()["state"] == "failed" and w.queue.get(sid).state == "failed"
    w.env.runner_for = lambda sub: CountingReplay(w.bundles[sub.bundle_url], pod_seconds=600)
    assert w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["state"] == "queued"


def test_deregistered_current_king_gives_the_crown_back(tmp_path):
    w = World(tmp_path)
    a, b = SeedSigner(b"\x05" * 32).ss58_address, SeedSigner(b"\x06" * 32).ss58_address
    url_b = "https://huggingface.co/b/bundle/resolve/main/b.tar"
    w.bundles[URL] = _outputs(w.ctx, broken=set(range(10)))  # A: better than the baseline
    w.bundles[url_b] = _outputs(w.ctx)  # B: better than A
    for hk in (a, b):
        w.pay(hk)
    sa = w.svc.submit(a, bundle_url=URL, bundle_sha256=_sha(a), bundle_bytes=10)["submission_id"]
    assert w.worker.run_once()["crowned"]
    sb = w.svc.submit(b, bundle_url=url_b, bundle_sha256=_sha(b), bundle_bytes=10)["submission_id"]
    assert w.worker.run_once()["crowned"]
    assert KingRun.load(tmp_path / "king_run.p-test.enc", KEY).submission_id == sb
    # B loses its registration: A is king again, from A's saved outputs (no pod)
    w.gone.add(b)
    CountingReplay.calls = 0
    kr = w.worker.ensure_king(w.ctx)
    assert kr.submission_id == sa and kr.hotkey == a and CountingReplay.calls == 0
    st = KingState.from_json((tmp_path / "king_state.json").read_text())
    assert st.current.submission_id == sb and st.current_active(w.is_registered).submission_id == sa
    assert st.payout_weights(w.is_registered) == {a: 1.0}
    # A's own record in this period is untouched by the owner's run
    assert w.store_for("p-test").status(sa)["crowned"] is True


def test_payouts_close_ranks_when_a_king_is_deregistered():
    from datetime import UTC, datetime, timedelta

    t = datetime(2026, 11, 1, tzinfo=UTC)
    st = KingState.initial("base", "owner", t)
    for i, hk in enumerate(("A", "B", "C", "D")):
        st.apply_crown(f"s{hk}", hk, t + timedelta(days=i + 1), 0.01, registered_at=100 + i)
    regs = {"A": 100, "B": 101, "C": 102, "D": 103}

    def reg(hk, at):
        return regs.get(hk) == at

    assert st.payout_weights(reg) == {"D": 0.85, "C": 0.10, "B": 0.05}
    regs.pop("C")  # the 4th-newest king (A) comes back into the third slot
    assert st.payout_weights(reg) == {"D": 0.85, "B": 0.10, "A": 0.05}
    regs["D"] = 500  # D registered again: a new registration is not the old king
    assert st.payout_weights(reg) == {"B": 0.85 + 0.05, "A": 0.10}
    assert st.current_active(reg).submission_id == "sB"


def test_snapshot_registration_blocks(tmp_path):
    from humanizer_eval.intake import MetagraphSnapshot, write_snapshot

    p = tmp_path / "mg.json"
    write_snapshot(p, 32, 1000, ["5A", "5B"], {"5A": 10, "5B": 20})
    m = MetagraphSnapshot(p)
    assert m.registration("5A") == 10 and m.registration("5C") is None
    assert m.is_registered("5B", 20) and not m.is_registered("5B", 21) and not m.is_registered("5C", None)


# --- paying by link from the terminal: submit -> Stripe link -> pay -> "paid" ------------------------
def _link_world(tmp_path, **kw):
    lim = IntakeLimits(
        stripe_poll_interval_s=float("inf"), paid_check_interval_s=0, allow_raw_urls=True, **kw
    )
    w = World(tmp_path, limits=lim)
    w.fake.paid = False
    return w


def test_pay_by_link_then_paid_command(tmp_path):
    w = _link_world(tmp_path)
    c = w.client(SeedSigner(b"\x06" * 32))
    sub = c.submit(URL, SHA, 10**9)
    sid, pay = sub["submission_id"], sub["payment"]
    form = next(f for m, p, f, _ in w.fake.calls if m == "POST" and p == "/v1/checkout/sessions")
    assert form["payment_method_types[0]"] == "card"
    assert int(form["expires_at"]) == pay["expires_at"] == int(sub["quote"]["expires_ts"])
    assert form["success_url"] == f"https://eval.test/v1/checkout/done?sid={sid}"
    assert pay["then"] == f"POST /v1/submissions/{sid}/paid"
    r = c.paid(sid)  # not paid yet: the link again, and the deadline
    assert r["state"] == "awaiting_payment" and "no payment yet" in r["message"]
    assert r["payment"] == {
        "url": pay["url"],
        "amount_cents": pay["amount_cents"],
        "expires_ts": sub["quote"]["expires_ts"],
    }
    w.fake.paid = True
    r = c.paid(sid)
    assert r["state"] == "queued" and r["queue_position"] == 1 and "confirmed" in r["message"]
    assert w.billing.accounts.hold(sid).status == "open"
    bal = c.balance()["balance_cents"]
    assert c.paid(sid)["state"] == "queued" and c.balance()["balance_cents"] == bal  # credited once
    page = c.http.get(f"/v1/checkout/done?sid={sid}").text
    assert f"paid {sid}" in page
    assert "<submission_id>" in c.http.get("/v1/checkout/done?sid=<script>").text


def test_paid_before_the_link_expired_counts_after_the_window(tmp_path):
    w = _link_world(tmp_path)
    sid = w.svc.submit("5Fhk-late", bundle_url=URL, bundle_sha256=_sha("late"), bundle_bytes=10**9)[
        "submission_id"
    ]
    w.fake.paid = True  # paid in the last minute ...
    w.t[0] += 3700  # ... confirmed after the window
    r = w.svc.paid("5Fhk-late", sid)
    assert r["state"] == "queued" and w.billing.accounts.hold(sid).status == "open"


def test_unpaid_link_expires_but_a_stripe_outage_decides_nothing(tmp_path):
    w = _link_world(tmp_path)
    hk = "5Fhk-unpaid"
    sub = w.svc.submit(hk, bundle_url=URL, bundle_sha256=_sha("u1"), bundle_bytes=10**9)
    sid, cs = sub["submission_id"], sub["payment"]["session_id"]
    w.t[0] += 3700
    w.fake.down = True
    assert w.svc.status(hk, sid)["state"] == "awaiting_payment"  # Stripe did not answer: wait
    w.fake.down = False
    w.t[0] += 61
    assert w.svc.status(hk, sid)["state"] == "awaiting_payment"  # the link is still open at Stripe
    w.fake.sessions[cs]["expired"] = True
    w.t[0] += 61
    r = w.svc.paid(hk, sid)
    assert r["state"] == "expired" and "submit again" in r["message"] and "payment" not in r
    # an unpaid quote does not use the registration
    assert (
        w.svc.submit(hk, bundle_url=URL, bundle_sha256=_sha("u2"), bundle_bytes=10**9)["state"]
        == "awaiting_payment"
    )


def test_a_paid_link_is_noticed_without_the_command(tmp_path):
    w = _link_world(tmp_path)
    w.svc.limits.stripe_poll_interval_s = 0
    sid = w.svc.submit("5Fhk-bg", bundle_url=URL, bundle_sha256=_sha("bg"), bundle_bytes=10**9)[
        "submission_id"
    ]
    assert w.svc.enqueue_ready() == []
    w.fake.paid = True
    assert w.svc.enqueue_ready() == [sid] and w.queue.get(sid).state == "queued"


def test_public_results_show_scores_not_texts_or_money(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(stripe_poll_interval_s=float("inf"), allow_raw_urls=True))
    m = SeedSigner(b"\x07" * 32)
    w.pay(m.ss58_address)
    sid = w.svc.submit(m.ss58_address, bundle_url=URL, bundle_sha256=_sha("pub"), bundle_bytes=10, name="v7")[
        "submission_id"
    ]
    assert w.worker.run_once()["state"] == "done"
    http = w.client(m).http
    r = http.get("/v1/results")
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "*"
    body = r.json()
    assert body["period_id"] == "p-test" and [x["submission_id"] for x in body["results"]] == [sid]
    res = body["results"][0]
    assert res["hotkey"] == m.ss58_address and res["name"] == "v7" and res["crowned"] is True
    assert set(res) <= {"submission_id", "hotkey", "name", "finished_ts", "S", "crowned", "S_king", "delta",
                        "lower_bound", "alpha", "levels", "gates", "judge_errors", "detectors"}  # fmt: skip
    assert "charged_cents" not in res and "reasons" not in json.dumps(res)
    assert http.get("/v1/results?period=other").json()["results"] == []
    assert http.get("/v1/results?period=../x").status_code == 422
    assert http.get("/v1/info").headers["access-control-allow-origin"] == "*"


def test_test_mode_daily_cap_and_info(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(stripe_poll_interval_s=float("inf"), allow_raw_urls=True,
                                            max_evals_per_day=2))  # fmt: skip
    w.svc.about = {"mode": "test", "network": "test", "netuid": 87}
    info = w.svc.info()
    assert (info["mode"], info["network"], info["netuid"]) == ("test", "test", 87)
    assert info["limits"]["max_evals_per_day"] == 2 and info["limits"]["evals_last_24h"] == 0
    a = w.svc.submit("5Fcap-a", bundle_url=URL, bundle_sha256=_sha("a"), bundle_bytes=10)
    w.svc.submit("5Fcap-b", bundle_url=URL, bundle_sha256=_sha("b"), bundle_bytes=10)
    m = SeedSigner(b"\x08" * 32)
    http = w.client(m).http
    with pytest.raises(Exception, match="429"):
        w.client(m).submit(URL, _sha("c"), 10)
    assert http.get("/v1/info").json()["limits"]["evals_last_24h"] == 2
    w.svc.cancel("5Fcap-a", a["submission_id"])  # a cancelled one frees its place
    assert w.svc.submit("5Fcap-d", bundle_url=URL, bundle_sha256=_sha("d"), bundle_bytes=10)["state"]
    w.t[0] += 86401  # a day later the window is empty again
    assert w.svc.info()["limits"]["evals_last_24h"] == 0


def test_closed_evaluator_is_read_only(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(stripe_poll_interval_s=float("inf"), allow_raw_urls=True,
                                            submissions_open=False))  # fmt: skip
    m = SeedSigner(b"\x09" * 32)
    c = w.client(m)
    with pytest.raises(Exception, match="403"):
        c.submit(URL, _sha("closed"), 10)
    info = c.http.get("/v1/info").json()
    assert info["limits"]["submissions_open"] is False and w.queue.counts() == {}
    for path in ("/v1/leaderboard", "/v1/results", "/v1/kings"):
        assert c.http.get(path).status_code in (200, 404)  # public routes stay up (404: not wired in tests)


def test_claim_allows_up_to_max_running(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(stripe_poll_interval_s=float("inf"), allow_raw_urls=True))
    for k in ("a", "b", "c"):
        hk = f"5F{k}par"
        w.pay(hk)
        w.svc.submit(hk, bundle_url=URL, bundle_sha256=_sha(hk), bundle_bytes=10)
    assert w.queue.claim(max_running=2) is not None and w.queue.claim(max_running=2) is not None
    assert w.queue.claim(max_running=2) is None  # two running: the third waits
    assert w.queue.claim(max_running=3) is not None


def test_crown_is_re_decided_when_the_king_changed_during_the_run(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(stripe_poll_interval_s=float("inf"), allow_raw_urls=True))
    a, b = "5Farace", "5Fbrace"
    for hk in (a, b):
        w.pay(hk)
    sa = w.svc.submit(a, bundle_url=URL, bundle_sha256=_sha(a), bundle_bytes=10)["submission_id"]
    sb = w.svc.submit(b, bundle_url=URL, bundle_sha256=_sha(b), bundle_bytes=10)["submission_id"]
    k0 = w.worker.ensure_king(w.ctx)  # the baseline, as B's evaluation will see it when it starts
    assert w.worker.run_once()["crowned"] is True  # A beats the baseline and is crowned
    w.worker.ensure_king = lambda ctx: k0  # B started before A's crown landed
    out = w.worker.run_once()
    ev = w.svc.status(b, sb)["evaluation"]
    # B has the same outputs as A: it beat the baseline, but against the new king A it does not
    assert out["submission_id"] == sb and out["crowned"] is False
    assert ev["king_changed_during_run"] is True and ev["crowned"] is False and abs(ev["delta"]) < 1e-9
    state = KingState.from_json((tmp_path / "king_state.json").read_text())
    assert state.current.submission_id == sa


def test_next_period_is_prepared_before_the_switch(tmp_path):
    from humanizer_eval.intake.worker import CurrentPeriod

    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    assert w.worker.run_once()["crowned"]
    pdir = tmp_path / "pp"
    for p in ("p1", "p2"):
        (pdir / p).mkdir(parents=True)
    (pdir / "CURRENT").write_text("p1\n")
    loads = []

    def loader(d):
        loads.append(d.name)
        return replace(w.ctx, period_id=d.name, dir=d)

    cp = CurrentPeriod(pdir, loader)
    assert cp().period_id == "p1" and cp.upcoming() is None
    (pdir / "NEXT").write_text("p2\n")
    nxt = cp.upcoming()
    CountingReplay.calls = 0
    kr = w.worker.prepare_next(nxt)
    assert kr.submission_id == sid and CountingReplay.calls == 1  # the king ran on p2 ahead of time
    assert json.loads((pdir / "p2" / "READY").read_text())["king"] == sid
    assert w.worker.prepare_next(nxt).submission_id == sid and CountingReplay.calls == 1  # once
    # the switch: p2 is served from memory, p1 is dropped; nothing is loaded twice
    (pdir / "CURRENT").write_text("p2\n")
    (pdir / "NEXT").unlink()
    assert cp().period_id == "p2" and cp.upcoming() is None
    assert loads == ["p1", "p2"]
    w.env.period = cp
    assert w.worker.ensure_king(cp()).submission_id == sid and CountingReplay.calls == 1


def test_period_schedule_and_reveal_files_are_public(tmp_path):
    from fastapi.testclient import TestClient

    w = World(tmp_path)
    pdir, rdir = tmp_path / "periods", tmp_path / "public"
    (rdir / "p1" / "submissions").mkdir(parents=True)
    (rdir / "p1" / "files.json").write_text('{"a": "b"}')
    (rdir / "secret.key").write_text("not public")
    app = create_app(w.svc, w.verifier, periods_dir=pdir, reveal_dir=rdir)
    http = TestClient(app)
    assert http.get("/v1/periods").status_code == 404
    pdir.mkdir()
    (pdir / "schedule.json").write_text(json.dumps({"current": "p1", "periods": {"p1": {"commitment": "c"}}}))
    r = http.get("/v1/periods")
    assert r.status_code == 200 and r.json()["current"] == "p1"
    assert r.headers["access-control-allow-origin"] == "*"
    assert http.get("/v1/periods/p1/files.json").json() == {"a": "b"}
    assert http.get("/v1/periods/p2/files.json").status_code == 404
    assert http.get("/v1/periods/p1/../secret.key").status_code == 404
    assert http.get("/v1/periods/p1/%2e%2e/secret.key").status_code == 404
    assert http.get("/v1/periods/..%2fsecret.key/x").status_code == 404
