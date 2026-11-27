import hashlib
import json
import time
import uuid
from dataclasses import replace

import pytest

from billing.test_billing import CARD
from billing.test_tao import COLD, FakeRoute
from evaluation.test_pipeline import KEY, FakeGates, FakeLLM, N, _ctx, _detectors, _outputs
from humanizer_eval.billing import Accounts, BillingService
from humanizer_eval.billing.tao import WEI, TaoConfig, TaoRail
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


def _sha(hotkey: str) -> str:
    return hashlib.sha256(hotkey.encode()).hexdigest()


class CountingReplay(ReplayRunner):
    calls = 0

    def run(self, *a, **kw):
        CountingReplay.calls += 1
        return super().run(*a, **kw)


class World:
    """Intake service, queue, billing (TAO on a fake chain), worker on the fake period from the pipeline tests."""

    def __init__(self, tmp_path, *, registered=None, limits=None):
        self.t = [time.time()]
        now = lambda: self.t[0]  # noqa: E731
        self.route = FakeRoute()
        acc = Accounts(tmp_path / "b.sqlite")
        rail = TaoRail(tmp_path / "tao.sqlite", self.route, b"m" * 32, acc, TaoConfig(), now=now)
        self.billing = BillingService(CARD, acc, rail, now=now)
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
            limits=limits or IntakeLimits(payment_poll_interval_s=0, allow_raw_urls=True),
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
            cfg=EvalConfig(workers=4, crown_kw={"n_boot": 300}, retry_wait_s=0),
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

        app = create_app(self.svc, self.verifier)
        return MinerClient("http://testserver", signer, http=TestClient(app))

    def pay(self, hotkey, cents=100_000):
        self.billing.accounts.record_payment(f"tao:{uuid.uuid4().hex}", hotkey, cents, 0)

    def send_tao(self, payment, tao=None):
        """The miner sends the TAO the payment asks for (or ``tao``) to its deposit address."""
        a = payment["evm_address"]
        self.route.bt[a] = self.route.bt.get(a, 0) + int(float(tao or payment["amount_tao"]) * WEI)

    def converge(self, n=5):
        for _ in range(n):
            self.t[0] += 31  # past the deposit poll interval of a paying hotkey
            self.svc.enqueue_ready()


def test_full_flow_through_the_api(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
    miner = SeedSigner(b"\x05" * 32)
    c = w.client(miner)
    info = c.info()
    assert info["period_id"] == "p-test" and info["n_inputs"] == N and info["queue"]["waiting"] == 0
    sub = c.submit(URL, SHA, 10**9, name="v1")
    sid = sub["submission_id"]
    pay = sub["payment"]
    assert sub["state"] == "awaiting_payment" and pay["method"] == "tao" and float(pay["amount_tao"]) > 0
    assert c.deposit()["address"] == pay["address"]  # one deposit address per hotkey
    st = c.status(sid)
    assert st["state"] == "awaiting_payment" and st["payment"]["due_usdc"] == sub["quote"]["hold_usdc"]
    # the miner sends the TAO; the background pass bridges, swaps, credits and queues it
    w.send_tao(pay, tao=float(pay["amount_tao"]) + 0.5)  # more than due: the rest stays as USDC
    w.converge()
    st = c.status(sid)
    assert st["state"] == "queued" and st["queue_position"] == 1
    assert w.billing.accounts.hold(sid).status == "open"
    with pytest.raises(Exception, match="409"):
        c.refund("refund-key-1", COLD)  # money is held while the submission waits or runs
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
    assert bal["available_usdc"] == bal["balance_usdc"] and float(bal["balance_usdc"]) > 0
    assert float(c.refund_quote(COLD)["tao_estimate"]) > 0
    ref = c.refund("refund-key-1", COLD)  # any address the miner names
    assert ref["refund"]["to"] == COLD and ref["balance_usdc"] == "0.00"
    w.converge()
    assert c.refund_status("refund-key-1")["stage"] == "delivered" and len(w.route.delivered) == 1


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
    w = World(tmp_path, limits=IntakeLimits(max_waiting=2, payment_poll_interval_s=0, allow_raw_urls=True))
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
    assert float(out["released_usdc"]) > 0 and w.billing.accounts.available(m) == 100_000
    assert w.queue.claim() is None
    # an unpaid quote expires; the miner submits again later
    m2 = SeedSigner(b"\x06" * 32).ss58_address
    sub2 = w.svc.submit(m2, bundle_url=URL, bundle_sha256=_sha(m2), bundle_bytes=10)
    w.t[0] += 3601
    assert w.svc.status(m2, sub2["submission_id"])["state"] == "expired"


def test_status_polls_follow_the_tao_to_the_queue(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    sub = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)
    assert sub["state"] == "awaiting_payment"
    w.send_tao(sub["payment"])
    states = [w.svc.status(m, sub["submission_id"])["state"] for _ in range(4)]  # bridge, arrive, swap
    assert states[-1] == "queued"


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


def test_services_down_past_max_pauses_park_the_submission_never_close_it(tmp_path):
    w = World(tmp_path)
    w.worker.max_pauses = 1
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    w.worker.ensure_king(w.ctx)
    w.env.detectors = _failing_detectors(w.env.detectors, "gptzero")
    assert w.worker.run_once()["state"] == "paused"
    w.t[0] += 10_000
    assert w.worker.run_once()["state"] == "paused"  # parked: its hold and its place kept, retried later
    row = w.queue.get(sid)
    assert row.state == "paused" and '"parked"' in row.note and w.billing.accounts.hold(sid).status == "open"
    assert "waiting for us" in w.svc.status(m, sid)["evaluation"]["reason"]
    with pytest.raises(Exception, match="cannot be cancelled"):  # the owner decides, not the miner
        w.svc.cancel(m, sid)
    for _ in range(3):  # never retried by itself (it waits for the owner)
        w.t[0] += 86400
        assert w.worker.run_once() is None


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
    assert kr.record_id.startswith(f"{sid}-kingrun")  # one owner run per scoring conditions and H
    assert w.store_for("p-next").status(kr.record_id)["stage"] == "done"


def test_worker_crash_is_retried_then_parked(tmp_path):
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
    assert w.worker.run_once()["state"] == "paused"
    assert '"parked"' in w.queue.get(sid).note and w.billing.accounts.hold(sid).status == "open"
    w.env.runner_for = lambda sub: CountingReplay(w.bundles[sub.bundle_url], pod_seconds=600)
    w.queue.update(sid, state="queued", retry_at=None, pauses=0, attempts=0, note=None)  # parked.py resume
    assert w.worker.run_once()["state"] == "done"


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
    # the first miner king: the only payout hotkey (the validators give it the whole humanizer part)
    assert p["kings"] == [m] and p["king"]["hotkey"] == m
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


def test_our_failure_never_makes_the_miner_resubmit(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    w.worker.ensure_king(w.ctx)
    w.worker.max_attempts = 1

    def boom(sub):
        raise RuntimeError("bug")

    w.env.runner_for = boom
    assert w.worker.run_once()["state"] == "paused" and w.queue.get(sid).state == "paused"  # parked
    w.env.runner_for = lambda sub: CountingReplay(w.bundles[sub.bundle_url], pod_seconds=600)
    w.queue.update(
        sid, state="queued", retry_at=None, pauses=0, attempts=0, note=None
    )  # the owner resumes it
    assert w.worker.run_once()["state"] == "done"  # the same submission, nothing to resubmit


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


# --- paying in TAO from the terminal: submit -> send TAO -> "paid" -----------------------------------
def _pay_world(tmp_path, **kw):
    lim = IntakeLimits(
        payment_poll_interval_s=float("inf"), paid_check_interval_s=0, allow_raw_urls=True, **kw
    )
    return World(tmp_path, limits=lim)


def test_paid_command_follows_the_tao(tmp_path):
    w = _pay_world(tmp_path)
    c = w.client(SeedSigner(b"\x06" * 32))
    sub = c.submit(URL, SHA, 10**9)
    sid, pay = sub["submission_id"], sub["payment"]
    assert (
        pay["then"] == f"POST /v1/submissions/{sid}/paid" and pay["expires_ts"] == sub["quote"]["expires_ts"]
    )
    r = c.paid(sid)
    assert r["state"] == "awaiting_payment" and "no TAO yet" in r["message"]
    w.send_tao(pay)
    r = c.paid(sid)  # the bridge is on its way
    assert r["state"] == "awaiting_payment" and "on its way" in r["message"]
    for _ in range(3):
        r = c.paid(sid)
    assert (
        r["state"] == "queued"
        and "confirmed" in r["message"]
        and w.billing.accounts.hold(sid).status == "open"
    )
    bal = c.balance()["balance_usdc"]
    assert c.paid(sid)["state"] == "queued" and c.balance()["balance_usdc"] == bal  # credited once


def test_tao_sent_in_time_counts_after_the_window(tmp_path):
    w = _pay_world(tmp_path)
    sub = w.svc.submit("5Fhk-late", bundle_url=URL, bundle_sha256=_sha("late"), bundle_bytes=10**9)
    sid = sub["submission_id"]
    w.send_tao(sub["payment"])
    w.svc.paid("5Fhk-late", sid)  # sent in the last minute: the bridge starts ...
    w.t[0] += 3700  # ... and lands after the window
    w.converge()
    assert w.queue.get(sid).state == "queued" and w.billing.accounts.hold(sid).status == "open"


def test_unpaid_submission_expires_and_late_tao_stays_on_the_balance(tmp_path):
    w = _pay_world(tmp_path)
    hk = "5Fhk-unpaid"
    sub = w.svc.submit(hk, bundle_url=URL, bundle_sha256=_sha("u1"), bundle_bytes=10**9)
    sid = sub["submission_id"]
    w.t[0] += 3700
    r = w.svc.paid(hk, sid)
    assert r["state"] == "expired" and "submit again" in r["message"]
    w.send_tao(sub["payment"])  # too late: it becomes USDC on the balance anyway
    for _ in range(4):
        w.billing.collect(hk)
    assert w.billing.accounts.available(hk) > 0
    # an unpaid quote does not use the registration; the balance pays the next one at once
    nxt = w.svc.submit(hk, bundle_url=URL, bundle_sha256=_sha("u2"), bundle_bytes=10**9)
    assert nxt["state"] in ("queued", "awaiting_payment")


def test_tao_is_noticed_without_the_command(tmp_path):
    w = _pay_world(tmp_path)
    sub = w.svc.submit("5Fhk-bg", bundle_url=URL, bundle_sha256=_sha("bg"), bundle_bytes=10**9)
    sid = sub["submission_id"]
    assert w.svc.enqueue_ready() == []
    w.send_tao(sub["payment"])
    w.converge()
    assert w.queue.get(sid).state == "queued"


def test_public_results_show_scores_not_texts_or_money(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
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
    assert set(res) <= {"submission_id", "uid", "hotkey", "name", "finished_ts", "S", "crowned", "S_king", "delta",
                        "lower_bound", "alpha", "levels", "gates", "judge_errors", "detectors"}  # fmt: skip
    assert "charged_cents" not in res and "reasons" not in json.dumps(res)
    assert http.get("/v1/results?period=other").json()["results"] == []
    assert http.get("/v1/results?period=../x").status_code == 422
    assert http.get("/v1/info").headers["access-control-allow-origin"] == "*"


def test_test_mode_daily_cap_and_info(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True,
                                            max_evals_per_day=2))  # fmt: skip
    w.svc.about = {"mode": "test", "network": "test", "netuid": 87}
    info = w.svc.info()
    assert (info["mode"], info["network"], info["netuid"]) == ("test", "test", 87)
    assert info["limits"]["max_evals_per_day"] == 2 and info["limits"]["evals_last_24h"] == 0
    # an unpaid quote holds no place (it costs nothing to make: a few hotkeys could block everyone)
    for hk in ("5Fcap-x", "5Fcap-y", "5Fcap-z"):
        w.svc.submit(hk, bundle_url=URL, bundle_sha256=_sha(hk), bundle_bytes=10)
    assert w.svc.info()["limits"]["evals_last_24h"] == 0
    for hk in ("5Fcap-a", "5Fcap-b"):
        w.pay(hk)
    a = w.svc.submit("5Fcap-a", bundle_url=URL, bundle_sha256=_sha("a"), bundle_bytes=10)
    w.svc.submit("5Fcap-b", bundle_url=URL, bundle_sha256=_sha("b"), bundle_bytes=10)
    w.converge()
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
    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True,
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
    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
    for k in ("a", "b", "c"):
        hk = f"5F{k}par"
        w.pay(hk)
        w.svc.submit(hk, bundle_url=URL, bundle_sha256=_sha(hk), bundle_bytes=10)
    assert w.queue.claim(max_running=2) is not None and w.queue.claim(max_running=2) is not None
    assert w.queue.claim(max_running=2) is None  # two running: the third waits
    assert w.queue.claim(max_running=3) is not None


def test_crown_is_re_decided_when_the_king_changed_during_the_run(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
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


def test_the_king_is_rescored_from_his_saved_outputs_when_conditions_change(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    assert w.worker.run_once()["crowned"]
    path = tmp_path / f"king_run.{w.ctx.period_id}.enc"
    assert KingRun.load(path, KEY).fingerprint is None
    # a deploy changes the scoring code or config: the stored king run no longer matches
    w.env.cfg = replace(w.env.cfg, fingerprint="fp-new")
    CountingReplay.calls = 0
    kr = w.worker.ensure_king(w.ctx)
    assert kr.submission_id == sid and kr.fingerprint == "fp-new"
    assert CountingReplay.calls == 0  # his saved outputs, no new pod
    assert KingRun.load(path, KEY).fingerprint == "fp-new"
    assert w.store_for(w.ctx.period_id).status(sid)["stage"] == "done"  # the miner's own record untouched


def test_a_challenger_pruned_while_waiting_is_not_evaluated_or_charged(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x07" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    before = w.billing.accounts.balance(m)
    w.gone.add(m)  # its UID was pruned (immunity over) while the submission waited in the queue
    CountingReplay.calls = 0
    out = w.worker.run_once()
    assert out == {"submission_id": sid, "state": "failed"} and CountingReplay.calls == 0  # no pod
    assert w.billing.accounts.balance(m) == before and w.billing.accounts.hold(sid).status == "released"
    assert "lost its registration" in w.store_for(w.ctx.period_id).status(sid)["reason"]


def test_a_failing_king_run_is_retried_with_backoff_not_on_every_step(tmp_path):
    """Every king attempt pays the detectors for the canaries: an empty queue polled every few seconds
    must not retry a failing king run each time (it once cost hundreds of dollars in a night)."""
    w = World(tmp_path)
    calls = []
    real = w.worker.ensure_king

    def failing(ctx):
        calls.append(w.t[0])
        if len(calls) < 3:
            raise RuntimeError("zerogpt: canary not scored")
        return real(ctx)

    w.worker.ensure_king = failing
    for _ in range(20):  # the worker's idle loop
        assert w.worker.run_once() is None
        w.t[0] += 5
    assert len(calls) == 1
    w.t[0] = calls[0] + w.worker.retry_s
    w.worker.run_once()
    assert len(calls) == 2
    assert w.worker.prepare_next(w.ctx) is None  # the same backoff for the rotation's preparation
    w.t[0] += w.worker.retry_s  # the wait doubled: not yet
    w.worker.run_once()
    assert len(calls) == 2
    w.t[0] += w.worker.retry_s
    w.worker.run_once()
    assert len(calls) == 3 and not w.worker._king_wait  # done: the next steps reuse the stored run


def test_a_run_killed_with_the_process_is_not_rerun_forever(tmp_path):
    """A process killed mid-run (out of memory) never reaches the crash handler; after a restart the row is
    queued again. Each claim counts an attempt: past max_attempts the row is released, no pod is rented."""
    w = World(tmp_path)
    w.worker.max_attempts = 2
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    w.worker.ensure_king(w.ctx)
    for _ in range(2):  # claimed, then the process dies; the restart puts it back
        assert w.queue.claim().id == sid
        w.queue.recover()
    rented = []
    w.env.runner_for = lambda sub: rented.append(sub) or CountingReplay(w.bundles[URL])
    assert w.worker.run_once()["state"] == "paused" and not rented  # parked, no pod rented in a loop
    assert w.billing.accounts.hold(sid).status == "open"
    for _ in range(5):  # not claimed again before the park interval
        assert w.worker.run_once() is None
    assert not rented


def test_a_failed_period_load_is_not_repeated_on_every_call(tmp_path):
    """A load scores H and the canaries (paid): the worker and every API request ask for the period."""
    from humanizer_eval.intake.worker import CurrentPeriod

    (tmp_path / "p1").mkdir()
    (tmp_path / "CURRENT").write_text("p1\n")
    t, loads = [0.0], []

    def loader(d):
        loads.append(t[0])
        if len(loads) < 3:
            raise RuntimeError("zerogpt: human texts could not be scored")
        return d.name

    cp = CurrentPeriod(tmp_path, loader, retry_s=600, now=lambda: t[0])
    for _ in range(50):
        with pytest.raises(RuntimeError, match="next try in 10m"):
            cp()
        t[0] += 5
    assert len(loads) == 1
    t[0] = 600
    with pytest.raises(RuntimeError, match="next try in 20m"):
        cp()
    t[0] = 1800
    assert cp() == "p1" and len(loads) == 3


def _changed(ctx, d="zerogpt"):
    """A period whose canaries no longer match: conditions.check reports it without calling anyone."""
    return replace(
        ctx, meta={**ctx.meta, "detector_baseline": {"_problem": f"{d}: canary scores moved by 0.032"}}
    )


def test_a_changed_detector_holds_the_period_until_the_owner_rescores(tmp_path):
    """A different answer is not retried: the period is held in a file (a restart keeps it), nothing is
    called, the owner is told what to run; only lifting the hold lets evaluations go on."""
    from humanizer_eval.intake.worker import CONDITIONS_FILE

    w = World(tmp_path)
    (tmp_path / "p").mkdir()
    good = replace(w.ctx, dir=tmp_path / "p")
    w.ctx = _changed(good)
    alerts, calls = [], []
    w.worker._alert = lambda text, dedupe=None: alerts.append(text)
    real = w.worker.ensure_king
    w.worker.ensure_king = lambda ctx: calls.append(1) or real(ctx)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    # the baseline king calls nothing; the first challenger meets the change and the period is held
    assert w.worker.run_once()["state"] == "paused"
    for _ in range(100):  # the worker's idle loop, hours of it
        assert w.worker.run_once() is None
        w.t[0] += 600
    assert len(calls) == 1 and w.queue.get(sid).state == "paused"
    held = json.loads((tmp_path / "p" / CONDITIONS_FILE).read_text())
    assert held["detectors"] == ["zerogpt"]
    assert "rescore_period.py --period p-test --detector zerogpt --yes" in alerts[0]
    assert w.worker.prepare_next(w.ctx) is None and len(calls) == 1  # the rotation's preparation too
    # the owner rescored (the file is moved aside, the period loaded again)
    (tmp_path / "p" / CONDITIONS_FILE).unlink()
    w.ctx = good
    assert w.worker.run_once()["state"] == "done"


def test_a_challenger_that_meets_a_changed_detector_waits_without_using_a_pause(tmp_path):
    w = World(tmp_path)
    w.worker.ensure_king(w.ctx)  # the king was scored before the change
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    good, w.ctx = w.ctx, _changed(w.ctx, "pangram")
    assert w.worker.run_once()["state"] == "paused"
    row = w.queue.get(sid)
    assert row.pauses == 0 and row.attempts == 0 and w.billing.accounts.hold(sid).status == "open"
    for _ in range(20):
        assert w.worker.run_once() is None  # held: not claimed again
        w.t[0] += 3600
    assert w.queue.get(sid).state == "paused"
    w.worker._conditions.clear()  # rescored
    w.ctx = good
    assert w.worker.run_once()["state"] == "done"


def test_a_parallel_crown_is_re_decided_under_a_production_fingerprint(tmp_path):
    """The new king's run keeps the fingerprint it was scored under: the challenger that ran in parallel
    is re-decided against it (it once raised and the row stayed 'running' with the miner charged)."""
    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
    w.env.cfg = replace(w.env.cfg, fingerprint="fp-prod")
    a, b = "5Farace", "5Fbrace"
    for hk in (a, b):
        w.pay(hk)
    sa = w.svc.submit(a, bundle_url=URL, bundle_sha256=_sha(a), bundle_bytes=10)["submission_id"]
    sb = w.svc.submit(b, bundle_url=URL, bundle_sha256=_sha(b), bundle_bytes=10)["submission_id"]
    k0 = w.worker.ensure_king(w.ctx)
    assert w.worker.run_once()["crowned"] is True
    assert KingRun.load(w.env.king_run_path(w.ctx), KEY).fingerprint == "fp-prod"
    real = w.worker.ensure_king
    w.worker.ensure_king = lambda ctx: k0  # B started before A's crown landed
    out = w.worker.run_once()
    w.worker.ensure_king = real
    assert out["submission_id"] == sb and out["crowned"] is False and w.queue.get(sb).state == "done"
    import humanizer_eval.intake.worker as wm

    calls, real_eval = [], wm.evaluate_submission
    wm.evaluate_submission = lambda *a, **k: calls.append(1) or real_eval(*a, **k)
    try:
        w.worker.run_once()
    finally:
        wm.evaluate_submission = real_eval
    assert calls == []  # the new king's stored run is used as it is: no paid rescore
    assert KingState.from_json((tmp_path / "king_state.json").read_text()).current.submission_id == sa


def test_a_crash_after_the_crown_was_applied_is_finished_not_crowned_twice(tmp_path):
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    store = w.store_for(w.ctx.period_id)
    real_close = w.worker._close_crown
    w.worker._close_crown = lambda *a: (_ for _ in ()).throw(SystemExit("killed"))  # dies after apply_crown
    with pytest.raises(SystemExit):
        w.worker.run_once()
    w.worker._close_crown = real_close
    w.queue.recover()
    out = w.worker.run_once()
    assert out["state"] == "done" and w.queue.get(sid).state == "done"
    assert store.status(sid)["crowned"] is True and store.status(sid)["stage"] == "done"
    hist = KingState.from_json((tmp_path / "king_state.json").read_text()).history
    assert [e.submission_id for e in hist].count(sid) == 1


def test_two_concurrent_enqueues_never_release_the_hold_of_a_queued_row(tmp_path):
    """The miner's status poll and the worker's loop both saw 'awaiting_payment'; the second one used to
    release the hold of the row the first had just queued, and the paid submission then failed."""
    import threading

    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    w.pay(m)
    stale = w.queue.get(sid)
    gate = threading.Barrier(2)
    out = []

    def go():
        gate.wait()
        out.append(w.svc._try_enqueue(stale))

    ts = [threading.Thread(target=go) for _ in range(2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert w.queue.get(sid).state == "queued" and w.billing.accounts.hold(sid).status == "open"
    # and sequentially with a stale row, as the unlocked code saw it
    assert w.svc._try_enqueue(stale).state == "queued" and w.billing.accounts.hold(sid).status == "open"


def test_a_step_that_read_the_period_before_a_switch_does_not_run_on_it(tmp_path):
    """The step read period t, then waited (the king lock) while CURRENT moved to t+1: it must put the
    row back instead of evaluating on t's inputs, which the reveal may publish."""
    w = World(tmp_path)
    m = SeedSigner(b"\x05" * 32).ss58_address
    w.pay(m)
    sid = w.svc.submit(m, bundle_url=URL, bundle_sha256=_sha(m), bundle_bytes=10)["submission_id"]
    w.worker._current_period_id = lambda: "p-next"
    assert w.worker.run_once() is None
    row = w.queue.get(sid)
    assert row.state == "queued" and row.attempts == 0
    w.worker._current_period_id = lambda: None
    assert w.worker.run_once()["state"] == "done"


def test_the_baseline_king_is_fixed_without_any_call(tmp_path):
    """Owner decision 2026-11-26: the baseline is 'every output caught, every gate passed' (S = 0), not a
    paid run of the unchanged inputs; it is ready for any period at once."""
    import humanizer_eval.intake.worker as wm

    w = World(tmp_path)
    calls, real = [], wm.evaluate_submission
    wm.evaluate_submission = lambda *a, **k: calls.append(1) or real(*a, **k)
    try:
        kr = w.worker.ensure_king(w.ctx)
    finally:
        wm.evaluate_submission = real
    assert calls == [] and kr.submission_id == "baseline-identity"
    assert set(kr.quality) == {1.0} and all(set(v) == {1.0} for v in kr.scores.values())


def test_a_crown_settled_later_is_decided_against_the_king_of_that_moment(tmp_path):
    """B won against the baseline, but its crown waited (the current king's run was not ready) while A,
    with the same outputs, was crowned. On recovery B must be compared with A (and lose), not crowned on
    its old decision (audit 2, A1)."""

    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
    a, b = "5Farace", "5Fbrace"
    for hk in (a, b):
        w.pay(hk)
    sb = w.svc.submit(b, bundle_url=URL, bundle_sha256=_sha(b), bundle_bytes=10)["submission_id"]
    w.t[0] += 1  # B paid first
    sa = w.svc.submit(a, bundle_url=URL, bundle_sha256=_sha(a), bundle_bytes=10)["submission_id"]
    k0 = w.worker.ensure_king(w.ctx)
    real_eval_king, real_cur = w.worker.ensure_king, w.worker._current_king_run
    # B is evaluated against the baseline; while it runs, A is crowned (simulated by crowning A in between)
    w.worker.ensure_king = lambda ctx: k0
    orig_settle = w.worker._settle_crown

    def settle_after_a_crowned(row, ctx, king, store):
        if (
            row.id == sb
            and not any(
                e.submission_id == sa
                for e in KingState.from_json((tmp_path / "king_state.json").read_text()).history
            )
            if (tmp_path / "king_state.json").exists()
            else row.id == sb
        ):
            w.queue.update(sb, state="paused", retry_at=1e18)
            return None
        return orig_settle(row, ctx, king, store)

    w.worker._settle_crown = settle_after_a_crowned
    assert w.worker.run_once()["submission_id"] == sb  # B ran and was charged; its crown waits
    w.worker.ensure_king = real_eval_king
    w.worker._settle_crown = orig_settle
    # A won too, but B's earlier crown is still pending: A waits (crowns follow the decision order)
    assert w.worker.run_once()["state"] == "paused"
    w.worker._current_king_run = real_cur
    w.queue.update(sb, state="queued", retry_at=None)
    out = w.worker.run_once()
    assert out["submission_id"] == sb and out["crowned"] is True  # B first
    w.queue.update(sa, retry_at=None)
    out = w.worker.run_once()
    assert out["submission_id"] == sa and out["crowned"] is False  # A then compared with B: same outputs
    hist = [e.submission_id for e in KingState.from_json((tmp_path / "king_state.json").read_text()).history]
    assert hist[-1] == sb and sa not in hist


def test_a_rehosted_copy_is_caught_when_the_author_was_private_at_submission(tmp_path):
    """The author's repo was private when it submitted (sha unknown); a copier re-hosts the bundle in its
    own repo (other repo@commit, same sha), commits later and is claimed first. Before the duplicate check
    the earlier rows' unknown shas are read, so the copy is dropped (audit 3)."""
    from types import SimpleNamespace

    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
    sha = _sha("bundle")
    w.queue.create("author", "5Fauthor", bundle_url=URL, bundle_sha256="", bundle_bytes=None, hf_repo="a/r",
                   hf_revision="1" * 40, commit_block=100, reg_block=7, expires_ts=w.t[0] + 3600)  # fmt: skip
    w.queue.create("copy", "5Fcopier", bundle_url=URL, bundle_sha256=sha, bundle_bytes=10, hf_repo="c/r",
                   hf_revision="2" * 40, commit_block=150, reg_block=7)  # fmt: skip
    copy = w.queue.get("copy")
    assert w.queue.committed_earlier(copy) is None  # what the check alone saw before

    class Hub:
        def bundle_info(self, repo, rev):
            return SimpleNamespace(sha256=sha, size=10, private=False, gated=False)

    w.env.hub = Hub()
    for other in w.queue.unknown_earlier(copy):
        info = w.env.hub.bundle_info(other.hf_repo, other.hf_revision)
        w.queue.update(other.id, bundle_sha256=info.sha256, bundle_bytes=info.size)
    assert w.queue.committed_earlier(copy) == "author"


def test_a_pending_winner_is_rescored_when_a_deploy_changed_the_conditions(tmp_path):
    """B won against the baseline; before its crown was settled A was crowned and a deploy changed the
    scoring fingerprint. B is rescored from its saved outputs and re-decided against A (it used to raise
    forever and the row never closed) (audit 4)."""
    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
    w.env.cfg = replace(w.env.cfg, fingerprint="fp-old")
    a, b = "5Farace", "5Fbrace"
    for hk in (a, b):
        w.pay(hk)
    sa = w.svc.submit(a, bundle_url=URL, bundle_sha256=_sha(a), bundle_bytes=10)["submission_id"]
    sb = w.svc.submit(b, bundle_url=URL, bundle_sha256=_sha(b), bundle_bytes=10)["submission_id"]
    k0 = w.worker.ensure_king(w.ctx)
    assert w.worker.run_once()["crowned"] is True  # A
    real = w.worker.ensure_king
    w.worker.ensure_king = lambda ctx: k0  # B started before A's crown
    orig = w.worker._settle_crown

    def deploy_then_settle(row, ctx, king, store):
        w.env.cfg = replace(w.env.cfg, fingerprint="fp-new")  # the deploy, between B's run and its settlement
        w.worker.ensure_king = real
        return orig(row, ctx, king, store)

    w.worker._settle_crown = deploy_then_settle
    out = w.worker.run_once()
    assert out["submission_id"] == sb and out["crowned"] is False and w.queue.get(sb).state == "done"
    hist = [e.submission_id for e in KingState.from_json((tmp_path / "king_state.json").read_text()).history]
    assert hist[-1] == sa and sb not in hist


def test_a_copy_is_dropped_even_when_the_authors_evaluation_failed(tmp_path):
    """The author's run failed on our side (row 'failed', resubmittable): a copier committed later must not
    be evaluated and crowned meanwhile (audit 5)."""
    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
    sha = _sha("bundle")
    w.queue.create("author", "5Fauthor", bundle_url=URL, bundle_sha256=sha, bundle_bytes=10, hf_repo="a/r",
                   hf_revision="1" * 40, commit_block=100, reg_block=7)  # fmt: skip
    w.queue.update("author", state="failed")
    w.queue.create("copy", "5Fcopier", bundle_url=URL, bundle_sha256=sha, bundle_bytes=10, hf_repo="c/r",
                   hf_revision="2" * 40, commit_block=150, reg_block=7)  # fmt: skip
    assert w.queue.committed_earlier(w.queue.get("copy")) == "author"


def test_an_earlier_winner_keeps_its_crown_when_only_our_archive_was_slow(tmp_path):
    """X beat the king first and paid; our archive download failed for now. Y, later, beat the same king.
    Y waits for X's crown; X is crowned when the archive works; Y is then compared with X (audit 5)."""

    class FlakyArchive:
        def __init__(self, root, fail_ids):
            self.root, self.fail_ids, self.done = root, set(fail_ids), set()

        def has(self, sid):
            return sid in self.done

        def path(self, sid):
            return self.root / sid / "bundle.tar"

        def store(self, sid, url, sha, mx):
            if sid in self.fail_ids:
                self.fail_ids.discard(sid)
                raise RuntimeError("bundle could not be downloaded: RemoteProtocolError")
            self.done.add(sid)
            return self.path(sid)

    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
    x, y = "5Fxxx", "5Fyyy"
    for hk in (x, y):
        w.pay(hk)
    sx = w.svc.submit(x, bundle_url=URL, bundle_sha256=_sha(x), bundle_bytes=10)["submission_id"]
    w.env.archive = FlakyArchive(tmp_path / "arch", {sx})
    assert w.worker.run_once()["state"] == "paused"  # X won and paid; the archive failed for now
    w.t[0] += 1
    sy = w.svc.submit(y, bundle_url=URL, bundle_sha256=_sha(y), bundle_bytes=10)["submission_id"]
    assert w.worker.run_once()["state"] == "paused"  # Y won too: it waits for X
    w.t[0] += 601
    for _ in range(3):
        w.worker.run_once()
        w.t[0] += 601
    hist = [e.submission_id for e in KingState.from_json((tmp_path / "king_state.json").read_text()).history]
    assert hist[-1] == sx and sy not in hist  # X kept its crown; Y (same outputs) does not beat X
    assert w.queue.get(sy).state == "done"


def test_end_to_end_first_miner_crowned_then_overtaken_by_a_better_one(tmp_path):
    """The basic path, through the miner API, the queue, billing, the worker, the gates, the detectors, the
    crown test and the signed leaderboard: miner A beats the baseline and becomes king; miner B, better than
    A, is crowned next; the leaderboard pays B 85% and A 10% (5% rolls to B: no third king)."""
    from fastapi.testclient import TestClient

    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
    url_a, url_b = URL.replace("b.tar", "a.tar"), URL.replace("b.tar", "bb.tar")
    ids = [it["id"] for it in w.ctx.items]
    # A: half of its outputs read as human to the detectors; B: all of them
    w.bundles[url_a] = {i: (f"A human wrote this about topic {k}." if k % 2 else f"Generated text number {k}.")
                        for k, i in enumerate(ids)}  # fmt: skip
    w.bundles[url_b] = {i: f"A human wrote this about topic {k}." for k, i in enumerate(ids)}
    a, b = SeedSigner(b"\x0a" * 32), SeedSigner(b"\x0b" * 32)
    for s in (a, b):
        w.pay(s.ss58_address)
    sa = w.client(a).submit(url_a, _sha("a"), 10, name="first")["submission_id"]
    out = w.worker.run_once()
    assert out["submission_id"] == sa and out["state"] == "done" and out["crowned"] is True
    ev = w.svc.status(a.ss58_address, sa)["evaluation"]
    assert ev["S"] > 0 and ev["crowned"] is True and ev["charged_cents"] > 0
    w.t[0] += 1
    sb = w.client(b).submit(url_b, _sha("b"), 10, name="better")["submission_id"]
    out = w.worker.run_once()
    assert out["submission_id"] == sb and out["crowned"] is True
    evb = w.svc.status(b.ss58_address, sb)["evaluation"]
    assert evb["S"] > ev["S"] and evb.get("S_king") is not None
    hist = [e.submission_id for e in KingState.from_json((tmp_path / "king_state.json").read_text()).history]
    assert hist[-2:] == [sa, sb]
    # the signed leaderboard validators read: B 85%, A 10%, the missing third slot's 5% to B
    from nacl.signing import SigningKey

    from humanizer_eval.intake.api import create_app
    from humanizer_eval.publish.leaderboard import build_payload, shares, sign, verify

    st = KingState.from_json((tmp_path / "king_state.json").read_text())
    key = SigningKey(b"\x07" * 32)
    env = sign(build_payload(st, now=w.t[0]), key)
    app = create_app(w.svc, w.verifier, leaderboard=lambda: env)
    got = TestClient(app).get("/v1/leaderboard").json()
    p = verify(got, key.verify_key.encode().hex(), netuid=32, now=w.t[0])
    assert p["kings"] == [b.ss58_address, a.ss58_address]
    assert shares(p) == pytest.approx({b.ss58_address: 0.90, a.ss58_address: 0.10})
    # both were charged for what was used, and given statements
    for s, sid in ((a, sa), (b, sb)):
        assert w.billing.accounts.hold(sid).status == "settled"
        assert w.svc.statement(s.ss58_address, sid)["items"]


def test_periods_list_shows_the_current_periods_commitments_without_a_schedule(tmp_path):
    """A period set up by hand has no rotation schedule: /v1/periods still publishes its commitment and
    Merkle root before anyone is evaluated on it."""
    from fastapi.testclient import TestClient

    w = World(tmp_path)
    pdir = tmp_path / "periods"
    (pdir / "p1").mkdir(parents=True)
    (pdir / "CURRENT").write_text("p1\n")
    (pdir / "p1" / "manifest.json").write_text(json.dumps({"commitment": "c" * 64, "block": 7, "block_hash": "b" * 64,
                                                            "merkle_root": "m" * 64, "n_items": 484,
                                                            "secret_hint": "never"}))  # fmt: skip
    got = TestClient(create_app(w.svc, w.verifier, periods_dir=pdir)).get("/v1/periods").json()
    assert got["current"] == "p1" and got["periods"]["p1"]["commitment"] == "c" * 64
    assert got["periods"]["p1"]["merkle_root"] == "m" * 64 and "secret_hint" not in got["periods"]["p1"]


def test_the_public_queue_names_who_waits_but_not_the_bundle_or_money(tmp_path):
    w = World(tmp_path, limits=IntakeLimits(payment_poll_interval_s=float("inf"), allow_raw_urls=True))
    w.svc.uid = {"5Fa": 3, "5Fb": 7}.get
    for hk in ("5Fa", "5Fb"):
        w.pay(hk)
        w.svc.submit(hk, bundle_url=URL, bundle_sha256=_sha(hk), bundle_bytes=10, name=f"n-{hk}")
        w.t[0] += 1
    q = w.svc.public_queue()["queue"]
    assert [(x["uid"], x["hotkey"], x["state"]) for x in q] == [(3, "5Fa", "waiting"), (7, "5Fb", "waiting")]
    text = json.dumps(q)
    assert URL not in text and "hold" not in text and "sha" not in text


def test_snapshot_uid_is_the_metagraph_position_not_the_sorted_one(tmp_path):
    from humanizer_eval.intake.registry import MetagraphSnapshot, write_snapshot

    p = tmp_path / "mg.json"
    write_snapshot(p, 32, 100, ["5Zlast", "5Afirst", "5Mmid"])  # UID 0, 1, 2 (unsorted on purpose)
    snap = MetagraphSnapshot(p)
    assert [snap.uid(h) for h in ("5Zlast", "5Afirst", "5Mmid", "5Xnone")] == [0, 1, 2, None]
