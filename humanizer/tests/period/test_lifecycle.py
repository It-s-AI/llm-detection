import contextlib
import json
import stat

import httpx
import pytest

from humanizer_eval.period.chain import Chain
from humanizer_eval.period.lifecycle import Lifecycle, LifecycleConfig, next_period_id
from humanizer_eval.period.seed import commitment

DAY = 86400.0


class FakeChain:
    def __init__(self, head=1000):
        self.h = head

    def head(self):
        return self.h

    def block_hash(self, n):
        return f"{n:064x}"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _p1(d):
    (d / "p1").mkdir(parents=True)
    (d / "p1" / "manifest.json").write_text(json.dumps({"period_id": "p1", "commitment": "c1", "block": 5,
                                                        "block_hash": "ab" * 32, "n_items": 3}))  # fmt: skip
    (d / "CURRENT").write_text("p1\n")


def _world(tmp_path, **kw):
    kw.setdefault("cfg", LifecycleConfig(period_s=4 * DAY))  # the scenarios below use 4-day periods
    d = tmp_path / "periods"
    _p1(d)
    clock, chain = Clock(), FakeChain()
    built, alerts, published, revealed = [], [], [], []

    def builder(pid, secret, bh, block):
        (d / pid).mkdir()
        (d / pid / "secret.hex").write_text(secret.hex())
        (d / pid / "LOADED").touch()  # the server loads it (intake_server's loader writes this)
        built.append((pid, bh, block))
        return {"commitment": commitment(secret), "block_hash": bh, "merkle_root": "mr", "n_items": 7,
                "cost_usd": 3.2}  # fmt: skip

    def revealer(pid):
        revealed.append(pid)
        return {"submissions": 2}

    lc = Lifecycle(d, chain, builder, revealer=revealer, now=clock,
                   alert=lambda text, dedupe=None: alerts.append(text),
                   publish=lambda kind, pid: published.append((kind, pid)), **kw)  # fmt: skip
    return lc, d, clock, chain, built, alerts, published, revealed


def test_next_period_id():
    assert next_period_id("p1") == "p2" and next_period_id("pilot-9") == "pilot-10"
    with pytest.raises(ValueError):
        next_period_id("p")


def test_full_rotation(tmp_path):
    lc, d, clock, chain, built, alerts, published, revealed = _world(tmp_path)
    cfg = lc.cfg
    # first pass: p1 adopted, the commitment for p2 published with a seed block 3.5 days ahead
    assert lc.step() == ["adopted p1 as the current period", "committed p2 (seed block 26200)"]
    sch = lc.schedule()
    rec = sch["periods"]["p2"]
    assert sch["current"] == "p1" and sch["next"] == "p2"
    assert rec["seed_block"] == 1000 + int((cfg.period_s - cfg.build_lead_s) / cfg.block_s)
    secret_file = d / ".p2.secret"
    assert stat.S_IMODE(secret_file.stat().st_mode) == 0o600
    assert rec["commitment"] == commitment(bytes.fromhex(secret_file.read_text()))
    assert "secret" not in json.dumps(sch)  # the schedule is public
    assert published == [("schedule", "p2")]
    # nothing happens until B is final
    clock.t += 3 * DAY
    chain.h = rec["seed_block"] + cfg.finality_blocks - 1
    assert lc.step() == [] and not built
    chain.h += 1
    assert lc.step() == ["built p2"]
    assert built == [("p2", f"{rec['seed_block']:064x}", rec["seed_block"])]
    assert (d / "NEXT").read_text().strip() == "p2" and not secret_file.exists()
    # the period is not over yet; then over but not READY: wait (and alert) within the grace time
    assert lc.step() == []
    clock.t += 1 * DAY + 60  # over, not READY, within the grace time
    assert lc.step() == [] and any("not ready" in a for a in alerts)
    (d / "p2" / "READY").write_text("{}")
    b3 = chain.h + int((cfg.period_s - cfg.build_lead_s) / cfg.block_s)
    assert lc.step() == ["switched p1 -> p2", f"committed p3 (seed block {b3})"]
    sch = lc.schedule()
    assert (d / "CURRENT").read_text().strip() == "p2" and not (d / "NEXT").exists()
    assert sch["periods"]["p2"]["started_at"] == clock.t and not sch["periods"]["p1"].get("revealed_at")
    assert sch["periods"]["p3"]["planned_start"] == clock.t + cfg.period_s
    # never in the switch's pass: a worker step that read p1 just before may still claim a run on it
    clock.t += cfg.reveal_grace_s
    assert lc.step() == ["revealed p1"]
    assert lc.schedule()["periods"]["p1"]["revealed_at"]
    assert revealed == ["p1"] and ("reveal", "p1") in published


def test_switch_without_ready_after_the_grace_time(tmp_path):
    lc, _, clock, chain, *_ = _world(tmp_path, cfg=LifecycleConfig(period_s=4 * DAY, ready_grace_s=3600))
    lc.step()
    chain.h = 10**6
    clock.t += 3.6 * DAY
    assert lc.step() == ["built p2"]
    clock.t += 0.4 * DAY + 3599
    assert lc.step() == []
    clock.t += 2
    assert lc.step()[0] == "switched p1 -> p2"
    assert lc.schedule()["periods"]["p2"]["ready"] is False


def test_reveal_waits_for_running_evaluations_of_that_period(tmp_path):
    running = {"p1": 1}
    lc, d, clock, chain, *_, revealed = _world(tmp_path, running=lambda p: running.get(p, 0))
    lc.step()
    chain.h, clock.t = 10**6, clock.t + 4 * DAY
    lc.step()
    (d / "p2" / "READY").write_text("{}")
    lc.step()
    clock.t += 3600
    assert "revealed p1" not in lc.step() and revealed == []
    running["p1"] = 0
    assert lc.step() == ["revealed p1"]


def test_a_failed_build_alerts_and_is_retried_an_hour_later(tmp_path):
    lc, d, clock, chain, _, alerts, *_ = _world(tmp_path)
    lc.step()
    chain.h = 10**6
    good = lc.builder
    lc.builder = lambda *a: (_ for _ in ()).throw(RuntimeError("openrouter down"))
    with pytest.raises(RuntimeError):
        lc.step()
    assert any("building p2 failed" in a for a in alerts)
    assert (d / ".p2.secret").exists() and lc.schedule()["next"] == "p2"
    lc.builder = good
    assert "built p2" not in lc.step()  # the next timer pass (15 min): not yet, a build is paid
    clock.t += 3600
    assert lc.step() == ["built p2"]


def test_a_build_that_keeps_failing_stops_after_build_tries(tmp_path):
    from humanizer_eval.period.lifecycle import BUILD_TRIES

    lc, _, clock, chain, _, alerts, *_ = _world(tmp_path)
    lc.step()
    chain.h = 10**6
    calls = []
    lc.builder = lambda *a: calls.append(1) or (_ for _ in ()).throw(RuntimeError("openrouter down"))
    for _ in range(200):  # ~2 days of timer passes
        with contextlib.suppress(RuntimeError):
            lc.step()
        clock.t += 900
    assert len(calls) == BUILD_TRIES and any("giving up" in a for a in alerts)
    assert "tries used" in alerts[-1]  # each later pass says the rotation waits for the owner


def test_a_manifest_that_does_not_match_the_commitment_is_refused(tmp_path):
    lc, _, _, chain, *_ = _world(tmp_path)
    lc.step()
    chain.h = 10**6
    lc.builder = lambda pid, secret, bh, block: {"commitment": "other", "block_hash": bh}
    with pytest.raises(RuntimeError, match="does not match"):
        lc.step()


def test_late_commitment_keeps_a_minimum_lead(tmp_path):
    lc, d, _, chain, *_ = _world(tmp_path)
    lc.step()
    sch = lc.schedule()
    # p1 started long ago (adopted late): the seed block still lies min_lead_blocks ahead
    sch["periods"]["p1"]["started_at"] -= 10 * DAY
    sch["next"] = None
    del sch["periods"]["p2"]
    (d / "schedule.json").write_text(json.dumps(sch))
    (d / ".p2.secret").unlink()
    lc.step()
    assert lc.schedule()["periods"]["p2"]["seed_block"] == chain.h + lc.cfg.min_lead_blocks


def test_chain_json_rpc():
    def handler(req):
        body = json.loads(req.content)
        res = {"chain_getFinalizedHead": "0xfeed", "chain_getHeader": {"number": "0x915dc0"},
               "chain_getBlockHash": "0x" + "56" * 32}[body["method"]]  # fmt: skip
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": res})

    c = Chain("https://rpc.test", http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.head() == 0x915DC0
    assert c.block_hash(9526464) == "56" * 32


class FakeNotary:
    def __init__(self, chain):
        self.chain, self.posts, self.fail = chain, [], False

    def post(self, text):
        if self.fail:
            raise RuntimeError("node unreachable")
        self.posts.append(text)
        return {"block": self.chain.h + 2, "signer_pubkey": "ab" * 32, "signer": "5X", "extrinsic_index": 3}


def test_notary_remarks_gate_the_build_and_the_switch(tmp_path):
    lc, d, clock, chain, _, alerts, *_ = _world(tmp_path)
    lc.notary = FakeNotary(chain)
    lc.notary.fail = True
    lc.step()  # committed, but the remark failed: alert, retried on the next pass
    assert "notary_commitment" not in lc.schedule()["periods"]["p2"] and any("notary" in a for a in alerts)
    lc.notary.fail = False
    assert lc.step() == ["commitment of p2 on chain (block 1002)"]
    rec = lc.schedule()["periods"]["p2"]
    assert json.loads(rec["notary_commitment"]["text"].split(" ", 1)[1]) == {
        "kind": "commitment", "netuid": 32, "period": "p2", "commitment": rec["commitment"],
        "seed_block": rec["seed_block"]}  # fmt: skip
    # built after B; NEXT only once the Merkle root is on chain
    chain.h, clock.t = rec["seed_block"] + 20, clock.t + 3.6 * DAY
    lc.notary.fail = True
    assert lc.step() == ["built p2"] and not (d / "NEXT").exists()
    clock.t += 0.5 * DAY
    (d / "p2" / "READY").write_text("{}")
    assert lc.step() == [] and any("Merkle root of p2 is not on chain" in a for a in alerts)
    lc.notary.fail = False
    assert lc.step() == [f"Merkle root of p2 on chain (block {chain.h + 2})"]
    out = lc.step()  # the next pass switches, commits p3 and puts it on chain, reveals p1
    assert out[0] == "switched p1 -> p2" and out[2].startswith("commitment of p3 on chain")
    assert '"merkle_root":"mr"' in lc.notary.posts[1] and '"n_items":7' in lc.notary.posts[1]


def test_a_commitment_that_misses_its_seed_block_is_abandoned_and_redone(tmp_path):
    lc, _, _, chain, built, alerts, *_ = _world(tmp_path)
    lc.notary = FakeNotary(chain)
    lc.notary.fail = True
    lc.step()
    first = lc.schedule()["periods"]["p2"]["commitment"]
    chain.h = lc.schedule()["periods"]["p2"]["seed_block"] + 50
    lc.notary.fail = False
    assert lc.step() == [] and not built  # never built from a commitment that was not on chain in time
    assert any("did not reach the chain" in a for a in alerts)
    assert lc.recommit("remark missed the seed block") == "p2"
    out = lc.step()
    assert out[0].startswith("committed p2") and out[1].startswith("commitment of p2 on chain")
    sch = lc.schedule()
    assert sch["periods"]["p2"]["commitment"] != first
    assert sch["abandoned"][0]["commitment"] == first and sch["abandoned"][0]["reason"]


def test_chain_free_balance_and_block_bodies():
    pk = "b853864e1d747a09c664e15b5b3a9a30a1298e4af270c985122a2fefeab5ec7f"
    free = (996106239).to_bytes(16, "little")
    info = "0x" + (b"\x01\x00\x00\x00" * 4 + free + b"\x00" * 48).hex()
    seen = {}

    def handler(req):
        body = json.loads(req.content)
        seen[body["method"]] = body["params"]
        res = {"state_getStorage": info, "chain_getBlockHash": "0x" + "aa" * 32,
               "chain_getBlock": {"block": {"extrinsics": ["0x01", "0x02"]}}}[body["method"]]  # fmt: skip
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": res})

    c = Chain("https://rpc.test", http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.free_balance(pk) == 0.996106239
    assert seen["state_getStorage"][0].endswith(pk)  # blake2_128_concat: hash then the key itself
    assert c.block_extrinsics(5) == ["0x01", "0x02"] and seen["chain_getBlock"] == ["0x" + "aa" * 32]


def test_a_commitment_on_chain_cannot_be_redone_once_its_seed_block_is_reached(tmp_path):
    """Otherwise the owner, who knows the seed after B, could re-roll the period's dataset."""
    lc, _, _, chain, *_ = _world(tmp_path)
    lc.notary = FakeNotary(chain)
    lc.step()
    lc.step()
    rec = lc.schedule()["periods"]["p2"]
    assert rec.get("notary_commitment")
    chain.h = rec["seed_block"] + 1
    with pytest.raises(RuntimeError, match="cannot be redone"):
        lc.recommit("try again")


def test_never_switches_to_a_period_the_server_could_not_load(tmp_path):
    """Past the grace time without READY the rotation switches only to a period the server has loaded at
    least once: a period that never loads would stop every evaluation."""
    lc, d, clock, chain, *_ = _world(tmp_path, cfg=LifecycleConfig(period_s=4 * DAY, ready_grace_s=3600))
    lc.step()
    chain.h = 10**6
    clock.t += 3.6 * DAY
    assert lc.step() == ["built p2"]
    (d / "p2" / "LOADED").unlink()  # the server never managed to load p2
    clock.t += 0.4 * DAY + 7200
    assert lc.step() == []
    assert (d / "CURRENT").read_text().strip() == "p1"
    (d / "p2" / "LOADED").touch()
    assert lc.step()[0] == "switched p1 -> p2"


def test_a_period_lasts_seven_days_by_default():
    assert LifecycleConfig().period_s == 7 * DAY  # owner's decision 2026-11-27 (was 4)
