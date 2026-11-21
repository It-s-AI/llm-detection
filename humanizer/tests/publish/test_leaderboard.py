import json
from datetime import UTC, datetime, timedelta

import pytest
from nacl.signing import SigningKey

from humanizer_eval.publish import LeaderboardError, build_payload, load_or_create_signing_key, sign, verify
from humanizer_eval.scoring.crown import BURN_KEY, KingState

T0 = 1_800_000_000.0
KEY = SigningKey(b"\x03" * 32)
PUB = KEY.verify_key.encode().hex()


def _state():
    t = datetime(2026, 11, 1, tzinfo=UTC)
    st = KingState.initial("baseline", "owner", t)
    st.apply_crown("s1", "5Fa", t + timedelta(days=1), 0.02)
    st.apply_crown("s2", "5Fb", t + timedelta(days=2), 0.01)
    return st


def test_round_trip_and_payout_shares():
    env = sign(build_payload(_state(), now=T0), KEY)
    p = verify(env, PUB, netuid=32, now=T0 + 10)
    assert p["weights"] == {"5Fb": 0.85 + 0.05, "5Fa": 0.10}  # the missing third slot rolls to the king
    assert p["king"]["hotkey"] == "5Fb" and [h["submission_id"] for h in p["history"]] == [
        "s2",
        "s1",
        "baseline",
    ]


def test_baseline_king_burns_everything():
    st = KingState.initial("baseline", "owner", datetime(2026, 11, 1, tzinfo=UTC))
    for s in (st, None):
        p = verify(sign(build_payload(s, now=T0), KEY), PUB, netuid=32, now=T0)
        assert p["weights"] == {BURN_KEY: 1.0}


@pytest.mark.parametrize(
    "case", ["payload", "signature", "other_key", "netuid", "mechid", "expired", "rollback"]
)
def test_invalid_envelopes_rejected(case):
    env = sign(build_payload(_state(), now=T0), KEY)
    kw = {"netuid": 32, "now": T0 + 10, "public_key_hex": PUB}
    if case == "payload":
        p = json.loads(env["payload"])
        p["weights"] = {"5Fevil": 1.0}
        env["payload"] = json.dumps(p, sort_keys=True, separators=(",", ":"))
    if case == "signature":
        env["signature"] = "00" * 64
    if case == "other_key":
        env = sign(build_payload(_state(), now=T0), SigningKey(b"\x04" * 32))
    if case == "netuid":
        kw["netuid"] = 33
    if case == "mechid":
        kw["mechid"] = 0
    if case == "expired":
        kw["now"] = T0 + 3 * 3600 + 1
    if case == "rollback":
        kw["min_seq"] = int(T0 * 1000) + 1
    with pytest.raises(LeaderboardError):
        verify(env, **kw)


def test_signed_weights_must_sum_to_one():
    p = build_payload(_state(), now=T0)
    p["weights"] = {"5Fa": 0.5}
    with pytest.raises(LeaderboardError, match="sum to 1"):
        verify(sign(p, KEY), PUB, netuid=32, now=T0)


def test_signing_key_is_created_once_private(tmp_path):
    k1 = load_or_create_signing_key(tmp_path / "k" / "lb.key")
    k2 = load_or_create_signing_key(tmp_path / "k" / "lb.key")
    assert k1.encode() == k2.encode()
    assert (tmp_path / "k" / "lb.key").stat().st_mode & 0o777 == 0o600


def test_deregistered_kings_drop_out_of_the_signed_weights():
    st = _state()  # baseline, s1 (5Fa), s2 (5Fb)

    def reg(hk, at):
        return hk != "5Fb"

    p = verify(sign(build_payload(st, now=T0, is_registered=reg), KEY), PUB, netuid=32, now=T0)
    assert p["weights"] == {"5Fa": 1.0} and p["king"]["submission_id"] == "s1"
    assert [h["submission_id"] for h in p["history"]] == ["s1", "baseline"]


def test_launch_ramp_pays_a_quarter_for_three_days_after_the_first_miner_crown():
    t = datetime(2026, 12, 1, tzinfo=UTC)
    st = KingState.initial("baseline", "owner", t)
    assert build_payload(st, now=t.timestamp() + 86400)["weights"] == {BURN_KEY: 1.0}  # baseline: all burned
    first = t + timedelta(days=2)
    st.apply_crown("s1", "5Fa", first, 0.02)
    p = verify(sign(build_payload(st, now=first.timestamp() + 3600), KEY), PUB, netuid=32,
               now=first.timestamp() + 3600)  # fmt: skip
    assert p["weights"] == pytest.approx({"5Fa": 0.25, BURN_KEY: 0.75})
    assert p["launch_ramp"]["until"] == first.timestamp() + 3 * 86400
    # a second crown inside the window: the usual proportions, still scaled to a quarter
    st.apply_crown("s2", "5Fb", first + timedelta(days=1), 0.01)
    p = build_payload(st, now=(first + timedelta(days=1, hours=1)).timestamp())
    assert p["weights"] == pytest.approx({"5Fb": 0.25 * 0.90, "5Fa": 0.25 * 0.10, BURN_KEY: 0.75})
    # the window is counted from the FIRST miner crown, not from the latest one
    p = build_payload(st, now=first.timestamp() + 3 * 86400 + 1)
    assert p["weights"] == pytest.approx({"5Fb": 0.90, "5Fa": 0.10}) and p["launch_ramp"] is None
