import json
from datetime import UTC, datetime, timedelta

import pytest
from nacl.signing import SigningKey

from humanizer_eval.publish import (
    LeaderboardError,
    build_payload,
    load_or_create_signing_key,
    shares,
    sign,
    verify,
)
from humanizer_eval.scoring.crown import KingState

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
    assert p["kings"] == ["5Fb", "5Fa"]  # newest first, the baseline left out
    assert shares(p) == {"5Fb": 0.85 + 0.05, "5Fa": 0.10}  # the missing third slot rolls to the king
    assert "weights" not in p and "launch_ramp" not in p
    assert p["king"]["hotkey"] == "5Fb" and [h["submission_id"] for h in p["history"]] == [
        "s2",
        "s1",
        "baseline",
    ]


def test_baseline_king_has_no_kings_nothing_burned():
    st = KingState.initial("baseline", "owner", datetime(2026, 11, 1, tzinfo=UTC))
    for s in (st, None):
        p = verify(sign(build_payload(s, now=T0), KEY), PUB, netuid=32, now=T0)
        assert p["kings"] == [] and p["king"] is None and shares(p) == {}


@pytest.mark.parametrize(
    "case", ["payload", "signature", "other_key", "netuid", "mechid", "expired", "rollback"]
)
def test_invalid_envelopes_rejected(case):
    env = sign(build_payload(_state(), now=T0), KEY)
    kw = {"netuid": 32, "now": T0 + 10, "public_key_hex": PUB}
    if case == "payload":
        p = json.loads(env["payload"])
        p["kings"] = ["5Fevil"]
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


@pytest.mark.parametrize("kings", [None, "5Fa", ["5Fa", "5Fa"], ["a", "b", "c", "d"], [""], [3]])
def test_signed_kings_must_be_at_most_three_distinct_hotkeys(kings):
    p = build_payload(_state(), now=T0)
    p["kings"] = kings
    with pytest.raises(LeaderboardError, match="kings"):
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
    assert p["kings"] == ["5Fa"] and p["king"]["submission_id"] == "s1"
    assert [h["submission_id"] for h in p["history"]] == ["s1", "baseline"]
