"""The humanizer competition's share of the weights: split and 85/10/5 in code, the kings from the signed leaderboard.

Loads detection/validator/humanizer_mechanism.py by path, so these tests need only pynacl and numpy
(plus bittensor for the weight conversion), not the detection models.
"""

import importlib.util
import json
import os

import pytest
from nacl.signing import SigningKey

_PATH = os.path.join(os.path.dirname(__file__), "..", "detection", "validator", "humanizer_mechanism.py")
_spec = importlib.util.spec_from_file_location("humanizer_mechanism", _PATH)
hm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hm)

T0 = 1_800_000_000.0
KEY = SigningKey(b"\x03" * 32)
PUB = KEY.verify_key.encode().hex()
HOTKEYS = ["5Fowner", "5Fa", "5Fb", "5Fc"]
NONE = [0.0, 0.0, 0.0, 0.0]


def envelope(kings, *, now=T0, ttl=3 * 3600, netuid=32, mechid=1, key=KEY, seq=None, regs=None):
    p = {
        "schema": hm.SCHEMA,
        "netuid": netuid,
        "mechid": mechid,
        "seq": int(now * 1000) if seq is None else seq,
        "issued_at": now,
        "expires_at": now + ttl,
        "kings": kings,
        "king": None,
        "history": [],
    }
    if regs is not None:
        p["king_registered_at"] = regs
    body = json.dumps(p, sort_keys=True, separators=(",", ":"))
    return {"payload": body, "signature": key.sign(body.encode()).signature.hex(), "public_key": PUB}


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def board(served, tmp_path=None, clock=None, urls=("https://a", "https://b")):
    def fetch(url, timeout):
        v = served.get(url)
        if isinstance(v, Exception) or v is None:
            raise v or ConnectionError("down")
        return v

    return hm.HumanizerLeaderboard(
        list(urls),
        PUB,
        32,
        state_path=None if tmp_path is None else str(tmp_path / "lb.json"),
        fetch=fetch,
        now=clock or Clock(),
    )


def vec(**by_uid):
    out = [0.0] * len(HOTKEYS)
    for k, v in by_uid.items():
        out[int(k[1:])] = v
    return out


def test_shares_are_constants_of_the_code():
    assert hm.HUMANIZER_SHARE == 0.2 and hm.KING_SHARES == (0.85, 0.10, 0.05)
    assert not hasattr(hm, "BURN_KEY")


def test_kings_get_85_10_5_in_their_order():
    d = board({"https://a": envelope(["5Fb", "5Fa", "5Fc"])}).dense(HOTKEYS)
    assert d == pytest.approx(vec(u2=0.85, u1=0.10, u3=0.05))


def test_missing_and_deregistered_slots_go_to_the_current_king():
    assert board({"https://a": envelope(["5Fb"])}).dense(HOTKEYS) == pytest.approx(vec(u2=1.0))
    assert board({"https://a": envelope(["5Fb", "5Fa"])}).dense(HOTKEYS) == pytest.approx(vec(u2=0.90, u1=0.10))
    # the current king left the metagraph: the next one moves up
    d = board({"https://a": envelope(["5Fgone", "5Fa", "5Fc"])}).dense(HOTKEYS)
    assert d == pytest.approx(vec(u1=0.90, u3=0.10))


def test_no_king_means_detection_only():
    assert board({"https://a": envelope([])}).dense(HOTKEYS) is None  # the owner's baseline reigns
    assert board({"https://a": envelope(["5Fgone"])}).dense(HOTKEYS) is None
    assert hm.blend([0.0, 1.0, 1.0, 0.0], None) == [0.0, 1.0, 1.0, 0.0]


def test_never_a_valid_leaderboard_means_detection_only():
    bad = envelope(["5Fa"], key=SigningKey(b"\x09" * 32))
    for served in ({}, {"https://a": bad}, {"https://a": envelope(["5Fa"], netuid=33)}):
        assert board(served).dense(HOTKEYS) is None


def test_no_public_key_means_detection_only():
    b = hm.HumanizerLeaderboard(["https://a"], "", 32, fetch=lambda u, t: envelope(["5Fa"]), now=Clock())
    assert b.dense(HOTKEYS) is None


def test_second_url_used_when_first_is_down():
    d = board({"https://a": ConnectionError("x"), "https://b": envelope(["5Fc"])}).dense(HOTKEYS)
    assert d == vec(u3=1.0)


def test_last_kings_keep_their_share_through_an_outage(tmp_path):
    clock = Clock()
    served = {"https://a": envelope(["5Fa"])}
    b = board(served, tmp_path, clock)
    assert b.dense(HOTKEYS) == vec(u1=1.0)
    served.clear()  # the owner's server goes down
    clock.t += 3600
    assert b.dense(HOTKEYS) == vec(u1=1.0)
    clock.t += 30 * 86400  # long past the expiry: still the last king
    assert b.dense(HOTKEYS) == vec(u1=1.0)
    b2 = board(served, tmp_path, clock)  # a validator restart reloads the saved (expired) envelope
    assert b2.dense(HOTKEYS) == vec(u1=1.0)


def test_an_expired_envelope_is_never_accepted_as_new(tmp_path):
    clock = Clock()
    b = board({"https://a": envelope(["5Fa"])}, tmp_path, clock)
    assert b.dense(HOTKEYS) == vec(u1=1.0)
    clock.t += 10 * 3600
    stale_newer = envelope(["5Fc"], now=T0 + 3600)  # newer seq, but already expired when fetched
    b.urls, b._fetch = ["https://a"], lambda url, timeout: stale_newer
    assert b.dense(HOTKEYS) == vec(u1=1.0)
    with pytest.raises(hm.LeaderboardError, match="expired"):
        hm.verify_envelope(stale_newer, PUB, netuid=32, now=clock.t)


def test_replayed_older_leaderboard_refused(tmp_path):
    clock = Clock()
    new = envelope(["5Fb"], now=T0)
    old = envelope(["5Fa"], now=T0 - 600)  # still within its validity
    served = {"https://a": new}
    b = board(served, tmp_path, clock)
    assert b.dense(HOTKEYS) == vec(u2=1.0)
    served["https://a"] = old
    assert b.dense(HOTKEYS) == vec(u2=1.0)  # kept the newer one
    b2 = board({"https://a": old}, tmp_path, clock)  # even after a restart
    assert b2.dense(HOTKEYS) == vec(u2=1.0)


@pytest.mark.parametrize("kings", [None, "5Fa", ["5Fa", "5Fa"], ["a", "b", "c", "d"], [""], [3]])
def test_malformed_kings_refused(kings):
    with pytest.raises(hm.LeaderboardError, match="kings"):
        hm.verify_envelope(envelope(kings), PUB, netuid=32, now=T0)


def test_an_old_schema_with_weights_is_refused():
    p = json.loads(envelope(["5Fa"])["payload"])
    p["schema"] = "sn32-humanizer-leaderboard/1"
    body = json.dumps(p, sort_keys=True, separators=(",", ":"))
    env = {"payload": body, "signature": KEY.sign(body.encode()).signature.hex(), "public_key": PUB}
    with pytest.raises(hm.LeaderboardError, match="schema"):
        hm.verify_envelope(env, PUB, netuid=32, now=T0)


def test_blend_is_80_20():
    det = [0.0, 2.0, 1.0, 1.0]  # validator scores, not normalized
    hum = [0.0, 0.10, 0.85, 0.05]
    out = hm.blend(det, hum)
    assert sum(out) == pytest.approx(1.0)
    assert out == pytest.approx([0.0, 0.4 + 0.02, 0.2 + 0.17, 0.2 + 0.01])
    assert sum(o - 0.8 * d / 4 for o, d in zip(out, det)) == pytest.approx(hm.HUMANIZER_SHARE)


def test_blend_with_no_detection_scores_and_misaligned_lengths():
    # no detection weight at all: its 80% is spread evenly, the king still gets only his 20%
    a = 0.8 / 3
    assert hm.blend([0.0, 0.0, 0.0], [0.0, 1.0]) == pytest.approx([a, a + 0.2, a])
    assert hm.blend([1.0, 1.0], [0.0, 0.5, 0.5]) == pytest.approx([0.4, 0.5])  # UID 2 not known yet


def test_a_king_that_registered_again_is_not_paid_even_from_a_stale_leaderboard():
    """The evaluator drops a king whose hotkey re-registered (D-66); a validator holding an old envelope
    (the owner's server down) must not keep paying him 85%."""
    env = envelope(["5Fb", "5Fa"], regs=[100, 50])
    regs_now = [1, 50, 100, 7]
    assert board({"https://a": env}).dense(HOTKEYS, regs_now) == pytest.approx(vec(u2=0.90, u1=0.10))
    regs_now[2] = 4321  # 5Fb was deregistered and registered again
    assert board({"https://a": env}).dense(HOTKEYS, regs_now) == pytest.approx(vec(u1=1.0))
    assert board({"https://a": env}).dense(HOTKEYS) == pytest.approx(vec(u2=0.90, u1=0.10))  # no metagraph data


def test_malformed_registrations_refused():
    with pytest.raises(hm.LeaderboardError):
        hm.verify_envelope(envelope(["5Fb"], regs=[1, 2]), PUB, netuid=32, mechid=1, now=T0)


def test_blend_never_gives_the_kings_more_than_their_share():
    nan = float("nan")
    out = hm.blend([nan, nan, nan, nan], [0.0, 0.0, 0.0, 1.0])
    assert out[3] == pytest.approx(0.8 / 4 + 0.2) and sum(out) == pytest.approx(1.0)
    out = hm.blend([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    assert out == pytest.approx([0.2, 0.2, 0.2, 0.4])
