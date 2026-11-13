"""Mechanism-1 weights from the signed humanizer leaderboard.

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


def envelope(weights, *, now=T0, ttl=3 * 3600, netuid=32, mechid=1, key=KEY, seq=None):
    p = {
        "schema": hm.SCHEMA,
        "netuid": netuid,
        "mechid": mechid,
        "seq": int(now * 1000) if seq is None else seq,
        "issued_at": now,
        "expires_at": now + ttl,
        "weights": weights,
        "king": None,
        "history": [],
    }
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


def test_weights_map_hotkeys_and_burn_to_owner_uid():
    env = envelope({"5Fb": 0.85, "5Fa": 0.10, "5Fgone": 0.05})
    uids, w = board({"https://a": env}).weights(HOTKEYS, "5Fowner")
    assert dict(zip(uids, w)) == pytest.approx({2: 0.85, 1: 0.10, 0: 0.05})  # deregistered -> burn


def test_nothing_valid_burns_everything():
    bad = envelope({"5Fa": 1.0}, key=SigningKey(b"\x09" * 32))
    for served in ({}, {"https://a": bad}, {"https://a": envelope({"5Fa": 1.0}, netuid=33)}):
        uids, w = board(served).weights(HOTKEYS, "5Fowner")
        assert (uids, w) == ([0], [1.0])


def test_second_url_used_when_first_is_down():
    uids, w = board({"https://a": ConnectionError("x"), "https://b": envelope({"5Fc": 1.0})}).weights(
        HOTKEYS, "5Fowner"
    )
    assert (uids, w) == ([3], [1.0])


def test_last_valid_kept_through_outage_until_expiry(tmp_path):
    clock = Clock()
    served = {"https://a": envelope({"5Fa": 1.0})}
    b = board(served, tmp_path, clock)
    assert b.weights(HOTKEYS, "5Fowner") == ([1], [1.0])
    served.clear()  # the owner's server goes down
    clock.t += 3600
    assert b.weights(HOTKEYS, "5Fowner") == ([1], [1.0])
    # a validator restart reloads the saved envelope
    b2 = board(served, tmp_path, clock)
    assert b2.weights(HOTKEYS, "5Fowner") == ([1], [1.0])
    clock.t += 2 * 3600 + 1  # past the expiry: burn
    assert b2.weights(HOTKEYS, "5Fowner") == ([0], [1.0])


def test_replayed_older_leaderboard_refused(tmp_path):
    clock = Clock()
    new = envelope({"5Fb": 1.0}, now=T0)
    old = envelope({"5Fa": 1.0}, now=T0 - 600)  # still within its validity
    served = {"https://a": new}
    b = board(served, tmp_path, clock)
    assert b.weights(HOTKEYS, "5Fowner") == ([2], [1.0])
    served["https://a"] = old
    assert b.weights(HOTKEYS, "5Fowner") == ([2], [1.0])  # kept the newer one
    b2 = board({"https://a": old}, tmp_path, clock)  # even after a restart
    assert b2.weights(HOTKEYS, "5Fowner") == ([2], [1.0])


def test_bad_weight_sums_refused():
    with pytest.raises(hm.LeaderboardError, match="sum to 1"):
        hm.verify_envelope(envelope({"5Fa": 0.6}), PUB, netuid=32, now=T0)
    with pytest.raises(hm.LeaderboardError):
        hm.verify_envelope(envelope({"5Fa": -0.5, "5Fb": 1.5}), PUB, netuid=32, now=T0)


def test_owner_not_registered_raises():
    with pytest.raises(hm.LeaderboardError, match="owner"):
        board({"https://a": envelope({"5Fa": 1.0})}).weights(["5Fa"], "5Fowner")


def test_no_public_key_burns():
    b = hm.HumanizerLeaderboard(["https://a"], "", 32, fetch=lambda u, t: envelope({"5Fa": 1.0}), now=Clock())
    assert b.weights(HOTKEYS, "5Fowner") == ([0], [1.0])


class FakeSubtensor:
    def __init__(self, mechanisms):
        self.mechanisms = mechanisms
        self.calls = []

    def get_mechanism_count(self, netuid):
        return self.mechanisms

    def get_subnet_owner_hotkey(self, netuid):
        return "5Fowner"

    def set_weights(self, **kw):
        self.calls.append(kw)
        return True, "ok"


class Meta:
    hotkeys = HOTKEYS


def test_set_mechanism_weights_only_when_the_mechanism_exists():
    pytest.importorskip("bittensor")
    b = board({"https://a": envelope({"5Fb": 0.9, "5Fa": 0.1})})
    one = FakeSubtensor(1)
    assert hm.set_mechanism_weights(b, one, "wallet", Meta(), 32, 1000) is False and not one.calls
    two = FakeSubtensor(2)
    assert hm.set_mechanism_weights(b, two, "wallet", Meta(), 32, 1000) is True
    call = two.calls[0]
    assert call["mechid"] == 1 and call["netuid"] == 32 and list(call["uids"]) == [1, 2]
    assert call["weights"][1] == 65535 and 7000 < call["weights"][0] < 7400  # u16, max-normalized
