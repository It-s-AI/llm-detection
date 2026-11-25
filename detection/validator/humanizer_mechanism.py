# The MIT License (MIT)
# Copyright © 2026 It's AI
"""The humanizer competition's share of the validator's weights (one weight vector for both competitions).

SN32 runs two competitions on one set of weights. The split is a constant of this code, the same for every
validator running it: detection ``1 - HUMANIZER_SHARE`` (80%), humanizer ``HUMANIZER_SHARE`` (20%). Inside the
humanizer part the current king gets 85%, the two previous kings with distinct hotkeys 10% and 5%
(``KING_SHARES``); the share of a missing slot goes to the current king. Nothing is burned.

The only thing a validator learns from outside is WHO the kings are: the subnet owner's evaluator runs the
miners' humanizers against paid AI detectors and publishes the kings' hotkeys, newest first, in a
leaderboard signed with a dedicated ed25519 key. A validator:

1. fetches the envelope from the configured URLs (any of them; first valid wins),
2. checks the signature against the pinned public key, the schema, netuid, competition id, the expiry and
   that the sequence number never goes back (a replayed older leaderboard is refused),
3. drops kings whose hotkeys are not in the metagraph and gives the rest 85/10/5 by their order,
4. blends: ``(1 - HUMANIZER_SHARE) * detection + HUMANIZER_SHARE * humanizer``.

The validator keeps the last valid envelope on disk. When no fresh one can be had (the owner's server is
down, a bad signature), the kings of the last valid leaderboard keep their share, even past its expiry.
With no king (the owner's baseline reigns, or none of the kings is registered) or no valid leaderboard ever
received, the detection weights are used alone: the humanizer part goes to detection.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCHEMA = "sn32-humanizer-leaderboard/2"
#: competition id in the signed payload ("mechid": 1 = the humanizer competition); not a chain mechanism
MECHID = 1
#: share of the weights that goes to the humanizer competition; the rest stays with detection
HUMANIZER_SHARE = 0.2
#: the current king, the previous distinct king, the one before; a missing slot goes to the current king
KING_SHARES: Tuple[float, ...] = (0.85, 0.10, 0.05)
# the public key of the subnet owner's leaderboard signing key and where the leaderboard is served
DEFAULT_PUBLIC_KEY = "68e3af98273b94279c06381deaaef3811723304f238478e489d7f95fc7d880cf"
DEFAULT_URLS: Tuple[str, ...] = ("https://188.245.168.78/v1/leaderboard",)


class LeaderboardError(ValueError):
    pass


def verify_envelope(
    envelope: Dict[str, Any],
    public_key_hex: str,
    *,
    netuid: int,
    mechid: int = MECHID,
    now: Optional[float] = None,
    min_seq: int = 0,
    allow_expired: bool = False,
) -> Dict[str, Any]:
    """The payload of a valid envelope (the same rules as humanizer/src/humanizer_eval/publish/leaderboard.py).
    ``allow_expired``: the last accepted envelope, kept on disk, stays usable after its expiry."""
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey

    now = time.time() if now is None else now
    try:
        body = envelope["payload"]
        VerifyKey(bytes.fromhex(public_key_hex)).verify(body.encode(), bytes.fromhex(envelope["signature"]))
        p = json.loads(body)
    except (KeyError, TypeError, ValueError, BadSignatureError) as e:
        raise LeaderboardError(f"bad envelope or signature: {type(e).__name__}") from e
    if p.get("schema") != SCHEMA or p.get("netuid") != netuid or p.get("mechid") != mechid:
        raise LeaderboardError("leaderboard is for another schema, subnet or mechanism")
    if now < p["issued_at"] - 300 or (now >= p["expires_at"] and not allow_expired):
        raise LeaderboardError("leaderboard expired or not yet valid")
    if int(p["seq"]) < min_seq:
        raise LeaderboardError("leaderboard is older than one already accepted")
    k = p.get("kings")
    if (
        not isinstance(k, list)
        or len(k) > len(KING_SHARES)
        or any(not isinstance(h, str) or not h for h in k)
        or len(set(k)) != len(k)
    ):
        raise LeaderboardError("kings must be at most 3 distinct hotkeys")
    return p


def king_shares(kings: Sequence[str], hotkeys: Sequence[str]) -> Optional[List[float]]:
    """One value per UID summing to 1: ``KING_SHARES`` over the kings that are in the metagraph, in their
    order, the shares of missing slots to the first of them. None when no king is registered."""
    index = {hk: uid for uid, hk in enumerate(hotkeys)}
    present = [hk for hk in dict.fromkeys(kings) if hk in index][: len(KING_SHARES)]
    if not present:
        return None
    out = [0.0] * len(hotkeys)
    for i, hk in enumerate(present):
        out[index[hk]] += KING_SHARES[i]
    out[index[present[0]]] += sum(KING_SHARES[len(present):])
    return out


def _http_get_json(url: str, timeout: float) -> Dict[str, Any]:
    import requests

    r = requests.get(url, timeout=timeout, headers={"User-Agent": "sn32-validator"})
    r.raise_for_status()
    if len(r.content) > 1024 * 1024:
        raise LeaderboardError("leaderboard too large")
    return r.json()


class HumanizerLeaderboard:
    def __init__(
        self,
        urls: Sequence[str],
        public_key_hex: str,
        netuid: int,
        *,
        state_path: Optional[str] = None,
        mechid: int = MECHID,
        timeout: float = 10.0,
        fetch: Callable[[str, float], Dict[str, Any]] = _http_get_json,
        now: Callable[[], float] = time.time,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.urls = [u for u in urls if u]
        self.public_key_hex = public_key_hex
        self.netuid = netuid
        self.mechid = mechid
        self.state_path = state_path
        self.timeout = timeout
        self._fetch = fetch
        self._now = now
        self._log = log or (lambda m: None)
        self.envelope: Optional[Dict[str, Any]] = None
        self.payload: Optional[Dict[str, Any]] = None
        self.seq = 0
        self._load()

    # --- persistence of the last valid envelope -------------------------------------------------
    def _load(self) -> None:
        if not self.state_path or not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                st = json.load(f)
            self.seq = int(st.get("seq", 0))
            env = st.get("envelope")
            if env is not None:
                self.payload = verify_envelope(env, self.public_key_hex, netuid=self.netuid,
                                               mechid=self.mechid, now=self._now(), allow_expired=True)
                self.envelope = env
        except (OSError, ValueError, LeaderboardError) as e:
            self._log(f"humanizer leaderboard state not used: {e}")

    def _save(self) -> None:
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"seq": self.seq, "envelope": self.envelope}, f)
        os.replace(tmp, self.state_path)

    # --- refresh ---------------------------------------------------------------------------------
    def refresh(self) -> Optional[Dict[str, Any]]:
        """Fetch a newer valid leaderboard if any URL has one. Returns the newest accepted payload, even
        past its expiry (its kings keep their share through an outage), or None if there never was one."""
        if not self.public_key_hex:
            self._log("humanizer leaderboard: no public key configured")
            return None
        for url in self.urls:
            try:
                env = self._fetch(url, self.timeout)
                p = verify_envelope(env, self.public_key_hex, netuid=self.netuid, mechid=self.mechid,
                                    now=self._now(), min_seq=self.seq)
            except Exception as e:  # network, JSON, signature, stale: try the next URL
                self._log(f"humanizer leaderboard from {url} not used: {e}")
                continue
            if int(p["seq"]) >= self.seq:
                self.seq, self.envelope, self.payload = int(p["seq"]), env, p
                self._save()
            break
        if self.payload is not None and self._now() >= self.payload["expires_at"]:
            self._log("humanizer leaderboard: no fresh one; the kings of the last valid one keep their share")
        elif self.payload is None:
            self._log("humanizer leaderboard: none received yet; its share goes to detection")
        return self.payload

    # --- weights ---------------------------------------------------------------------------------
    def dense(self, hotkeys: Sequence[str]) -> Optional[List[float]]:
        """The humanizer part as one value per UID (``len(hotkeys)`` values summing to 1), or None when
        there is no king to pay (no valid leaderboard ever, the owner's baseline reigns, or no king is
        registered): then the detection weights are used alone."""
        p = self.refresh()
        return None if p is None else king_shares(p["kings"], hotkeys)


def blend(
    detection: Sequence[float], humanizer: Optional[Sequence[float]], share: float = HUMANIZER_SHARE
) -> List[float]:
    """One weight vector for both competitions: ``(1 - share) * detection + share * humanizer``.

    ``detection`` is normalized here (all zero: the detection part is empty and only the humanizer share
    is set). ``humanizer`` None (no king, or disabled) leaves the detection weights unchanged. The
    humanizer vector is aligned to the detection one by UID: missing UIDs count as 0, extra ones are cut."""
    det = [max(0.0, float(x)) for x in detection]
    if humanizer is None:
        return det
    total = sum(det)
    det = [x / total for x in det] if total > 0 else det
    hum = [float(humanizer[i]) if i < len(humanizer) else 0.0 for i in range(len(det))]
    return [(1.0 - share) * d + share * h for d, h in zip(det, hum)]
