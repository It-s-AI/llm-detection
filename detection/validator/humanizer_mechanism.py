# The MIT License (MIT)
# Copyright © 2026 It's AI
"""Mechanism 1 (humanizer competition): weights from the subnet owner's signed leaderboard.

The evaluator (github.com/ChervovNikita/humanizer_eval) runs miners' humanizers against paid AI
detectors and keeps one king; payouts are 85/10/5 to the current king and the two previous distinct
kings. It publishes those weights signed with a dedicated ed25519 key. A validator:

1. fetches the envelope from the configured URLs (any of them; first valid wins),
2. checks the signature against the pinned public key, the schema, netuid, mechanism, the expiry and
   that the sequence number never goes back (a replayed older leaderboard is refused),
3. maps hotkeys to UIDs; shares of hotkeys not registered and the "__burn__" share go to the subnet
   owner's UID (burn),
4. with nothing valid (owner server down past the expiry, bad signature): everything to burn.

The validator keeps the last valid envelope on disk, so a short outage of the owner's server changes
nothing until that envelope expires.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCHEMA = "sn32-humanizer-leaderboard/1"
BURN_KEY = "__burn__"
MECHID = 1
# Filled in at launch (the public key of the owner's leaderboard signing key and its URLs).
DEFAULT_PUBLIC_KEY = ""
DEFAULT_URLS: Tuple[str, ...] = ()


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
) -> Dict[str, Any]:
    """The payload of a valid envelope (the same rules as humanizer_eval/publish/leaderboard.py)."""
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
    if not (p["issued_at"] - 300 <= now < p["expires_at"]):
        raise LeaderboardError("leaderboard expired or not yet valid")
    if int(p["seq"]) < min_seq:
        raise LeaderboardError("leaderboard is older than one already accepted")
    w = p.get("weights")
    if not isinstance(w, dict) or not w:
        raise LeaderboardError("no weights")
    vals = list(w.values())
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in vals):
        raise LeaderboardError("weights must be non-negative numbers")
    if not math.isclose(sum(vals), 1.0, abs_tol=1e-6):
        raise LeaderboardError("weights must sum to 1")
    return p


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
                                               mechid=self.mechid, now=self._now())
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
        """Fetch a newer valid leaderboard if any URL has one; the current valid payload or None."""
        if not self.public_key_hex:
            self._log("humanizer leaderboard: no public key configured; burning mechanism 1")
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
            self._log("humanizer leaderboard expired; burning mechanism 1")
            self.payload = None
        return self.payload

    # --- weights ---------------------------------------------------------------------------------
    def weights(self, hotkeys: Sequence[str], owner_hotkey: str) -> Tuple[List[int], List[float]]:
        """``(uids, weights)`` for mechanism 1; burn = the subnet owner's UID. Raises if the owner's
        hotkey is not registered (no safe place to send the burn)."""
        if owner_hotkey not in hotkeys:
            raise LeaderboardError("subnet owner hotkey is not in the metagraph")
        burn_uid = list(hotkeys).index(owner_hotkey)
        p = self.refresh()
        index = {hk: uid for uid, hk in enumerate(hotkeys)}
        out: Dict[int, float] = {}
        shares = {BURN_KEY: 1.0} if p is None else p["weights"]
        for hk, w in shares.items():
            uid = index.get(hk, burn_uid) if hk != BURN_KEY else burn_uid
            out[uid] = out.get(uid, 0.0) + float(w)
        uids = sorted(out)
        return uids, [out[u] for u in uids]


def set_mechanism_weights(
    board: HumanizerLeaderboard,
    subtensor: Any,
    wallet: Any,
    metagraph: Any,
    netuid: int,
    version_key: int,
    *,
    mechid: int = MECHID,
    block_time: float = 12.0,
    info: Callable[[str], None] = print,
    error: Callable[[str], None] = print,
) -> bool:
    """Set mechanism-1 weights from the leaderboard; False (and nothing sent) when the subnet has no
    such mechanism yet or anything fails. Mechanism 0 is never touched here.

    Call it BEFORE setting mechanism-0 weights in the same cycle: bittensor's ``set_weights`` checks the
    rate limit against mechanism 0's last update (whatever the mechid) and silently sends nothing if it
    was just updated, while the chain limits each mechanism on its own (localnet, report 11).
    ``block_time`` only matters with commit-reveal (12 s on finney, 0.25 s on a fast-block localnet)."""
    try:
        if subtensor.get_mechanism_count(netuid) <= mechid:
            return False
        from bittensor.utils.weight_utils import convert_weights_and_uids_for_emit
        import numpy as np

        owner = subtensor.get_subnet_owner_hotkey(netuid)
        uids, weights = board.weights(list(metagraph.hotkeys), owner)
        uint_uids, uint_weights = convert_weights_and_uids_for_emit(
            uids=np.array(uids, dtype=np.int64), weights=np.array(weights, dtype=np.float32)
        )
        info(f"humanizer weights (mechanism {mechid}): {dict(zip(uint_uids, uint_weights))}")
        result, msg = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uint_uids,
            weights=uint_weights,
            mechid=mechid,
            block_time=block_time,
            wait_for_finalization=False,
            wait_for_inclusion=False,
            version_key=version_key,
        )
        if result is True:
            info("humanizer set_weights on chain successfully!")
            return True
        error(f"humanizer set_weights failed {msg}")
    except Exception as e:
        error(f"humanizer weights not set: {e}")
    return False
