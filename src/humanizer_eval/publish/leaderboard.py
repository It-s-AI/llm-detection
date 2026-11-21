"""Signed leaderboard of the humanizer competition (D-12, D-13, D-92): its share of the validators' weights.

The owner's evaluator signs the payout weights of the current king state with a dedicated ed25519 key;
validators pin its public key, check the signature, the subnet, the competition id ("mechid": 1, a
label, not a chain mechanism since SN32 v4), the expiry and that the
sequence number never goes back, then map hotkeys to UIDs. Anything wrong or missing -> burn.

Envelope (JSON)::

    {"payload": "<the payload as a JSON string, signed byte for byte>",
     "signature": "<hex ed25519 signature of the payload bytes>",
     "public_key": "<hex>"}

Payload::

    {"schema": "sn32-humanizer-leaderboard/1", "netuid": 32, "mechid": 1, "seq": <ms>,
     "issued_at": <unix s>, "expires_at": <unix s>,
     "weights": {"<hotkey>": 0.85, ..., "__burn__": 0.15},
     "king": {...current king entry...}, "history": [...last coronations, newest first...],
     "launch_ramp": {"until": <unix s>, "paid_share": 0.25} | null}

Launch ramp (D-13, owner's decision 2026-11-21): while the owner's baseline is king, everything is burned.
For the first ``LAUNCH_RAMP_S`` after the first miner coronation the miners get only ``LAUNCH_RAMP_PAID`` of
the competition's weights (in the usual 85/10/5 proportions) and the rest is burned, so that an exploit
that slipped through at launch can be found and stopped before it earns the full share.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

from humanizer_eval.scoring.crown import BURN_KEY, IsRegistered, KingState

SCHEMA = "sn32-humanizer-leaderboard/1"
LAUNCH_RAMP_S = 3 * 86400.0
LAUNCH_RAMP_PAID = 0.25


class LeaderboardError(ValueError):
    pass


def load_or_create_signing_key(path: Path | str) -> SigningKey:
    p = Path(path)
    if p.exists():
        return SigningKey(bytes.fromhex(p.read_text().strip()))
    p.parent.mkdir(parents=True, exist_ok=True)
    sk = SigningKey.generate()
    p.write_text(sk.encode().hex())
    p.chmod(0o600)
    return sk


def build_payload(
    state: KingState | None,
    *,
    netuid: int = 32,
    mechid: int = 1,
    ttl_s: float = 3 * 3600,
    history: int = 10,
    now: float | None = None,
    is_registered: IsRegistered | None = None,
    ramp_s: float = LAUNCH_RAMP_S,
    ramp_paid: float = LAUNCH_RAMP_PAID,
) -> dict[str, Any]:
    """Payout weights of the kings that still hold their registration (a deregistered king drops out
    and the others close ranks, D-66); everything to burn without a miner king; during the launch ramp
    the miners' part is scaled to ``ramp_paid`` and the rest goes to burn."""
    now = time.time() if now is None else now
    active = state.active_history(is_registered) if state is not None else []
    weights = state.payout_weights(is_registered) if state is not None and active else {BURN_KEY: 1.0}
    cur = active[-1] if active else None
    first = next((e for e in state.history if not e.is_baseline), None) if state is not None else None
    ramp = None
    if first is not None and now < first.crowned_at.timestamp() + ramp_s:
        ramp = {"until": first.crowned_at.timestamp() + ramp_s, "paid_share": ramp_paid}
        paid = {k: v * ramp_paid for k, v in weights.items() if k != BURN_KEY}
        burn = 1.0 - sum(paid.values())
        weights = {**paid, BURN_KEY: burn} if burn > 1e-12 else paid
    return {
        "schema": SCHEMA,
        "netuid": netuid,
        "mechid": mechid,
        "seq": int(now * 1000),
        "issued_at": now,
        "expires_at": now + ttl_s,
        "weights": weights,
        "king": None if cur is None else cur.to_dict(),
        "history": [e.to_dict() for e in reversed(active[-history:])],
        "launch_ramp": ramp,
    }


def sign(payload: dict[str, Any], key: SigningKey) -> dict[str, str]:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sig = key.sign(body.encode()).signature
    return {"payload": body, "signature": sig.hex(), "public_key": key.verify_key.encode().hex()}


def verify(
    envelope: dict[str, Any],
    public_key_hex: str,
    *,
    netuid: int,
    mechid: int = 1,
    now: float | None = None,
    min_seq: int = 0,
) -> dict[str, Any]:
    """The payload if the envelope is valid for this subnet and mechanism right now (else raises).

    The same rules as the validator's copy in llm-detection (``detection/validator/humanizer_mechanism.py``).
    """
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
    if any(not isinstance(v, int | float) or not math.isfinite(v) or v < 0 for v in vals) or not math.isclose(
        sum(vals), 1.0, abs_tol=1e-6
    ):
        raise LeaderboardError("weights must be non-negative and sum to 1")
    return p
