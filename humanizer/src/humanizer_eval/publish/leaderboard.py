"""Signed leaderboard of the humanizer competition (D-12, D-92): who its kings are.

The weights themselves are constants of the validator code (owner's decision 2026-11-25): detection 80%,
humanizer 20%, and inside the humanizer part 85/10/5 to the current king and the two previous distinct
kings, the shares of missing slots to the current king. Nothing is burned. The evaluator only tells the
validators who the kings are: their hotkeys, newest first, signed with a dedicated ed25519 key. Validators
pin its public key, check the signature, the subnet, the competition id ("mechid": 1, a label), the expiry
and that the sequence number never goes back; with no king (the owner's baseline reigns) or no valid
leaderboard ever received, the humanizer part goes to detection.

Envelope (JSON)::

    {"payload": "<the payload as a JSON string, signed byte for byte>",
     "signature": "<hex ed25519 signature of the payload bytes>",
     "public_key": "<hex>"}

Payload::

    {"schema": "sn32-humanizer-leaderboard/2", "netuid": 32, "mechid": 1, "seq": <ms>,
     "issued_at": <unix s>, "expires_at": <unix s>,
     "kings": ["<current king hotkey>", "<previous>", "<one before>"],   # 0..3, distinct
     "king": {...current king entry...} | null, "history": [...last coronations, newest first...]}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from nacl.exceptions import BadSignatureError
from nacl.signing import SigningKey, VerifyKey

from humanizer_eval.scoring.crown import PAYOUT_SHARES_BP, IsRegistered, KingState, payout_shares

SCHEMA = "sn32-humanizer-leaderboard/2"


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
) -> dict[str, Any]:
    """The kings that still hold their registration (a deregistered king drops out and the others close
    ranks, D-66), newest first; none while the owner's baseline is king."""
    now = time.time() if now is None else now
    active = state.active_history(is_registered) if state is not None else []
    kings = state.payout_hotkeys(is_registered) if state is not None else []
    cur = active[-1] if active else None
    return {
        "schema": SCHEMA,
        "netuid": netuid,
        "mechid": mechid,
        "seq": int(now * 1000),
        "issued_at": now,
        "expires_at": now + ttl_s,
        "kings": kings,
        "king": None if cur is None or cur.is_baseline else cur.to_dict(),
        "history": [e.to_dict() for e in reversed(active[-history:])],
    }


def shares(payload: dict[str, Any]) -> dict[str, float]:
    """``{hotkey: share of the humanizer part}`` as the validators compute it (for display)."""
    return payout_shares(list(payload.get("kings") or []))


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
    k = p.get("kings")
    if (
        not isinstance(k, list)
        or len(k) > len(PAYOUT_SHARES_BP)
        or any(not isinstance(h, str) or not h for h in k)
        or len(set(k)) != len(k)
    ):
        raise LeaderboardError("kings must be at most 3 distinct hotkeys")
    return p
