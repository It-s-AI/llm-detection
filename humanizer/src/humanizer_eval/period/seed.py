"""Commit-reveal seed for a period (D-57).

In period t-1 the owner publishes c_t = sha256(secret_t). The period seed is
seed_t = sha256(secret_t || blockhash(B_t)), where B_t is a future block, so the
owner cannot grind seeds and miners cannot predict them. Sub-seeds for every
random choice are derived deterministically so a release can be re-verified.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets


def make_secret() -> bytes:
    return secrets.token_bytes(32)


def commitment(secret: bytes) -> str:
    return hashlib.sha256(secret).hexdigest()


def verify_commitment(commit_hex: str, secret: bytes) -> bool:
    return hmac.compare_digest(commitment(secret), commit_hex)


def derive_seed(secret: bytes, block_hash_hex: str) -> bytes:
    bh = bytes.fromhex(block_hash_hex.removeprefix("0x"))
    return hashlib.sha256(secret + bh).digest()


def subseed(seed: bytes, label: str) -> int:
    """Deterministic 63-bit integer seed for a named random choice (numpy-compatible)."""
    d = hmac.new(seed, label.encode(), hashlib.sha256).digest()
    return int.from_bytes(d[:8], "big") >> 1
