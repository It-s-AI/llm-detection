"""Miner requests are signed with the miner's Bittensor hotkey (sr25519, SS58 prefix 42).

The signed message binds the method, the path, a timestamp, a one-time nonce and the body hash:

    humeval-v1\\n{METHOD}\\n{path}\\n{timestamp}\\n{nonce}\\n{sha256(body) hex}

Headers: ``X-Hotkey`` (SS58), ``X-Timestamp`` (unix seconds), ``X-Nonce`` (8-64 chars [A-Za-z0-9_-]),
``X-Signature`` (hex, 64 bytes). A signature is accepted within ``max_skew`` seconds of our clock and once
only (the nonce is remembered for twice that window), so a captured request cannot be replayed.

Miners sign with ``bittensor_wallet``: ``wallet.hotkey.sign(message)`` (see ``intake/client.py``).
The Polkadot.js wrapping ``<Bytes>...</Bytes>`` is accepted too.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

SS58_PREFIX = 42  # Bittensor (generic Substrate)
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_IDX = {c: i for i, c in enumerate(_B58)}
_NONCE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
DOMAIN = "humeval-v1"


class AuthError(ValueError):
    """The request is not signed by the hotkey it names (HTTP 401)."""


def _b58decode(s: str) -> bytes:
    n = 0
    for c in s:
        if c not in _B58_IDX:
            raise AuthError("hotkey is not a valid SS58 address")
        n = n * 58 + _B58_IDX[c]
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + raw


def _b58encode(b: bytes) -> str:
    n = int.from_bytes(b, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = _B58[r] + out
    pad = len(b) - len(b.lstrip(b"\x00"))
    return "1" * pad + out


def _checksum(payload: bytes) -> bytes:
    return hashlib.blake2b(b"SS58PRE" + payload, digest_size=64).digest()[:2]


def ss58_decode(address: str, prefix: int = SS58_PREFIX) -> bytes:
    """32-byte public key of an SS58 address with the given (one-byte) network prefix."""
    if not 40 <= len(address) <= 60:
        raise AuthError("hotkey is not a valid SS58 address")
    raw = _b58decode(address)
    if len(raw) != 35 or raw[0] != prefix or _checksum(raw[:33]) != raw[33:]:
        raise AuthError("hotkey is not a valid SS58 address")
    return raw[1:33]


def ss58_encode(public_key: bytes, prefix: int = SS58_PREFIX) -> str:
    if len(public_key) != 32 or not 0 <= prefix < 64:
        raise ValueError("32-byte key and a one-byte prefix")
    payload = bytes([prefix]) + public_key
    return _b58encode(payload + _checksum(payload))


def signing_message(method: str, path: str, timestamp: int, nonce: str, body: bytes) -> bytes:
    return "\n".join(
        [DOMAIN, method.upper(), path, str(int(timestamp)), nonce, hashlib.sha256(body).hexdigest()]
    ).encode()


def verify_sr25519(public_key: bytes, message: bytes, signature: bytes) -> bool:
    import sr25519  # py-sr25519-bindings (the `api` extra)

    if len(signature) != 64:
        return False
    try:
        if sr25519.verify(signature, message, public_key):
            return True
        return bool(sr25519.verify(signature, b"<Bytes>" + message + b"</Bytes>", public_key))
    except (ValueError, TypeError):
        return False


class NonceStore:
    """One-time nonces per hotkey (SQLite, shared by API worker processes)."""

    def __init__(self, path: Path | str, ttl: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS nonces (hotkey TEXT, nonce TEXT, ts REAL, "
                "PRIMARY KEY (hotkey, nonce))"
            )

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def use(self, hotkey: str, nonce: str, now: float) -> bool:
        """True the first time (hotkey, nonce) is seen within the TTL."""
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM nonces WHERE ts < ?", (now - self.ttl,))
            try:
                c.execute("INSERT INTO nonces VALUES (?,?,?)", (hotkey, nonce, now))
            except sqlite3.IntegrityError:
                return False
            return True


@dataclass
class Verifier:
    nonces: NonceStore
    max_skew: float = 300.0
    now: object = time.time

    def verify(self, method: str, path: str, headers: Mapping[str, str], body: bytes) -> str:
        """The verified hotkey, or AuthError."""
        h = {k.lower(): v for k, v in headers.items()}
        hotkey, ts, nonce, sig = (h.get(k, "") for k in ("x-hotkey", "x-timestamp", "x-nonce", "x-signature"))
        if not (hotkey and ts and nonce and sig):
            raise AuthError("missing X-Hotkey / X-Timestamp / X-Nonce / X-Signature")
        pub = ss58_decode(hotkey)
        if not _NONCE.match(nonce):
            raise AuthError("bad nonce")
        try:
            t = int(ts)
            signature = bytes.fromhex(sig.removeprefix("0x"))
        except ValueError as e:
            raise AuthError("bad timestamp or signature encoding") from e
        now = float(self.now())  # type: ignore[operator]
        if abs(now - t) > self.max_skew:
            raise AuthError("timestamp outside the allowed window; check the clock")
        if not verify_sr25519(pub, signing_message(method, path, t, nonce, body), signature):
            raise AuthError("bad signature")
        if not self.nonces.use(hotkey, nonce, now):
            raise AuthError("nonce already used")
        return hotkey
