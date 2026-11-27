"""Encryption of hidden period data.

* At rest: NaCl SecretBox with a symmetric key held only by the owner services.
* To an evaluation VM/pod: SealedBox to the pod's ephemeral public key, so only
  that pod can read the inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

from nacl.public import PrivateKey, PublicKey, SealedBox
from nacl.secret import SecretBox
from nacl.utils import random


def write_private(path: Path | str, data: bytes | str) -> None:
    """Write a secret readable by its owner only from the first byte (no window at the default umask)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.parent.name == "keys":
        p.parent.chmod(0o700)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data.encode() if isinstance(data, str) else data)


def load_or_create_key(path: Path | str) -> bytes:
    p = Path(path)
    if p.exists():
        return p.read_bytes()
    p.parent.mkdir(parents=True, exist_ok=True)
    key = random(SecretBox.KEY_SIZE)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # it decrypts every period and token
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


def encrypt_at_rest(key: bytes, data: bytes) -> bytes:
    return SecretBox(key).encrypt(data)


def decrypt_at_rest(key: bytes, blob: bytes) -> bytes:
    return SecretBox(key).decrypt(blob)


def seal_to(public_key_hex: str, data: bytes) -> bytes:
    return SealedBox(PublicKey(bytes.fromhex(public_key_hex))).encrypt(data)


def open_sealed(private_key: PrivateKey, blob: bytes) -> bytes:
    return SealedBox(private_key).decrypt(blob)
