"""Merkle tree over dataset items (root published on-chain at build time, D-57)."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def leaf_hash(item: Any) -> bytes:
    canon = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(b"\x00" + canon).digest()


def _node(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + a + b).digest()


def merkle_root(leaves: list[bytes]) -> bytes:
    if not leaves:
        return hashlib.sha256(b"").digest()
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_node(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: list[bytes], index: int) -> list[tuple[str, str]]:
    """Sibling path [(side, hex)], side 'L' if the sibling is on the left."""
    proof, level, i = [], list(leaves), index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sib = i ^ 1
        proof.append(("L" if sib < i else "R", level[sib].hex()))
        level = [_node(level[k], level[k + 1]) for k in range(0, len(level), 2)]
        i //= 2
    return proof


def verify_proof(leaf: bytes, proof: list[tuple[str, str]], root: bytes) -> bool:
    h = leaf
    for side, sib_hex in proof:
        sib = bytes.fromhex(sib_hex)
        h = _node(sib, h) if side == "L" else _node(h, sib)
    return h == root
