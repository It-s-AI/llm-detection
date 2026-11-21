"""On-chain notary for the period commitments (D-57): proof of *when* the commitment and the Merkle root
were published, independent of us.

Twice per period a small dedicated wallet (not the owner's key) posts ``System.remark_with_event`` to the
same chain the seed blocks come from:

    sn32-humanizer/1 {"commitment":"<H(secret)>","kind":"commitment","netuid":32,"period":"p2","seed_block":B}
    sn32-humanizer/1 {"block_hash":"<hash of B>","kind":"merkle_root","merkle_root":"...","n_items":484,...}

A block is immutable, so the first remark proves the commitment existed before block B (we could not
pick a secret after seeing B's hash), the second that the texts were fixed before the first evaluation.
Anyone checks both from the block bodies over plain JSON-RPC (``find_remark``, ``block_time``): the
remark bytes and the signer's public key are inside the signed extrinsic. Posting needs ``bittensor``,
which lives in the validator environment, so it runs as a subprocess (``scripts/notary_post.py``).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PREFIX = "sn32-humanizer/1 "
POST_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "notary_post.py"


def remark_text(kind: str, **fields: Any) -> str:
    return PREFIX + json.dumps({"kind": kind, **fields}, sort_keys=True, separators=(",", ":"))


def commitment_text(netuid: int, period: str, commitment: str, seed_block: int) -> str:
    return remark_text(
        "commitment", netuid=netuid, period=period, commitment=commitment, seed_block=int(seed_block)
    )


def merkle_text(netuid: int, period: str, merkle_root: str, n_items: int, block_hash: str) -> str:
    return remark_text("merkle_root", netuid=netuid, period=period, merkle_root=merkle_root, n_items=int(n_items),
                       block_hash=block_hash.removeprefix("0x"))  # fmt: skip


class SubprocessNotary:
    """Posts with the notary wallet through ``scripts/notary_post.py`` run by ``python`` (one that has
    bittensor); ``post`` returns the receipt {block, block_hash, extrinsic_index, extrinsic_hash, signer,
    signer_pubkey, fee_tao, network} after finalization, or raises."""

    def __init__(self, python: str, wallet: str, wallet_path: str, network: str, timeout_s: float = 300.0):
        self.python, self.wallet, self.wallet_path = python, wallet, wallet_path
        self.network, self.timeout_s = network, timeout_s

    def post(self, text: str) -> dict[str, Any]:
        cmd = [self.python, str(POST_SCRIPT), "--wallet", self.wallet, "--path", self.wallet_path,
               "--network", self.network]  # fmt: skip
        # our own script with the configured interpreter; the remark goes on stdin
        r = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=self.timeout_s)  # noqa: S603
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if r.returncode != 0 or not lines:
            raise RuntimeError(f"notary post failed: {(r.stderr or r.stdout).strip()[-300:]}")
        return json.loads(lines[-1])


# --- the check (plain JSON-RPC, no bittensor) -----------------------------------------------------
def find_remark(chain: Any, block: int, text: str, signer_pubkey: str) -> int | None:
    """Index of the extrinsic in ``block`` that carries ``text`` signed by ``signer_pubkey``, or None."""
    want, signer = text.encode().hex(), "00" + signer_pubkey.removeprefix("0x").lower()
    for i, ex in enumerate(chain.block_extrinsics(block)):
        h = ex.removeprefix("0x").lower()
        if want in h and signer in h:
            return i
    return None


def _compact(b: bytes, i: int) -> tuple[int, int]:
    mode = b[i] & 3
    if mode == 0:
        return b[i] >> 2, i + 1
    if mode == 1:
        return int.from_bytes(b[i : i + 2], "little") >> 2, i + 2
    if mode == 2:
        return int.from_bytes(b[i : i + 4], "little") >> 2, i + 4
    n = (b[i] >> 2) + 4
    return int.from_bytes(b[i + 1 : i + 1 + n], "little"), i + 1 + n


def block_time(chain: Any, block: int) -> float:
    """Unix time of ``block`` from its timestamp inherent (the first extrinsic: length, version byte,
    call index, compact milliseconds)."""
    b = bytes.fromhex(chain.block_extrinsics(block)[0].removeprefix("0x"))
    _, i = _compact(b, 0)
    ms, _ = _compact(b, i + 3)
    if not 1.5e12 < ms < 4.1e12:
        raise ValueError(f"block {block}: no timestamp inherent where expected")
    return ms / 1000.0


_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def ss58_pubkey(address: str) -> str:
    """Public key (hex) of an SS58 address with a one-byte prefix; the checksum is verified."""
    n = 0
    for c in address:
        n = n * 58 + _B58.index(c)
    raw = n.to_bytes(35, "big")
    data, check = raw[:33], raw[33:]
    if hashlib.blake2b(b"SS58PRE" + data, digest_size=64).digest()[:2] != check:
        raise ValueError(f"{address}: bad SS58 checksum")
    return data[1:].hex()


def verify_notary(rec: dict[str, Any], manifest: dict[str, Any], chain: Any, *, starts: list[float] = (),
                  notary_address: str | None = None) -> tuple[list[str], list[str]]:  # fmt: skip
    """Check the two remarks of one period against its manifest; returns (passed, failed)."""
    ok: list[str] = []
    bad: list[str] = []

    def check(cond: bool, what: str) -> None:
        (ok if cond else bad).append(what)

    pid = manifest["period_id"]
    nc, nm = rec.get("notary_commitment"), rec.get("notary_merkle")
    if not nc or not nm:
        bad.append(f"{pid}: the commitment and the Merkle root were not notarized on chain")
        return ok, bad
    if manifest.get("block") is None:
        bad.append(f"{pid}: the manifest names no seed block")
        return ok, bad
    B = int(manifest["block"])
    netuid = json.loads(nc["text"].removeprefix(PREFIX))["netuid"]
    signer = nc["signer_pubkey"]
    if notary_address is not None:
        check(ss58_pubkey(notary_address) == signer.removeprefix("0x"), f"remarks signed by {notary_address}")
    check(nm["signer_pubkey"] == signer, "both remarks have the same signer")
    want = commitment_text(netuid, pid, manifest["commitment"], B)
    check(nc["text"] == want and find_remark(chain, nc["block"], want, signer) is not None,
          f"commitment on chain in block {nc['block']}")  # fmt: skip
    check(int(nc["block"]) < B, f"commitment block {nc['block']} < seed block {B}")
    want = merkle_text(netuid, pid, manifest["merkle_root"], manifest["n_items"], manifest["block_hash"])
    check(nm["text"] == want and find_remark(chain, nm["block"], want, signer) is not None,
          f"Merkle root on chain in block {nm['block']}")  # fmt: skip
    check(int(nm["block"]) > B, f"Merkle root block {nm['block']} after the seed block {B}")
    if starts:
        t = block_time(chain, int(nm["block"]))
        check(t < min(starts), f"Merkle root on chain before the first evaluation of {pid}")
    return ok, bad


def default_python() -> str:
    return sys.executable
