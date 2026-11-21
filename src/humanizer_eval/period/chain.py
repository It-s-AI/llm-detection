"""Read-only access to the finney chain over plain JSON-RPC (no bittensor needed): the finalized head and
block hashes for the period seed (D-57). Any public node works; the default is the OpenTensor lite node."""

from __future__ import annotations

import hashlib
from typing import Any

import httpx

FINNEY_RPC = "https://lite.chain.opentensor.ai"
#: block bodies of old blocks (notary remarks) are kept by archive nodes
FINNEY_ARCHIVE_RPC = "https://archive.chain.opentensor.ai"
TESTNET_RPC = "https://test.finney.opentensor.ai"
#: twox128("System") ++ twox128("Account"): the storage prefix of account balances
_SYSTEM_ACCOUNT = "26aa394eea5630e07c48ae0c9558cef7b99d880ec681799c0cf30e8886371da9"


class Chain:
    def __init__(self, url: str = FINNEY_RPC, http: Any = None, timeout: float = 20.0):
        self.url = url
        self.http = http or httpx.Client(timeout=timeout)

    def _call(self, method: str, *params: Any) -> Any:
        r = self.http.post(
            self.url, json={"id": 1, "jsonrpc": "2.0", "method": method, "params": list(params)}
        )
        r.raise_for_status()
        d = r.json()
        if d.get("error"):
            raise RuntimeError(f"{method}: {d['error']}")
        return d["result"]

    def head(self) -> int:
        """Number of the latest finalized block."""
        h = self._call("chain_getHeader", self._call("chain_getFinalizedHead"))
        return int(h["number"], 16)

    def block_extrinsics(self, number: int) -> list[str]:
        """The SCALE-encoded extrinsics (hex) of block ``number``."""
        blk = self._call("chain_getBlock", "0x" + self.block_hash(number))
        if not blk:
            raise RuntimeError(f"block {number}: body not available on {self.url} (use an archive node)")
        return list(blk["block"]["extrinsics"])

    def free_balance(self, pubkey_hex: str) -> float:
        """Free balance in TAO of the account with this public key (System.Account, blake2_128_concat)."""
        pk = bytes.fromhex(pubkey_hex.removeprefix("0x"))
        key = "0x" + _SYSTEM_ACCOUNT + hashlib.blake2b(pk, digest_size=16).hexdigest() + pk.hex()
        raw = self._call("state_getStorage", key)
        if not raw:
            return 0.0
        b = bytes.fromhex(raw.removeprefix("0x"))
        return int.from_bytes(b[16:32], "little") / 1e9  # nonce, consumers, providers, sufficients: 4 x u32

    def block_hash(self, number: int) -> str:
        """Hash of block ``number`` as 64 hex characters (no 0x)."""
        h = self._call("chain_getBlockHash", int(number))
        if not h:
            raise RuntimeError(f"block {number} is not known to {self.url}")
        return str(h).removeprefix("0x")
