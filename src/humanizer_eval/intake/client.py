"""Miner-side client: signs each request with the hotkey and calls the evaluation API.

    from bittensor_wallet import Wallet
    from humanizer_eval.intake.client import MinerClient

    c = MinerClient("https://eval.example", Wallet(name="miner", hotkey="default").hotkey)
    sub = c.submit("https://huggingface.co/.../resolve/main/bundle.tar", sha256, size)
    # pay sub["payment"]["url"] if the balance is short, then:
    print(c.wait(sub["submission_id"]))

Any object with ``ss58_address`` and ``sign(bytes) -> bytes`` works as the signer (a bittensor_wallet
Keypair does); ``SeedSigner`` is a test signer from a 32-byte seed.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any

import httpx

from humanizer_eval.intake.auth import signing_message, ss58_encode


class SeedSigner:
    def __init__(self, seed: bytes):
        import sr25519

        self._kp = sr25519.pair_from_seed(seed)
        self.public_key = self._kp[0]
        self.ss58_address = ss58_encode(self.public_key)

    def sign(self, message: bytes) -> bytes:
        import sr25519

        return sr25519.sign(self._kp, message)


def file_digest(path: Path | str) -> tuple[str, int]:
    """sha256 and size of a bundle file, for the submission fields."""
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def signed_headers(signer: Any, method: str, path: str, body: bytes, ts: int | None = None) -> dict[str, str]:
    ts = int(time.time()) if ts is None else ts
    nonce = secrets.token_urlsafe(16)
    sig = signer.sign(signing_message(method, path, ts, nonce, body))
    if isinstance(sig, str):
        sig = bytes.fromhex(sig.removeprefix("0x"))
    return {
        "X-Hotkey": signer.ss58_address,
        "X-Timestamp": str(ts),
        "X-Nonce": nonce,
        "X-Signature": sig.hex(),
    }


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status


class MinerClient:
    def __init__(self, base_url: str, signer: Any, http: httpx.Client | None = None, timeout: float = 60):
        self.base = base_url.rstrip("/")
        self.signer = signer
        self.http = http or httpx.Client(timeout=timeout)

    def _req(self, method: str, path: str, data: dict | None = None, sign: bool = True) -> dict[str, Any]:
        body = b"" if data is None else json.dumps(data, separators=(",", ":")).encode()
        headers = {"Content-Type": "application/json"} if body else {}
        if sign:
            headers.update(signed_headers(self.signer, method, path, body))
        r = self.http.request(method, self.base + path, content=body or None, headers=headers)
        try:
            out = r.json()
        except ValueError:
            out = {"error": r.text[:500]}
        if r.status_code >= 400:
            raise ApiError(r.status_code, out.get("error", r.text[:500]))
        return out

    def info(self) -> dict[str, Any]:
        return self._req("GET", "/v1/info", sign=False)

    def submit(
        self, bundle_url: str, bundle_sha256: str, bundle_bytes: int, name: str = ""
    ) -> dict[str, Any]:
        return self._req(
            "POST",
            "/v1/submissions",
            {
                "bundle_url": bundle_url,
                "bundle_sha256": bundle_sha256,
                "bundle_bytes": bundle_bytes,
                "name": name,
            },
        )

    def submit_hf(self, commitment: str | None = None, hf_token: str | None = None, name: str = "") -> dict:
        """Submit the bundle pinned by this hotkey's on-chain commitment ``<repo>@<commit>``."""
        d: dict[str, Any] = {"name": name}
        if commitment:
            d["commitment"] = commitment
        if hf_token:
            d["hf_token"] = hf_token
        return self._req("POST", "/v1/submissions", d)

    def status(self, sid: str) -> dict[str, Any]:
        return self._req("GET", f"/v1/submissions/{sid}")

    def statement(self, sid: str) -> dict[str, Any]:
        return self._req("GET", f"/v1/submissions/{sid}/statement")

    def paid(self, sid: str) -> dict[str, Any]:
        """Tell the evaluator the Checkout link is paid; it asks Stripe and queues the submission."""
        return self._req("POST", f"/v1/submissions/{sid}/paid")

    def cancel(self, sid: str) -> dict[str, Any]:
        return self._req("DELETE", f"/v1/submissions/{sid}")

    def balance(self) -> dict[str, Any]:
        return self._req("GET", "/v1/balance")

    def refund(self, key: str, amount_cents: int | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {"key": key}
        if amount_cents is not None:
            d["amount_cents"] = amount_cents
        return self._req("POST", "/v1/refunds", d)

    def wait(self, sid: str, poll_s: float = 30.0, timeout_s: float = 6 * 3600) -> dict[str, Any]:
        t0 = time.time()
        while True:
            st = self.status(sid)
            if st["state"] in ("done", "failed", "expired", "cancelled") or time.time() - t0 > timeout_s:
                return st
            time.sleep(poll_s)
