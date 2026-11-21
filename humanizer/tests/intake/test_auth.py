import pytest

from humanizer_eval.intake.auth import (
    AuthError,
    NonceStore,
    Verifier,
    signing_message,
    ss58_decode,
    ss58_encode,
)
from humanizer_eval.intake.client import SeedSigner, signed_headers

# produced with bittensor_wallet 4.x: Keypair.create_from_seed("0x" + "11" * 32).sign(MSG)
BT_ADDR = "5DtDLm5rQHShDqojQpsvcN8tRXHVFaecfDoRet1SU6BFD9Fi"
BT_PUB = "50780547322a1ceba67ea8c552c9bc6c686f8698ac9a8cafab7cd15a1db19859"
BT_SIG = (
    "863ee7a2d794769c72abe2b81b7d1867d6e3c28ffe83e3032043dbb5b4c6d009"
    "7acab11364b3a6856cc9766a09feb4dab7fa2c5d72aa1fe5ae3be7930f35838f"
)


def test_bittensor_wallet_signature_and_address_interop():
    from humanizer_eval.intake.auth import verify_sr25519

    assert ss58_decode(BT_ADDR).hex() == BT_PUB
    assert ss58_encode(bytes.fromhex(BT_PUB)) == BT_ADDR
    # the vector was signed over the literal message with 64 zeros as the body hash
    msg = ("humeval-v1\nPOST\n/v1/submissions\n1700000000\nabc\n" + "0" * 64).encode()
    assert verify_sr25519(bytes.fromhex(BT_PUB), msg, bytes.fromhex(BT_SIG))
    assert not verify_sr25519(bytes.fromhex(BT_PUB), msg + b"x", bytes.fromhex(BT_SIG))


def test_ss58_rejects_bad_checksum_and_other_networks():
    bad = BT_ADDR[:-1] + ("j" if BT_ADDR[-1] != "j" else "k")
    with pytest.raises(AuthError):
        ss58_decode(bad)
    polkadot = ss58_encode(bytes.fromhex(BT_PUB), prefix=0)
    with pytest.raises(AuthError):
        ss58_decode(polkadot)
    with pytest.raises(AuthError):
        ss58_decode("not-an-address-0OIl" * 3)


def _verifier(tmp_path, now=1_700_000_000.0):
    return Verifier(NonceStore(tmp_path / "n.sqlite", ttl=600), max_skew=300, now=lambda: now)


def test_signed_request_accepted_once(tmp_path):
    s = SeedSigner(b"\x07" * 32)
    v = _verifier(tmp_path)
    body = b'{"a":1}'
    h = signed_headers(s, "POST", "/v1/submissions", body, ts=1_700_000_010)
    assert v.verify("POST", "/v1/submissions", h, body) == s.ss58_address
    with pytest.raises(AuthError, match="nonce already used"):
        v.verify("POST", "/v1/submissions", h, body)


@pytest.mark.parametrize(
    "change",
    ["body", "path", "method", "stale", "other_key", "bad_hex"],
)
def test_tampered_requests_rejected(tmp_path, change):
    s, other = SeedSigner(b"\x07" * 32), SeedSigner(b"\x08" * 32)
    v = _verifier(tmp_path)
    body, path, method, ts = b'{"a":1}', "/v1/submissions", "POST", 1_700_000_010
    if change == "stale":
        ts -= 1000
    h = signed_headers(other if change == "other_key" else s, method, path, body, ts=ts)
    if change == "other_key":
        h["X-Hotkey"] = s.ss58_address
    if change == "bad_hex":
        h["X-Signature"] = "zz" + h["X-Signature"][2:]
    if change == "body":
        body = b'{"a":2}'
    if change == "path":
        path = "/v1/refunds"
    if change == "method":
        method = "DELETE"
    with pytest.raises(AuthError):
        v.verify(method, path, h, body)


def test_polkadot_js_wrapped_signature_accepted(tmp_path):
    s = SeedSigner(b"\x09" * 32)
    v = _verifier(tmp_path)
    msg = signing_message("GET", "/v1/balance", 1_700_000_000, "nonce-123", b"")
    h = {
        "X-Hotkey": s.ss58_address,
        "X-Timestamp": "1700000000",
        "X-Nonce": "nonce-123",
        "X-Signature": s.sign(b"<Bytes>" + msg + b"</Bytes>").hex(),
    }
    assert v.verify("GET", "/v1/balance", h, b"") == s.ss58_address
