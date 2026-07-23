# The MIT License (MIT)
# Copyright © 2024 It's AI
#
# End-to-end encryption of the miner -> validator predictions.
#
# The valuable secret in this subnet is the miner's OUTPUT. bittensor axons
# speak plaintext HTTP, so anyone able to read the validator<->miner traffic can
# copy a miner's predictions off the wire. Here the validator sends a fresh
# ephemeral X25519 public key with each query; the miner seals its predictions
# to that key (libsodium sealed box); only the validator, holding the matching
# private key, can open them. A passive reader of the wire sees only ciphertext.

import json

from nacl.public import PrivateKey, PublicKey, SealedBox
from nacl.encoding import HexEncoder


def generate_keypair():
    """Return (private_key, public_key_hex) for one query round."""
    sk = PrivateKey.generate()
    pub_hex = sk.public_key.encode(HexEncoder).decode()
    return sk, pub_hex


def encrypt_predictions(predictions, pubkey_hex: str) -> str:
    """Seal predictions to the validator's public key; return hex ciphertext."""
    box = SealedBox(PublicKey(pubkey_hex.encode(), HexEncoder))
    # `default=float` coerces numpy scalars (e.g. float32 from the deberta model)
    # which json.dumps cannot serialize on its own; float64 already serializes.
    plaintext = json.dumps(predictions, separators=(",", ":"), default=float).encode("utf-8")
    return box.encrypt(plaintext).hex()


def decrypt_predictions(ciphertext_hex: str, private_key: "PrivateKey"):
    """Open a sealed ciphertext with the validator's private key -> predictions."""
    box = SealedBox(private_key)
    plaintext = box.decrypt(bytes.fromhex(ciphertext_hex))
    return json.loads(plaintext.decode("utf-8"))
