import unittest
from unittest.mock import patch

from substrateinterface import Keypair

from fiber.logging_utils import get_logger

logger = get_logger(__name__)


def sign_message(keypair: Keypair, message: str) -> str:
    return keypair.sign(message).hex()


def verify_signature(message: str, signature: str, ss58_address: str) -> bool:
    keypair = Keypair(ss58_address=ss58_address)
    try:
        return keypair.verify(
            message,
            bytes.fromhex(signature[2:] if signature.startswith("0x") else signature),
        )
    except ValueError:
        return False


class TestSignatureVerification(unittest.TestCase):
    def setUp(self):
        self.mnemonic = "clip organ olive upper oak void inject side suit toilet stick narrow"
        # Don't be dumb and use this for anything...
        self.keypair = Keypair.create_from_mnemonic(self.mnemonic)
        self.message = "Test message"
        self.ss58_address = self.keypair.ss58_address
        logger.debug(f"SS58 address: {self.ss58_address}")

    def test_sign_and_verify(self):
        signature = sign_message(self.keypair, self.message)
        self.assertTrue(verify_signature(self.message, signature, self.ss58_address))

    def test_invalid_signature(self):
        invalid_signature = "0x" + "1" * 128
        self.assertFalse(verify_signature(self.message, invalid_signature, self.ss58_address))

    def test_tampered_message(self):
        signature = sign_message(self.keypair, self.message)
        tampered_message = "Tampered message"
        self.assertFalse(verify_signature(tampered_message, signature, self.ss58_address))

    @patch("substrateinterface.Keypair")
    def test_invalid_address(self, mock_keypair):
        mock_keypair.side_effect = ValueError("Invalid SS58 address")
        invalid_address = "invalid_address"
        with self.assertRaises(ValueError):
            verify_signature(self.message, "0x" + "1" * 128, invalid_address)


if __name__ == "__main__":
    unittest.main()
