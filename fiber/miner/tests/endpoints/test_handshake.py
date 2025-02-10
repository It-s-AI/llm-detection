import base64
import time
import unittest
from unittest.mock import Mock, patch

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fiber.miner.core.configuration import Config
from fiber.miner.core.models.encryption import SymmetricKeyExchange
from fiber.miner.endpoints.handshake import factory_router
from fiber.miner.security.nonce_management import NonceManager


class TestHandshake(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        router = factory_router()
        app.include_router(router)
        self.client = TestClient(app)

        self.mock_config = Mock(spec=Config)
        self.mock_encryption_keys_handler = Mock()
        self.mock_config.encryption_keys_handler = self.mock_encryption_keys_handler

        self.mock_encryption_keys_handler.public_bytes = b"mock_public_key"
        self.mock_encryption_keys_handler.nonce_manager = NonceManager()
        self.mock_encryption_keys_handler.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        self.mock_config.keypair = Mock()
        self.mock_config.keypair.hotkey = "test_hotkey"

    @patch("fiber.src.miner.core.config.factory_config")
    @patch("fiber.src.miner.security.signatures.sign_message")
    def test_get_public_key(self, mock_sign_message, mock_factory_config):
        # Configure the mock_factory_config
        mock_factory_config.return_value = self.mock_config

        # Configure mock_sign_message
        mock_sign_message.return_value = "mock_signature"

        # Make the request
        response = self.client.get("/public_key")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["public_key"], self.mock_encryption_keys_handler.public_bytes.decode())
        self.assertEqual(data["hotkey"], self.mock_config.keypair.hotkey)
        self.assertEqual(data["signature"], "mock_signature")
        self.assertIn("timestamp", data)

        mock_factory_config.assert_called_once()

    @patch("fiber.src.miner.security.signatures.verify_signature")
    @patch("fiber.src.miner.core.config.factory_config")
    def test_exchange_symmetric_key_success(self, mock_factory_config, mock_verify_signature):
        mock_factory_config.return_value = self.mock_config
        mock_verify_signature.return_value = True
        symmetric_key = b"test_symmetric_key"
        encrypted_symmetric_key = self.mock_encryption_keys_handler.private_key.public_key().encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        payload = SymmetricKeyExchange(
            encrypted_symmetric_key=base64.b64encode(encrypted_symmetric_key).decode(),
            symmetric_key_uuid="test_uuid",
            ss58_address="test_hotkey",
            timestamp=time.time(),
            nonce="test_nonce",
            signature="test_signature",
        )

        response = self.client.post("/exchange_symmetric_key", json=payload.model_dump())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "Symmetric key exchanged successfully"})
        self.mock_encryption_keys_handler.add_symmetric_key.assert_called_once_with(
            payload.symmetric_key_uuid,
            payload.ss58_address,
            base64.b64encode(symmetric_key).decode(),
        )

    @patch("fiber.src.miner.security.signatures.verify_signature")
    @patch("fiber.src.miner.core.config.factory_config")
    def test_exchange_symmetric_key_invalid_signature(self, mock_factory_config, mock_verify_signature):
        mock_factory_config.return_value = self.mock_config
        mock_verify_signature.return_value = False

        payload = SymmetricKeyExchange(
            encrypted_symmetric_key=base64.b64encode(b"test_key").decode(),
            symmetric_key_uuid="test_uuid",
            ss58_address="test_hotkey",
            timestamp=time.time(),
            nonce="test_nonce",
            signature="invalid_signature",
        )

        response = self.client.post("/exchange_symmetric_key", json=payload.model_dump())

        self.assertEqual(response.status_code, 400)
        self.assertIn("invalid signature", response.json()["detail"])

    @patch("fiber.src.miner.security.signatures.verify_signature")
    @patch("fiber.src.miner.core.config.factory_config")
    def test_exchange_symmetric_key_duplicate_nonce(self, mock_factory_config, mock_verify_signature):
        mock_factory_config.return_value = self.mock_config
        mock_verify_signature.return_value = True

        payload = SymmetricKeyExchange(
            encrypted_symmetric_key=base64.b64encode(b"test_key").decode(),
            symmetric_key_uuid="test_uuid",
            ss58_address="test_hotkey",
            timestamp=time.time(),
            nonce="duplicate_nonce",
            signature="test_signature",
        )

        self.mock_encryption_keys_handler.nonce_manager.add_nonce("duplicate_nonce")

        response = self.client.post("/exchange_symmetric_key", json=payload.model_dump())

        self.assertEqual(response.status_code, 400)
        self.assertIn("nonce", response.json()["detail"])

    def test_factory_router(self):
        router = factory_router()
        self.assertEqual(len(router.routes), 2)
        self.assertTrue(any(route.path == "/exchange_symmetric_key" for route in router.routes))
        self.assertTrue(any(route.path == "/public_key" for route in router.routes))


if __name__ == "__main__":
    unittest.main()
