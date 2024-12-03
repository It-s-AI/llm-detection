import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import HTTPException
from pydantic import BaseModel

from fiber.miner.core.models.config import Config
from fiber.miner.core.models.encryption import SymmetricKeyExchange, SymmetricKeyInfo
from fiber.miner.security.encryption import (
    decrypt_general_payload,
    decrypt_symmetric_key_exchange_payload,
)


class TestModel(BaseModel):
    field: str


class TestEncryption(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.config_mock = Mock()
        self.config_mock.encryption_keys_handler.private_key = self.private_key

    @patch("fiber.src.miner.security.encryption.get_config")
    async def test_decrypt_symmetric_key_exchange(self, mock_get_config):
        mock_get_config.return_value = self.config_mock

        test_data = SymmetricKeyExchange(
            encrypted_symmetric_key="encrypted_key",
            symmetric_key_uuid="test-uuid",
            ss58_address="test-hotkey",
            timestamp=datetime.now().timestamp(),
            nonce="test-nonce",
            signature="test-signature",
        )
        encrypted_payload = self.private_key.public_key().encrypt(
            test_data.model_dump_json().encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        result = await decrypt_symmetric_key_exchange_payload(self.config_mock, encrypted_payload)

        self.assertIsInstance(result, SymmetricKeyExchange)
        self.assertEqual(result.symmetric_key_uuid, test_data.symmetric_key_uuid)
        self.assertEqual(result.encrypted_symmetric_key, test_data.encrypted_symmetric_key)
        self.assertEqual(result.ss58_address, test_data.ss58_address)
        self.assertEqual(result.nonce, test_data.nonce)
        self.assertEqual(result.signature, test_data.signature)

    @patch("fiber.src.miner.security.encryption.get_config")
    @patch("fiber.src.miner.security.encryption.get_body")
    def test_decrypt_general_payload(self, mock_get_body, mock_get_config):
        fernet = Fernet(Fernet.generate_key())

        test_data = TestModel(field="test")
        encrypted_payload = fernet.encrypt(test_data.model_dump_json().encode())

        mock_get_body.return_value = encrypted_payload

        mock_config = MagicMock(spec=Config)

        mock_encryption_keys_handler = MagicMock()

        symmetric_key_info = SymmetricKeyInfo(fernet=fernet, expiration_time=datetime.now() + timedelta(hours=1))

        mock_encryption_keys_handler.get_symmetric_key.return_value = symmetric_key_info

        mock_config.encryption_keys_handler = mock_encryption_keys_handler

        mock_get_config.return_value = mock_config

        result = decrypt_general_payload(
            model=TestModel,
            encrypted_payload=encrypted_payload,
            key_uuid="test-uuid",
            hotkey="test-hotkey",
            config=mock_config,
        )

        self.assertIsInstance(result, TestModel)
        self.assertEqual(result.field, test_data.field)

        # Verify that get_symmetric_key was called with correct arguments
        mock_encryption_keys_handler.get_symmetric_key.assert_called_once_with("test-hotkey", "test-uuid")

    @patch("fiber.src.miner.security.encryption.get_config")
    @patch("fiber.src.miner.security.encryption.get_body")
    def test_decrypt_general_payload_no_key(self, mock_get_body, mock_get_config):
        mock_config = MagicMock(spec=Config)

        mock_encryption_keys_handler = MagicMock()
        mock_encryption_keys_handler.get_symmetric_key.return_value = None

        mock_config.encryption_keys_handler = mock_encryption_keys_handler

        mock_get_config.return_value = mock_config

        mock_get_body.return_value = b"test"

        with self.assertRaises(HTTPException) as context:
            decrypt_general_payload(
                model=TestModel,
                encrypted_payload=b"test",
                key_uuid="test-uuid",
                hotkey="test-hotkey",
                config=mock_config,
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.detail, "No symmetric key found for that hotkey and uuid")

        mock_encryption_keys_handler.get_symmetric_key.assert_called_once_with("test-hotkey", "test-uuid")


if __name__ == "__main__":
    asyncio.run(unittest.main())
