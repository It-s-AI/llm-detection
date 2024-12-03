import unittest
from datetime import datetime, timedelta
from unittest.mock import mock_open, patch

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import rsa

from fiber.miner.core import miner_constants as mcst
from fiber.miner.core.configuration import _derive_key_from_string
from fiber.miner.core.models.encryption import SymmetricKeyInfo
from fiber.miner.security.key_management import EncryptionKeysHandler
from fiber.miner.security.nonce_management import NonceManager


class TestKeyHandler(unittest.TestCase):
    def setUp(self):
        self.nonce_manager = NonceManager()
        self.hotkey = "test_hotkey"
        self.storage_encryption_key = _derive_key_from_string(mcst.DEFAULT_ENCRYPTION_STRING)
        self.encryption_keys_handler = EncryptionKeysHandler(self.nonce_manager, self.storage_encryption_key)

    def test_init(self):
        self.assertIsInstance(self.encryption_keys_handler.asymmetric_fernet, Fernet)
        self.assertIsInstance(self.encryption_keys_handler.symmetric_keys_fernets, dict)
        self.assertIsInstance(self.encryption_keys_handler.private_key, rsa.RSAPrivateKey)
        self.assertIsInstance(self.encryption_keys_handler.public_key, rsa.RSAPublicKey)
        self.assertIsInstance(self.encryption_keys_handler.public_bytes, bytes)

    def test_add_and_get_symmetric_key(self):
        uuid = "test_uuid"
        symmetric_key = Fernet.generate_key()
        fernet = Fernet(symmetric_key)
        self.encryption_keys_handler.add_symmetric_key(uuid, self.hotkey, fernet)
        retrieved_key = self.encryption_keys_handler.get_symmetric_key(self.hotkey, uuid)
        self.assertEqual(retrieved_key.fernet._encryption_key, fernet._encryption_key)
        self.assertEqual(retrieved_key.fernet._signing_key, fernet._signing_key)

    def test_get_nonexistent_symmetric_key(self):
        retrieved_key = self.encryption_keys_handler.get_symmetric_key("nonexistent_hotkey", "nonexistent_uuid")
        self.assertIsNone(retrieved_key)

    def test_clean_expired_keys(self):
        expired_key = SymmetricKeyInfo(Fernet(Fernet.generate_key()), datetime.now() - timedelta(seconds=1))
        valid_key = SymmetricKeyInfo(Fernet(Fernet.generate_key()), datetime.now() + timedelta(seconds=300))
        self.encryption_keys_handler.symmetric_keys_fernets = {
            "hotkey1": {"uuid1": expired_key, "uuid2": valid_key},
            "hotkey2": {"uuid3": expired_key},
        }
        self.encryption_keys_handler._clean_expired_keys()
        self.assertEqual(list(self.encryption_keys_handler.symmetric_keys_fernets.keys()), ["hotkey1"])
        self.assertEqual(
            list(self.encryption_keys_handler.symmetric_keys_fernets["hotkey1"].keys()),
            ["uuid2"],
        )

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=True)
    def test_save_and_load_symmetric_keys(self, mock_exists, mock_file):
        test_keys = {
            "hotkey1": {"uuid1": SymmetricKeyInfo(Fernet(Fernet.generate_key()), datetime.now() + timedelta(seconds=300))},
            "hotkey2": {"uuid2": SymmetricKeyInfo(Fernet(Fernet.generate_key()), datetime.now() + timedelta(seconds=300))},
        }
        self.encryption_keys_handler.symmetric_keys_fernets = test_keys

        self.encryption_keys_handler.save_symmetric_keys()

        mock_file().write.assert_called_once()
        encrypted_data = mock_file().write.call_args[0][0]

        mock_file().read.return_value = encrypted_data

        self.encryption_keys_handler.symmetric_keys_fernets = {}
        self.encryption_keys_handler.load_symmetric_keys()

        for hotkey, keys in self.encryption_keys_handler.symmetric_keys_fernets.items():
            for uuid, key_info in keys.items():
                self.assertIsInstance(key_info, SymmetricKeyInfo)
                self.assertEqual(
                    key_info.fernet._encryption_key,
                    test_keys[hotkey][uuid].fernet._encryption_key,
                )
                self.assertEqual(
                    key_info.fernet._signing_key,
                    test_keys[hotkey][uuid].fernet._signing_key,
                )

    @patch("os.path.exists", return_value=False)
    def test_load_symmetric_keys_file_not_exists(self, mock_exists):
        self.encryption_keys_handler.load_symmetric_keys()
        self.assertEqual(self.encryption_keys_handler.symmetric_keys_fernets, {})

    def test_load_asymmetric_keys(self):
        self.encryption_keys_handler.load_asymmetric_keys()
        self.assertIsInstance(self.encryption_keys_handler.private_key, rsa.RSAPrivateKey)
        self.assertIsInstance(self.encryption_keys_handler.public_key, rsa.RSAPublicKey)
        self.assertIsInstance(self.encryption_keys_handler.public_bytes, bytes)

    @patch.object(EncryptionKeysHandler, "save_symmetric_keys")
    def test_close(self, mock_save):
        self.encryption_keys_handler.close()
        mock_save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
