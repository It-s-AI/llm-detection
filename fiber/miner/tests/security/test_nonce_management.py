import unittest
from unittest.mock import patch

from fiber.miner.security.nonce_management import NonceManager
from fiber.validator import generate_nonce


class TestNonceManager(unittest.TestCase):
    def setUp(self):
        self.nonce_manager = NonceManager()

    def test_add_nonce(self):
        nonce = "test_nonce"
        self.nonce_manager.add_nonce(nonce)
        self.assertIn(nonce, self.nonce_manager._nonces)

    def test_nonce_in_nonces_new_nonce(self):
        nonce = generate_nonce.generate_nonce()
        result = self.nonce_manager.nonce_is_valid(nonce)
        self.assertTrue(result)
        self.assertIn(nonce, self.nonce_manager._nonces)

    def test_nonce_in_nonces_existing_nonce(self):
        nonce = generate_nonce.generate_nonce()
        self.nonce_manager.add_nonce(nonce)
        result = self.nonce_manager.nonce_is_valid(nonce)
        self.assertFalse(result)

    @patch("time.time_ns")
    def test_old_nonce(self, mock_time):
        mock_time.return_value = 1_000_000_000
        nonce = generate_nonce.generate_nonce()
        mock_time.return_value = 1_000_000_000_000

        result = self.nonce_manager.nonce_is_valid(nonce)
        self.assertFalse(result)

    @patch("time.time_ns")
    def test_too_new_nonce(self, mock_time):
        mock_time.return_value = 1_000_000_000_000
        nonce = generate_nonce.generate_nonce()
        mock_time.return_value = 1_000_000_000

        result = self.nonce_manager.nonce_is_valid(nonce)
        self.assertFalse(result)

    @patch("time.time")
    def test_cleanup_expired_nonces(self, mock_time):
        mock_time.return_value = 100
        self.nonce_manager.add_nonce("expired_nonce")
        mock_time.return_value = 100_000
        self.nonce_manager.add_nonce("valid_nonce")

        mock_time.return_value = 1000
        self.nonce_manager.cleanup_expired_nonces()

        self.assertNotIn("expired_nonce", self.nonce_manager._nonces)
        self.assertIn("valid_nonce", self.nonce_manager._nonces)

    def test_contains(self):
        nonce = "test_nonce"
        self.nonce_manager.add_nonce(nonce)
        self.assertIn(nonce, self.nonce_manager._nonces)

    def test_len(self):
        self.nonce_manager.add_nonce("nonce1")
        self.nonce_manager.add_nonce("nonce2")
        self.assertEqual(len(self.nonce_manager._nonces), 2)

    def test_iter(self):
        nonces = ["nonce1", "nonce2", "nonce3"]
        for nonce in nonces:
            self.nonce_manager.add_nonce(nonce)

        self.assertEqual(set(self.nonce_manager._nonces), set(nonces))


if __name__ == "__main__":
    unittest.main()
