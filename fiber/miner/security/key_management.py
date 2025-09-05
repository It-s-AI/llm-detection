import json
import os
import threading
import time
from datetime import datetime

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from fiber import utils
from fiber.logging_utils import get_logger
from fiber.miner.core import miner_constants as mcst
from fiber.miner.core.models.encryption import SymmetricKeyInfo
from fiber.miner.security.nonce_management import NonceManager

logger = get_logger(__name__)
SECRET_KEYS_DIR = "secret_keys"

class EncryptionKeysHandler:
    def __init__(self, nonce_manager: NonceManager, storage_encryption_key: str, hotkey: str):
        self.hotkey = hotkey
        self.nonce_manager = nonce_manager
        self.asymmetric_fernet = Fernet(storage_encryption_key)
        self.symmetric_keys_fernets: dict[str, dict[str, SymmetricKeyInfo]] = {}
        self.load_asymmetric_keys()
        self.load_symmetric_keys()

        self._running: bool = True
        self._cleanup_thread: threading.Thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()

    def add_symmetric_key(self, uuid: str, hotkey_ss58_address: str, fernet: Fernet) -> None:
        symmetric_key_info = SymmetricKeyInfo.create(fernet)
        if hotkey_ss58_address not in self.symmetric_keys_fernets:
            self.symmetric_keys_fernets[hotkey_ss58_address] = {}
        self.symmetric_keys_fernets[hotkey_ss58_address][uuid] = symmetric_key_info

    def get_symmetric_key(self, hotkey_ss58_address: str, uuid: str) -> SymmetricKeyInfo:
        print(hotkey_ss58_address, uuid, "this is the result of address and uuid")
        if hotkey_ss58_address not in self.symmetric_keys_fernets or uuid not in self.symmetric_keys_fernets[hotkey_ss58_address]:
            return None
        return self.symmetric_keys_fernets[hotkey_ss58_address][uuid]

    # def save_symmetric_keys(self) -> None:
    #     filename = f"{self.hotkey}_{mcst.SYMMETRIC_KEYS_FILENAME}"
    #     serializable_keys = {
    #         hotkey: {
    #             uuid: {
    #                 "key": utils.fernet_to_symmetric_key(key_info.fernet),
    #                 "expiration_time": key_info.expiration_time.isoformat(),
    #             }
    #             for uuid, key_info in keys.items()
    #         }
    #         for hotkey, keys in self.symmetric_keys_fernets.items()
    #     }
    #     json_data = json.dumps(serializable_keys)
    #     encrypted_data = self.asymmetric_fernet.encrypt(json_data.encode())

    #     logger.info(f"Saving {len(serializable_keys)} symmetric keys to {filename}")
    #     with open(filename, "wb") as file:
    #         file.write(encrypted_data)

    # def load_symmetric_keys(self) -> None:
    #     filename = f"{self.hotkey}_{mcst.SYMMETRIC_KEYS_FILENAME}"
    #     if os.path.exists(filename):
    #         with open(filename, "rb") as f:
    #             encrypted_data = f.read()

    #         decrypted_data = self.asymmetric_fernet.decrypt(encrypted_data)
    #         loaded_keys: dict[str, dict[str, dict[str, str]]] = json.loads(decrypted_data.decode())

    #         self.symmetric_keys_fernets = {
    #             hotkey: {
    #                 uuid: SymmetricKeyInfo(
    #                     Fernet(key_data["key"]),
    #                     datetime.fromisoformat(key_data["expiration_time"]),
    #                 )
    #                 for uuid, key_data in keys.items()
    #             }
    #             for hotkey, keys in loaded_keys.items()
    #         }
    #         logger.info(f"Loaded {len(self.symmetric_keys_fernets)} symmetric keys")

    

    def save_symmetric_keys(self) -> None:
        # Ensure the directory exists
        os.makedirs(SECRET_KEYS_DIR, exist_ok=True)
        
        filename = os.path.join(SECRET_KEYS_DIR, f"{self.hotkey}_{mcst.SYMMETRIC_KEYS_FILENAME}")
        serializable_keys = {
            hotkey: {
                uuid: {
                    "key": utils.fernet_to_symmetric_key(key_info.fernet),
                    "expiration_time": key_info.expiration_time.isoformat(),
                }
                for uuid, key_info in keys.items()
            }
            for hotkey, keys in self.symmetric_keys_fernets.items()
        }
        json_data = json.dumps(serializable_keys)
        encrypted_data = self.asymmetric_fernet.encrypt(json_data.encode())

        logger.info(f"Saving {len(serializable_keys)} symmetric keys to {filename}")
        with open(filename, "wb") as file:
            file.write(encrypted_data)

    def load_symmetric_keys(self) -> None:
        filename = os.path.join(SECRET_KEYS_DIR, f"{self.hotkey}_{mcst.SYMMETRIC_KEYS_FILENAME}")
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = self.asymmetric_fernet.decrypt(encrypted_data)
            loaded_keys: dict[str, dict[str, dict[str, str]]] = json.loads(decrypted_data.decode())

            self.symmetric_keys_fernets = {
                hotkey: {
                    uuid: SymmetricKeyInfo(
                        Fernet(key_data["key"]),
                        datetime.fromisoformat(key_data["expiration_time"]),
                    )
                    for uuid, key_data in keys.items()
                }
                for hotkey, keys in loaded_keys.items()
            }
            logger.info(f"Loaded {len(self.symmetric_keys_fernets)} symmetric keys")



    def _clean_expired_keys(self) -> None:
        for hotkey in list(self.symmetric_keys_fernets.keys()):
            self.symmetric_keys_fernets[hotkey] = {
                uuid: key_info for uuid, key_info in self.symmetric_keys_fernets[hotkey].items() if not key_info.is_expired()
            }
            if not self.symmetric_keys_fernets[hotkey]:
                del self.symmetric_keys_fernets[hotkey]

    def _periodic_cleanup(self) -> None:
        while self._running:
            self._clean_expired_keys()
            self.nonce_manager.cleanup_expired_nonces()
            time.sleep(65)

    def load_asymmetric_keys(self) -> None:
        # NOTE: Allow this to be passed in via env too? Does it matter?
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()
        self.public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def close(self) -> None:
        self.save_symmetric_keys()
