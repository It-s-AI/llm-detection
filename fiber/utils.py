import base64

from cryptography.fernet import Fernet

from fiber.logging_utils import get_logger

logger = get_logger(__name__)


def fernet_to_symmetric_key(fernet: Fernet) -> str:
    return base64.urlsafe_b64encode(fernet._signing_key + fernet._encryption_key).decode()


def construct_header_signing_message(nonce: str, miner_hotkey: str, symmetric_key_uuid: str) -> str:
    return f"{nonce}:{miner_hotkey}:{symmetric_key_uuid}"
