
from substrateinterface import Keypair

from fiber.logging_utils import get_logger

logger = get_logger(__name__)


def sign_message(keypair: Keypair, message: str | None) -> str | None:
    if message is None:
        return None
    return f"0x{keypair.sign(message).hex()}"


def verify_signature(message: str | None, signature: str, signer_ss58_address: str) -> bool:
    if message is None:
        return False
    try:
        keypair = Keypair(ss58_address=signer_ss58_address)
        return keypair.verify(data=message, signature=signature)
    except ValueError:
        return False
