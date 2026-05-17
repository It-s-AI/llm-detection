import time

from fiber.logging_utils import get_logger
from fiber.miner.core import miner_constants as mcst

logger = get_logger(__name__)


class NonceManager:
    def __init__(self) -> None:
        self._nonces: dict[str, float] = {}
        self.TTL: int = 60 * 2

    def add_nonce(self, nonce: str) -> None:
        self._nonces[nonce] = time.time() + self.TTL

    def nonce_is_valid(self, nonce: str) -> bool:
        logger.debug(f"Checking if nonce is valid: {nonce}")
        # Check for collision
        if nonce in self._nonces:
            logger.debug(f"Invalid nonce because it's a collision: {nonce}")
            return False

        # If nonce isn't the right format, don't add it to self._nonces to prevent abuse
        # Check for recency
        current_time_ns = time.time_ns()
        logger.debug(f"Current time: {current_time_ns}")
        try:
            logger.debug(f"Nonce: {nonce}")
            timestamp_ns = int(nonce.split("_")[0])
            if timestamp_ns > 10**20:
                logger.debug(f"Invalid nonce because it's too old: {nonce}")
                raise ValueError()
        except (ValueError, IndexError):
            logger.debug(f"Invalid nonce because it's not in the right format. Nonce: {nonce}")
            return False

        # Nonces, can only be used once.
        self.add_nonce(nonce)

        if current_time_ns - timestamp_ns > mcst.NONCE_WINDOW_NS:
            logger.debug(f"Invalid nonce because it's too old: {nonce}")
            return False  # What an Old Nonce

        if timestamp_ns - current_time_ns > mcst.NONCE_WINDOW_NS:
            logger.debug(f"Invalid nonce because it's from the distant future: {nonce}")
            return False  # That nonce is from the distant future, and will be suspectible to replay attacks

        return True

    def cleanup_expired_nonces(self) -> None:
        current_time = time.time()
        expired_nonces: list[str] = [nonce for nonce, expiry_time in self._nonces.items() if current_time > expiry_time]
        for nonce in expired_nonces:
            del self._nonces[nonce]
