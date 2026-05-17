import time

from cryptography.fernet import Fernet
from fastapi import APIRouter, Depends, Header

from fiber import constants as cst
from fiber.logging_utils import get_logger
from fiber.miner.core.configuration import Config
from fiber.miner.core.models.encryption import PublicKeyResponse, SymmetricKeyExchange
from fiber.miner.dependencies import blacklist_low_stake, get_config, verify_request
from fiber.miner.security.encryption import get_symmetric_key_b64_from_payload

logger = get_logger(__name__)


async def get_public_key(config: Config = Depends(get_config)):
    public_key = config.encryption_keys_handler.public_bytes.decode()
    return PublicKeyResponse(
        public_key=public_key,
        timestamp=time.time(),
    )


async def exchange_symmetric_key(
    payload: SymmetricKeyExchange,
    validator_hotkey_address: str = Header(..., alias=cst.VALIDATOR_HOTKEY),
    nonce: str = Header(..., alias=cst.NONCE),
    symmetric_key_uuid: str = Header(..., alias=cst.SYMMETRIC_KEY_UUID),
    config: Config = Depends(get_config),
):
    logger.debug("at least in exchange")
    base64_symmetric_key = get_symmetric_key_b64_from_payload(payload, config.encryption_keys_handler.private_key)
    fernet = Fernet(base64_symmetric_key)
    config.encryption_keys_handler.add_symmetric_key(
        uuid=symmetric_key_uuid,
        hotkey_ss58_address=validator_hotkey_address,
        fernet=fernet,
    )

    return {"status": "Symmetric key exchanged successfully"}


def factory_router() -> APIRouter:
    router = APIRouter(tags=["Handshake"])
    router.add_api_route("/public-encryption-key", get_public_key, methods=["GET"])
    router.add_api_route(
        "/exchange-symmetric-key",
        exchange_symmetric_key,
        methods=["POST"],
        dependencies=[
            Depends(blacklist_low_stake),
            Depends(verify_request),
        ],
    )
    return router
