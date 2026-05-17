from fastapi import Depends, Header, HTTPException

from fiber import constants as cst
from fiber import utils
from fiber.chain import signatures
from fiber.logging_utils import get_logger
from fiber.miner.core import configuration
from fiber.miner.core.models.config import Config

logger = get_logger(__name__)


def get_config() -> Config:
    return configuration.factory_config()


async def verify_request(
    validator_hotkey: str = Header(..., alias=cst.VALIDATOR_HOTKEY),
    signature: str = Header(..., alias=cst.SIGNATURE),
    miner_hotkey: str = Header(..., alias=cst.MINER_HOTKEY),
    nonce: str = Header(..., alias=cst.NONCE),
    symmetric_key_uuid: str = Header(..., alias=cst.SYMMETRIC_KEY_UUID),
    config: Config = Depends(get_config),
):
    if not config.encryption_keys_handler.nonce_manager.nonce_is_valid(nonce):
        logger.debug("Nonce is not valid!")
        raise HTTPException(
            status_code=401,
            detail="Oi, that nonce is not valid!",
        )

    if not signatures.verify_signature(
        message=utils.construct_header_signing_message(nonce, miner_hotkey, symmetric_key_uuid),
        signer_ss58_address=validator_hotkey,
        signature=signature,
    ):
        raise HTTPException(
            status_code=401,
            detail="Oi, invalid signature, you're not who you said you were!",
        )


async def blacklist_low_stake(
    validator_hotkey: str = Header(..., alias=cst.VALIDATOR_HOTKEY), config: Config = Depends(get_config)
):
    metagraph = config.metagraph
    logger.debug("at least in here")
    node = metagraph.nodes.get(validator_hotkey)
    print(validator_hotkey)
    print(node)
    if not node:
        raise HTTPException(status_code=403, detail="Hotkey not found in metagraph")

    # if node.stake < config.min_stake_threshold:
    if node.stake < 10:
        logger.debug(f"Node {validator_hotkey} has insufficient stake of {node.stake} - minimum is {config.min_stake_threshold}")
        raise HTTPException(status_code=403, detail=f"Insufficient stake of {node.stake} ")
    
