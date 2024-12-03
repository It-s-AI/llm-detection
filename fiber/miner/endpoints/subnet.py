"""
THIS IS AN EXAMPLE FILE OF A SUBNET ENDPOINT!

PLEASE IMPLEMENT YOUR OWN :)
"""

from functools import partial

from fastapi import Depends, Request, Header  # Added Request import
from fastapi.routing import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from fiber.miner.dependencies import blacklist_low_stake, verify_request
from fiber.miner.security.encryption import decrypt_general_payload
from fiber.logging_utils import get_logger
from fiber.miner.dependencies import get_config
from fiber.miner.core.models.config import Config
from fiber import constants as cst
from fiber.miner.security.encryption import T
logger = get_logger(__name__)

class ContentModel(BaseModel):
    data: Dict[str, Any]

# async def example_subnet_request(
#     request: Request,  # To capture the raw encrypted payload
#     config: Config = Depends(get_config),
# ):
#     # Log the encrypted payload received
#     encrypted_payload = await request.body()
#     print("Encrypted Payload (raw):", encrypted_payload)

#     # Decrypt the payload directly
#     decrypted_payload = decrypt_general_payload(ContentModel, encrypted_payload, config = config)

#     # Log the decrypted payload
#     if decrypted_payload:
#         print("Decrypted Payload (parsed):", decrypted_payload.dict())
#     else:
#         print("Failed to decrypt payload. Check symmetric key or decryption logic.")

#     print("The synapse received")

#     return {"status": "Example request received, haha"}

async def example_subnet_request(
    request: Request,
    config: Config = Depends(get_config),
    validator_hotkey: str = Header(..., alias=cst.VALIDATOR_HOTKEY),
    miner_hotkey: str = Header(..., alias=cst.MINER_HOTKEY),
    symmetric_key_uuid=Header(..., alias=cst.SYMMETRIC_KEY_UUID),
):
    # Log the encrypted payload received
    encrypted_payload = await request.body()
    print("Encrypted Payload (raw):", encrypted_payload)

    # Decrypt the payload directly
    decrypted_payload = decrypt_general_payload(
        model= ContentModel,
        encrypted_payload=encrypted_payload,
        symmetric_key_uuid=symmetric_key_uuid,
        validator_hotkey=validator_hotkey,
        miner_hotkey=miner_hotkey,
        config=config
    )

    # Log the decrypted payload
    if decrypted_payload:
        print("Decrypted Payload (parsed):", decrypted_payload.dict())
    else:
        print("Failed to decrypt payload. Check symmetric key or decryption logic.")

    print("The synapse received")

    return decrypted_payload
    # return {"status": "Example request received, haha"}


def factory_router() -> APIRouter:
    router = APIRouter()
    router.add_api_route(
        "/example-subnet-request",
        example_subnet_request,
        tags=["Example"],
        dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
        methods=["POST"],
    )
    return router
