import base64
import json
from typing import Type, TypeVar

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import Depends, Header, HTTPException, Request
from pydantic import BaseModel, parse_obj_as

from fiber.logging_utils import get_logger
from fiber.miner.core.models.config import Config
from fiber.miner.core.models.encryption import SymmetricKeyExchange
from fiber.miner.dependencies import get_config
from fiber import constants as cst
# from fiber.miner.endpoints.subnet import ContentModel
logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


async def get_body(request: Request) -> bytes:
    return await request.body()


def get_symmetric_key_b64_from_payload(payload: SymmetricKeyExchange, private_key: rsa.RSAPrivateKey) -> str:
    encrypted_symmetric_key = base64.b64decode(payload.encrypted_symmetric_key)
    try:
        decrypted_symmetric_key = private_key.decrypt(
            encrypted_symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
    except ValueError:
        raise HTTPException(status_code=401, detail="Oi, I can't decrypt that symmetric key, sorry")
    base64_symmetric_key = base64.urlsafe_b64encode(decrypted_symmetric_key).decode()
    return base64_symmetric_key


async def decrypt_symmetric_key_exchange_payload(
    config: Config = Depends(get_config), encrypted_payload: bytes = Depends(get_body)
):
    decrypted_data = config.encryption_keys_handler.private_key.decrypt(
        encrypted_payload,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )

    data_dict = json.loads(decrypted_data.decode())
    return SymmetricKeyExchange(**data_dict)

# def decrypt_general_payload(
#     model: Type[T],
#     # encrypted_payload: bytes = Depends(get_body),
#     encrypted_payload,
#     symmetric_key_uuid: str = Header(..., alias=cst.SYMMETRIC_KEY_UUID),
#     validator_hotkey: str = Header(..., alias=cst.VALIDATOR_HOTKEY),
#     miner_hotkey: str = Header(..., alias=cst.MINER_HOTKEY),
#     # config: Config = Depends(get_config),
#     config: Config = Depends(get_config),
# ) -> T:
#     print("this is executed")
#     print(config)
#     print(f"Decrypting payload from validator {validator_hotkey} for miner {miner_hotkey}")
#     symmetric_key_info = config.encryption_keys_handler.get_symmetric_key(validator_hotkey, symmetric_key_uuid)
#     if not symmetric_key_info:
#         raise HTTPException(status_code=400, detail="No symmetric key found for that hotkey and uuid")

#     decrypted_data = symmetric_key_info.fernet.decrypt(encrypted_payload)

#     data_dict: dict = json.loads(decrypted_data.decode())

#     return model(**data_dict)


def decrypt_general_payload(
    model,
    encrypted_payload,
    symmetric_key_uuid: str,
    validator_hotkey: str,
    miner_hotkey: str,
    config: Config,
) -> T:
    logger.info("Validator Hotkey:")
    logger.info(validator_hotkey)
    logger.info("miner_hotkey:")
    logger.info(miner_hotkey)
    print(f"Decrypting payload from validator {validator_hotkey} for miner {miner_hotkey}")
    
    symmetric_key_info = config.encryption_keys_handler.get_symmetric_key(validator_hotkey, symmetric_key_uuid)
    if not symmetric_key_info:
        raise HTTPException(status_code=400, detail="No symmetric key found for that hotkey and uuid")

    logger.info("passed successfully to get_symmetric_key = ", symmetric_key_info)
    decrypted_data = symmetric_key_info.fernet.decrypt(encrypted_payload)
    
    logger.info("passed data decryption", decrypted_data)
    data_dict: dict = json.loads(decrypted_data.decode())
    content_instance = model.parse_obj(data_dict)
    return content_instance
