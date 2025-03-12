import base64
import os
from functools import lru_cache
from typing import TypeVar

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv
from pydantic import BaseModel

from fiber.chain import chain_utils, interface
from fiber.chain.metagraph import Metagraph
from fiber.miner.core import miner_constants as mcst
from fiber.miner.core.models.config import Config
from fiber.miner.security import key_management, nonce_management

T = TypeVar("T", bound=BaseModel)

load_dotenv()


def _derive_key_from_string(input_string: str, salt: bytes = b"salt_") -> str:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(input_string.encode()))
    return key.decode()


@lru_cache
def factory_config() -> Config:
    nonce_manager = nonce_management.NonceManager()

    wallet_name = os.getenv("WALLET_NAME", "default")
    hotkey_name = os.getenv("HOTKEY_NAME", "default")
    netuid = os.getenv("NETUID")
    subtensor_network = os.getenv("SUBTENSOR_NETWORK")
    subtensor_address = os.getenv("SUBTENSOR_ADDRESS")
    load_old_nodes = bool(os.getenv("LOAD_OLD_NODES", False))
    min_stake_threshold = int(os.getenv("MIN_STAKE_THRESHOLD", 1_000))
    refresh_nodes = os.getenv("REFRESH_NODES", "true").lower() == "true"
    
    assert netuid is not None, "Must set NETUID env var please!"
    
    load_old_nodes = False
    print(load_old_nodes)
    if refresh_nodes:
        substrate = interface.get_substrate(subtensor_network, subtensor_address)
        metagraph = Metagraph(
            substrate=substrate,
            netuid=netuid,
            load_old_nodes=load_old_nodes,
        )
        metagraph.save_nodes()
        print(metagraph.nodes.keys())
    else:
        metagraph = Metagraph(substrate=None, netuid=netuid, load_old_nodes=load_old_nodes)

    keypair = chain_utils.load_hotkey_keypair(wallet_name, hotkey_name)
    cold_keypair = chain_utils.load_coldkeypub_keypair(wallet_name)
    
    # print("\n\n\n wallet keypair", keypair)
    # print(cold_keypair)

    storage_encryption_key = os.getenv("STORAGE_ENCRYPTION_KEY")
    if storage_encryption_key is None:
        storage_encryption_key = _derive_key_from_string(mcst.DEFAULT_ENCRYPTION_STRING)

    encryption_keys_handler = key_management.EncryptionKeysHandler(
        nonce_manager, storage_encryption_key, hotkey=hotkey_name
    )

    return Config(
        encryption_keys_handler=encryption_keys_handler,
        keypair=keypair,
        metagraph=metagraph,
        min_stake_threshold=min_stake_threshold,
        httpx_client=httpx.AsyncClient(),
    )
