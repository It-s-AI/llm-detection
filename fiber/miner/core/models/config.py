from dataclasses import dataclass

import httpx
from substrateinterface import Keypair

from fiber.chain.metagraph import Metagraph
from fiber.miner.security import key_management


@dataclass
class Config:
    encryption_keys_handler: key_management.EncryptionKeysHandler
    keypair: Keypair
    metagraph: Metagraph
    min_stake_threshold: float
    httpx_client: httpx.AsyncClient
