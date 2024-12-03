import os
import asyncio
import httpx
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from fiber.chain import chain_utils
from fiber.logging_utils import get_logger
from fiber.validator import client as vali_client
from fiber.validator import handshake
from protocol import TextRequest
import time
logger = get_logger(__name__)

async def send_queries(httpx_client, miner_address, fernet, keypair, symmetric_key_uuid, miner_hotkey_ss58_address, payload):
    for i in range(0, 5):
        await asyncio.sleep(3)  # Wait for 3 seconds before sending the next query
        logger.info("************ Just sent the query")
        
        # Create a task for the query without waiting for the response
        asyncio.create_task(
            vali_client.make_non_streamed_post(
                httpx_client=httpx_client,
                server_address=miner_address,
                fernet=fernet,
                keypair=keypair,
                symmetric_key_uuid=symmetric_key_uuid,
                validator_ss58_address=keypair.ss58_address,
                miner_ss58_address=miner_hotkey_ss58_address,
                payload=payload,
                endpoint="/detection-request",
                timeout=15
            )
        )

async def main():
    # Load needed stuff
    load_dotenv()
    wallet_name = os.getenv("WALLET_NAME", "default")
    hotkey_name = os.getenv("HOTKEY_NAME_2", "default")
    print(wallet_name, hotkey_name)
    keypair = chain_utils.load_hotkey_keypair(wallet_name, hotkey_name)
    httpx_client = httpx.AsyncClient()

    # Handshake with miner
    miner_address = "http://108.236.147.253:51685"
    miner_hotkey_ss58_address = "5CaFsXR78pDfrZd7xRPc79tcUFJaM8fDx9MkQ37qhWYuJ7M5"
    symmetric_key_str, symmetric_key_uuid = await handshake.perform_handshake(
        keypair=keypair,
        httpx_client=httpx_client,
        server_address=miner_address,
        miner_hotkey_ss58_address=miner_hotkey_ss58_address,
    )

    if symmetric_key_str is None or symmetric_key_uuid is None:
        raise ValueError("Symmetric key or UUID is None :-(")
    else:
        logger.info("Wohoo - handshake worked! :)")

    payload = {
        "texts": ["Hello world", "Pydantic is great!", "Sample text for testing."],
        "predictions": [[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]],
        "version": "1.1.6"
    }

    fernet = Fernet(symmetric_key_str)

    # Start sending queries
    # await send_queries(httpx_client, miner_address, fernet, keypair, symmetric_key_uuid, miner_hotkey_ss58_address, payload) # mutliple message handliing
    
    response = await vali_client.make_non_streamed_post(
        httpx_client=httpx_client,
        server_address=miner_address,
        fernet=fernet,
        keypair=keypair,
        symmetric_key_uuid=symmetric_key_uuid,
        validator_ss58_address=keypair.ss58_address,
        miner_ss58_address=miner_hotkey_ss58_address,
        payload=payload,
        endpoint="/detection-request",
        timeout=20
    )
    logger.info("send the request to miner successfully.")
    
    logger.info("have got the answer from miner: ")
    logger.info(response.text)    
    logger.info("Escaping the validation mechanism")

if __name__ == "__main__":
    asyncio.run(main())
