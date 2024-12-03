import netaddr
from substrateinterface import Keypair, SubstrateInterface
from tenacity import retry, stop_after_attempt, wait_exponential

from fiber.logging_utils import get_logger

logger = get_logger(__name__)


def ip_to_int(str_val: str) -> int:
    return int(netaddr.IPAddress(str_val))


def ip_version(str_val: str) -> int:
    """Returns the ip version (IPV4 or IPV6)."""
    return int(netaddr.IPAddress(str_val).version)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
def post_node_ip_to_chain(
    substrate: SubstrateInterface,
    keypair: Keypair,
    netuid: int,
    external_ip: str,
    external_port: int,
    coldkey_ss58_address: str,
    wait_for_inclusion=False,
    wait_for_finalization=True,
) -> bool:
    params = {
        "version": 1,  # I don't know why we even post this, can we just post 1?
        "ip": ip_to_int(external_ip),
        "port": external_port,
        "ip_type": ip_version(external_ip),
        "netuid": netuid,
        "hotkey": keypair.ss58_address,
        "coldkey": coldkey_ss58_address,
        "protocol": 4,
        "placeholder1": 0,
        "placeholder2": 0,
    }

    logger.info(f"Posting IP to chain. Params: {params}")

    with substrate as si:
        call = si.compose_call("SubtensorModule", "serve_axon", params)
        extrinsic = si.create_signed_extrinsic(call=call, keypair=keypair)
        response = si.submit_extrinsic(extrinsic, wait_for_inclusion, wait_for_finalization)

        if wait_for_inclusion or wait_for_finalization:
            response.process_events()
            if not response.is_success:
                logger.error(f"Failed: {response.error_message}")
            return response.is_success
    return True
