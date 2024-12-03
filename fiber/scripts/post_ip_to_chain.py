import argparse

from fiber import constants as fcst
from fiber.chain import chain_utils, interface, post_ip_to_chain
from fiber.logging_utils import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Post node IP to chain")
    parser.add_argument(
        "--subtensor.chain_endpoint",
        type=str,
        required=False,
        help="Chain address",
        default=None,
    )
    parser.add_argument(
        "--subtensor.network",
        type=str,
        required=False,
        help="Chain network",
        default=fcst.FINNEY_NETWORK,
    )
    parser.add_argument("--wallet.name", type=str, required=False, help="Wallet name", default="default")
    parser.add_argument("--wallet.hotkey", type=str, required=False, help="Hotkey name", default="default")
    parser.add_argument("--netuid", type=int, required=True, help="Network UID")
    parser.add_argument("--external_ip", required=True, help="External IP address")
    parser.add_argument("--external_port", type=int, required=True, help="External port")

    args = parser.parse_args()

    # Allows us to access with the dot notation
    args_dict = vars(args)
    chain_endpoint = args_dict.get("subtensor.chain_endpoint")
    network = args_dict.get("subtensor.network")
    wallet_name = args_dict.get("wallet.name")
    wallet_hotkey = args_dict.get("wallet.hotkey")

    assert isinstance(wallet_name, str)
    assert isinstance(wallet_hotkey, str)
    assert isinstance(network, str)


    substrate = interface.get_substrate(subtensor_address=chain_endpoint, subtensor_network=network)
    keypair = chain_utils.load_hotkey_keypair(wallet_name=wallet_name, hotkey_name=wallet_hotkey)
    coldkey_keypair_pub = chain_utils.load_coldkeypub_keypair(wallet_name=wallet_name)

    success = post_ip_to_chain.post_node_ip_to_chain(
        substrate=substrate,
        keypair=keypair,
        netuid=args.netuid,
        external_ip=args.external_ip,
        external_port=args.external_port,
        coldkey_ss58_address=coldkey_keypair_pub.ss58_address,
    )
    if success:
        logger.info("Successfully posted IP to chain")
    else:
        logger.error("Failed to post IP to chain :(")


if __name__ == "__main__":
    main()
