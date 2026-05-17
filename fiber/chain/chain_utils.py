import json
from pathlib import Path
from typing import Any

from scalecodec import ScaleBytes, ScaleType
from scalecodec.base import RuntimeConfiguration
from scalecodec.type_registry import load_type_registry_preset
from substrateinterface import Keypair

from fiber import SubstrateInterface
from fiber.chain import chain_utils as utils
from fiber.chain import type_registries
from fiber.logging_utils import get_logger

logger = get_logger(__name__)


def create_scale_object_from_scale_bytes(return_type: str, as_scale_bytes: ScaleBytes) -> ScaleType:
    custom_rpc_type_registry = type_registries.get_custom_type_registry()
    rpc_runtime_config = RuntimeConfiguration()
    rpc_runtime_config.update_type_registry(load_type_registry_preset("legacy"))
    rpc_runtime_config.update_type_registry(custom_rpc_type_registry)
    scale_object = rpc_runtime_config.create_scale_object(return_type, as_scale_bytes)
    return scale_object


def create_scale_object_from_scale_encoding(
    input_: list[int] | bytes | ScaleBytes,
    type_name: str,
    is_vec: bool = False,
    is_option: bool = False,
) -> dict | None:
    type_string = type_name
    if is_option:
        type_string = f"Option<{type_string}>"
    if is_vec:
        type_string = f"Vec<{type_string}>"

    if isinstance(input_, ScaleBytes):
        as_scale_bytes = input_
    else:
        if isinstance(input_, list) and all([isinstance(i, int) for i in input_]):
            vec_u8 = input_
            as_bytes = bytes(vec_u8)
        elif isinstance(input_, bytes):
            as_bytes = input_
        else:
            raise TypeError("input_ must be a List[int], bytes, or ScaleBytes")

        as_scale_bytes = ScaleBytes(as_bytes)

    scale_object = utils.create_scale_object_from_scale_bytes(type_string, as_scale_bytes)

    return scale_object.decode()


def format_error_message(error_message: dict | None) -> str:
    err_type, err_name, err_description = (
        "UnknownType",
        "UnknownError",
        "Unknown Description",
    )
    if isinstance(error_message, dict):
        err_type = error_message.get("type", err_type)
        err_name = error_message.get("name", err_name)
        err_description = error_message.get("docs", [err_description])[0]
    return f"substrate returned `{err_name} ({err_type})` error. Description: `{err_description}`"


def get_hotkey_file_path(wallet_name: str, hotkey_name: str) -> Path:
    file_path = Path.home() / ".bittensor" / "wallets" / wallet_name / "hotkeys" / hotkey_name
    return file_path


def get_coldkeypub_file_path(wallet_name: str) -> Path:
    file_path = Path.home() / ".bittensor" / "wallets" / wallet_name / "coldkeypub.txt"
    return file_path


def load_coldkeypub_keypair(wallet_name: str) -> Keypair:
    file_path = get_coldkeypub_file_path(wallet_name)
    try:
        with open(file_path, "r") as file:
            keypair_data = json.load(file)
        keypair = Keypair(ss58_address=keypair_data["ss58Address"])
        logger.info(f"Loaded keypair from {file_path}")
        return keypair
    except Exception as e:
        raise ValueError(f"Failed to load keypair: {str(e)}")


def load_hotkey_keypair(wallet_name: str, hotkey_name: str) -> Keypair:
    file_path = get_hotkey_file_path(wallet_name, hotkey_name)
    try:
        with open(file_path, "r") as file:
            keypair_data = json.load(file)
        keypair = Keypair.create_from_seed(keypair_data["secretSeed"])
        logger.info(f"Loaded keypair from {file_path}")
        return keypair
    except Exception as e:
        raise ValueError(f"Failed to load keypair: {str(e)}")


def sign_message(keypair: Keypair, message: str | None) -> str | None:
    if message is None:
        return None
    return f"0x{keypair.sign(message).hex()}"



def query_substrate(
    substrate: SubstrateInterface, module: str, method: str, params: list[Any], return_value: bool = True
) -> tuple[SubstrateInterface, Any]:
    try:
        query_result = substrate.query(module, method, params)

        return_val = query_result.value if return_value else query_result

        return substrate, return_val
    except Exception as e:
        logger.error(f"Query failed with error: {e}. Reconnecting and retrying.")

        substrate = SubstrateInterface(url=substrate.url)

        query_result = substrate.query(module, method, params)

        return_val = query_result.value if return_value else query_result

        return substrate, return_val


            # with self.substrate as substrate:
            #     return substrate.query(
            #         module="SubtensorModule",
            #         storage_function=name,
            #         params=params,
            #         block_hash=(
            #             None if block is None else substrate.get_block_hash(block)
            #         ),
            #     )