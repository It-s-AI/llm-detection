import time
from functools import wraps
from typing import Any, Callable, Tuple, List
import torch

from scalecodec import ScaleType
from substrateinterface import Keypair, SubstrateInterface
from tenacity import retry, stop_after_attempt, wait_exponential

from fiber import constants as fcst
from fiber.chain.chain_utils import format_error_message, query_substrate
from fiber.chain.interface import get_substrate
from fiber.logging_utils import get_logger
from fiber.constants import U16_MAX
from fiber.chain.metagraph import Metagraph



logger = get_logger(__name__)



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _query_subtensor(
    substrate: SubstrateInterface,
    name: str,
    block: int | None = None,
    params: int | None = None,
) -> ScaleType:
    try:
        return substrate.query(
            module="SubtensorModule",
            storage_function=name,
            params=params,  # type: ignore
            block_hash=(None if block is None else substrate.get_block_hash(block)),  # type: ignore
        )
    except Exception:
        # Should prevent SSL errors
        substrate = get_substrate(subtensor_address=substrate.url)
        raise


def _get_hyperparameter(
    substrate: SubstrateInterface,
    param_name: str,
    netuid: int,
    block: int | None = None,
) -> list[int] | int | None:
    subnet_exists = getattr(
        _query_subtensor(substrate, "NetworksAdded", block, [netuid]),  # type: ignore
        "value",
        False,
    )
    if not subnet_exists:
        return None
    return getattr(
        _query_subtensor(substrate, param_name, block, [netuid]),  # type: ignore
        "value",
        None,
    )


def _blocks_since_last_update(substrate: SubstrateInterface, netuid: int, node_id: int) -> int | None:
    current_block = substrate.get_block_number(None)  # type: ignore
    last_updated = _get_hyperparameter(substrate, "LastUpdate", netuid)
    assert not isinstance(last_updated, int), "LastUpdate should be a list of ints"
    if last_updated is None:
        return None
    return current_block - int(last_updated[node_id])


def _min_interval_to_set_weights(substrate: SubstrateInterface, netuid: int) -> int:
    weights_set_rate_limit = _get_hyperparameter(substrate, "WeightsSetRateLimit", netuid)
    assert isinstance(weights_set_rate_limit, int), "WeightsSetRateLimit should be an int"
    return weights_set_rate_limit


def _normalize_and_quantize_weights(node_ids: list[int], node_weights: list[float]) -> tuple[list[int], list[int]]:
    if len(node_ids) != len(node_weights) or any(uid < 0 for uid in node_ids) or any(weight < 0 for weight in node_weights):
        raise ValueError("Invalid input: length mismatch or negative values")
    if not any(node_weights):
        return [], []
    scaling_factor = fcst.U16_MAX / max(node_weights)

    node_weights_formatted = []
    node_ids_formatted = []
    for node_id, node_weight in zip(node_ids, node_weights):
        if node_weight > 0:
            node_ids_formatted.append(node_id)
            node_weights_formatted.append(round(node_weight * scaling_factor))

    return node_ids_formatted, node_weights_formatted


def _log_and_reraise(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {str(e)}")
            raise

    return wrapper


def _can_set_weights(substrate: SubstrateInterface, netuid: int, validator_node_id: int) -> bool:
    blocks_since_update = _blocks_since_last_update(substrate, netuid, validator_node_id)
    min_interval = _min_interval_to_set_weights(substrate, netuid)
    if min_interval is None:
        return True
    return blocks_since_update is not None and blocks_since_update > min_interval


def _send_weights_to_chain(
    substrate: SubstrateInterface,
    keypair: Keypair,
    node_ids: list[int],
    node_weights: list[float],
    netuid: int,
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
) -> tuple[bool, str | None]:
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1.5, min=2, max=5),
        reraise=True,
    )
    @_log_and_reraise
    def _set_weights():
        with substrate as si:
            rpc_call = si.compose_call(
                call_module="SubtensorModule",
                call_function="set_weights",
                call_params={
                    "dests": node_ids,
                    "weights": node_weights,
                    "netuid": netuid,
                    "version_key": version_key,
                },
            )
            extrinsic_to_send = si.create_signed_extrinsic(call=rpc_call, keypair=keypair, era={"period": 5})

            response = si.submit_extrinsic(
                extrinsic_to_send,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            if not wait_for_finalization and not wait_for_inclusion:
                return True, "Not waiting for finalization or inclusion."
            response.process_events()

            if response.is_success:
                return True, "Successfully set weights."

            return False, format_error_message(response.error_message)

    return _set_weights()


def set_node_weights(
    substrate: SubstrateInterface,
    keypair: Keypair,
    node_ids: list[int],
    node_weights: list[float],
    netuid: int,
    validator_node_id: int,
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    max_attempts: int = 1,
) -> bool:
    node_ids_formatted, node_weights_formatted = _normalize_and_quantize_weights(node_ids, node_weights)

    # Fetch a new substrate object to reset the connection
    substrate = get_substrate(subtensor_address=substrate.url)

    weights_can_be_set = False
    for attempt in range(1, max_attempts + 1):
        if not _can_set_weights(substrate, netuid, validator_node_id):
            logger.info(logger.info(f"Skipping attempt {attempt}/{max_attempts}. Too soon to set weights. Will wait 30 secs..."))
            time.sleep(30)
            continue
        else:
            weights_can_be_set = True
            break

    if not weights_can_be_set:
        logger.error("No attempt to set weightsmade. Perhaps it is too soon to set weights!")
        return False

    logger.info("Attempting to set weights...")
    success, error_message = _send_weights_to_chain(
        substrate,
        keypair,
        node_ids_formatted,
        node_weights_formatted,
        netuid,
        version_key,
        wait_for_inclusion,
        wait_for_finalization,
    )

    if not wait_for_finalization and not wait_for_inclusion:
        logger.info("Not waiting for finalization or inclusion to set weights. Returning immediately.")
        return success

    if success:
        if wait_for_finalization:
            logger.info("✅ Successfully set weights and finalized")
        elif wait_for_inclusion:
            logger.info("✅ Successfully set weights and included")
        else:
            logger.info("✅ Successfully set weights")
    else:
        logger.error(f"❌ Failed to set weights: {error_message}")

    substrate.close()
    return success

def process_weights_for_netuid(
    uids,
    weights: torch.Tensor,
    netuid: int,
    metagraph: Metagraph = None,
    exclude_quantile: int = 0,
) -> torch.FloatTensor:

    # Get latest metagraph from chain if metagraph is None.
    if metagraph == None:
        logger.error("metagraph is none when performing process_weights_for_netuid")

    # Cast weights to floats.
    if not isinstance(weights, torch.FloatTensor):
        weights = weights.type(torch.float32)

    # Network configuration parameters from an subtensor.
    # These parameters determine the range of acceptable weights for each neuron.
    quantile = exclude_quantile / U16_MAX
    # min_allowed_weights = subtensor.min_allowed_weights(netuid=netuid)
    # max_weight_limit = subtensor.max_weight_limit(netuid=netuid)

    substrate = SubstrateInterface(
        ss58_format=42,
        use_remote_preset=True,
        url="wss://entrypoint-finney.opentensor.ai:443",
    )
    min_allowed_weights = _query_subtensor(
        substrate=substrate,
        name = "MinAllowedWeights",
        params = [netuid]
    )
    max_weight_limit = _query_subtensor(
        substrate=substrate,
        name = "MaxWeightsLimit",
        params = [netuid]
    )
    logger.debug("quantile", quantile)
    logger.debug("min_allowed_weights", min_allowed_weights)
    logger.debug("max_weight_limit", max_weight_limit)

    

    # Find all non zero weights.
    non_zero_weight_idx = torch.argwhere(weights > 0).squeeze(dim=1)
    non_zero_weight_uids = uids[non_zero_weight_idx]
    non_zero_weights = weights[non_zero_weight_idx]
    if non_zero_weights.numel() == 0 or metagraph.n < min_allowed_weights:
        logger.warning("No non-zero weights returning all ones.")
        final_weights = torch.ones((metagraph.n)).to(metagraph.n) / metagraph.n
        logger.debug("final_weights", final_weights)
        return torch.tensor(list(range(len(final_weights)))), final_weights

    elif non_zero_weights.numel() < min_allowed_weights:
        logger.warning(
            "No non-zero weights less then min allowed weight, returning all ones."
        )
        # ( const ): Should this be torch.zeros( ( metagraph.n ) ) to reset everyone to build up weight?
        weights = (
            torch.ones((metagraph.n)).to(metagraph.n) * 1e-5
        )  # creating minimum even non-zero weights
        weights[non_zero_weight_idx] += non_zero_weights
        logger.debug("final_weights", weights)
        normalized_weights = normalize_max_weight(
            x=weights, limit=max_weight_limit
        )
        return torch.tensor(list(range(len(normalized_weights)))), normalized_weights

    logger.debug("non_zero_weights", non_zero_weights)

    # Compute the exclude quantile and find the weights in the lowest quantile
    max_exclude = max(0, len(non_zero_weights) - min_allowed_weights) / len(
        non_zero_weights
    )
    exclude_quantile = min([quantile, max_exclude])
    lowest_quantile = non_zero_weights.quantile(exclude_quantile)
    logger.debug("max_exclude", max_exclude)
    logger.debug("exclude_quantile", exclude_quantile)
    logger.debug("lowest_quantile", lowest_quantile)

    # Exclude all weights below the allowed quantile.
    non_zero_weight_uids = non_zero_weight_uids[lowest_quantile <= non_zero_weights]
    non_zero_weights = non_zero_weights[lowest_quantile <= non_zero_weights]
    logger.debug("non_zero_weight_uids", non_zero_weight_uids)
    logger.debug("non_zero_weights", non_zero_weights)

    # Normalize weights and return.
    normalized_weights = normalize_max_weight(
        x=non_zero_weights, limit=max_weight_limit
    )
    logger.debug("final_weights", normalized_weights)

    return non_zero_weight_uids, normalized_weights


def normalize_max_weight(
    x: torch.FloatTensor, limit: float = 0.1
) -> "torch.FloatTensor":
    r"""Normalizes the tensor x so that sum(x) = 1 and the max value is not greater than the limit.
    Args:
        x (:obj:`torch.FloatTensor`):
            Tensor to be max_value normalized.
        limit: float:
            Max value after normalization.
    Returns:
        y (:obj:`torch.FloatTensor`):
            Normalized x tensor.
    """
    epsilon = 1e-7  # For numerical stability after normalization

    weights = x.clone()
    values, _ = torch.sort(weights)

    if x.sum() == 0 or len(x) * limit <= 1:
        return torch.ones_like(x) / x.size(0)
    else:
        estimation = values / values.sum()

        if estimation.max() <= limit:
            return weights / weights.sum()

        # Find the cumlative sum and sorted tensor
        cumsum = torch.cumsum(estimation, 0)

        # Determine the index of cutoff
        estimation_sum = torch.tensor(
            [(len(values) - i - 1) * estimation[i] for i in range(len(values))]
        )
        n_values = (estimation / (estimation_sum + cumsum + epsilon) < limit).sum()

        # Determine the cutoff based on the index
        cutoff_scale = (limit * cumsum[n_values - 1] - epsilon) / (
            1 - (limit * (len(estimation) - n_values))
        )
        cutoff = cutoff_scale * values.sum()

        # Applying the cutoff
        weights[weights > cutoff] = cutoff

        y = weights / weights.sum()

        return y
    
    
    
def convert_weights_and_uids_for_emit(
    uids: torch.LongTensor, weights: torch.FloatTensor
) -> Tuple[List[int], List[int]]:
    r"""Converts weights into integer u32 representation that sum to MAX_INT_WEIGHT.
    Args:
        uids (:obj:`torch.LongTensor,`):
            Tensor of uids as destinations for passed weights.
        weights (:obj:`torch.FloatTensor,`):
            Tensor of weights.
    Returns:
        weight_uids (List[int]):
            Uids as a list.
        weight_vals (List[int]):
            Weights as a list.
    """
    # Checks.
    weights = weights.tolist()
    uids = uids.tolist()
    if min(weights) < 0:
        raise ValueError(
            "Passed weight is negative cannot exist on chain {}".format(weights)
        )
    if min(uids) < 0:
        raise ValueError("Passed uid is negative cannot exist on chain {}".format(uids))
    if len(uids) != len(weights):
        raise ValueError(
            "Passed weights and uids must have the same length, got {} and {}".format(
                len(uids), len(weights)
            )
        )
    if sum(weights) == 0:
        return [], []  # Nothing to set on chain.
    else:
        max_weight = float(max(weights))
        weights = [
            float(value) / max_weight for value in weights
        ]  # max-upscale values (max_weight = 1).

    weight_vals = []
    weight_uids = []
    for i, (weight_i, uid_i) in enumerate(list(zip(weights, uids))):
        uint16_val = round(
            float(weight_i) * int(U16_MAX)
        )  # convert to int representation.

        # Filter zeros
        if uint16_val != 0:  # Filter zeros
            weight_vals.append(uint16_val)
            weight_uids.append(uid_i)

    return weight_uids, weight_vals
