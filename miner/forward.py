import time
import typing
import bittensor as bt

import random

# Bittensor Miner Template:
import detection

from detection.utils.weight_version import is_version_in_range

# import base miner class which takes care of most of the boilerplate
from neurons.miners.ppl_model import PPLModel

from transformers.utils import logging as hf_logging

from neurons.miners.deberta_classifier import DebertaClassifier

hf_logging.set_verbosity(40)

from miner.config import Subnet_Config
from fiber.logging_utils import get_logger
logger = get_logger(__name__)



async def forward(
    request: detection.protocol.TextRequest,
    subnet_config: Subnet_Config,
) -> detection.protocol.TextRequest:
    """
    Processes the incoming 'Textrequest' request by performing a predefined operation on the input data.
    This method should be replaced with actual logic relevant to the miner's purpose.
    Args:
        request (detection.protocol.Textrequest): The request object containing the 'texts' data.
    Returns:
        detection.protocol.Textrequest: The request object with the 'predictions'.
    The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
    the miner's intended operation. This method demonstrates a basic transformation of input data.
    """
    start_time = time.time()
    version = "10.0.0"
    least_acceptable_version = "0.0.0"
    device = subnet_config.neuron.device
    if subnet_config.neuron.model_type == 'ppl':
        model = PPLModel(device=device)
        model.load_pretrained(subnet_config.neuron.ppl_model_path)
    else:
        model = DebertaClassifier(foundation_model_path=subnet_config.neuron.deberta_foundation_model_path,
                                       model_path=subnet_config.neuron.deberta_model_path,
                                       device=subnet_config.neuron.device)
    
    # Check if the validators version is correct
    version_check = is_version_in_range(request.version, version, least_acceptable_version)
    if not version_check:
        return request
    input_data = request.texts
    logger.info(f"Amount of texts recieved: {len(input_data)}")
    try:
        preds = model.predict_batch(input_data)
    except Exception as e:
        logger.error('Couldnt proceed text "{}..."'.format(input_data))
        logger.error(e)
        preds = [0] * len(input_data)
    preds = [[pred] * len(text.split()) for pred, text in zip(preds, input_data)]
    logger.info(f"Made predictions in {int(time.time() - start_time)}s")
    logger.info("Request recieved in Forward func")
    print(request)
    request.predictions = preds
    return request