# The MIT License (MIT)
# Copyright © 2024 It's AI
import asyncio
import copy
import datetime as dt
import json
import os
import random
import threading
import time
from datetime import date, datetime, timedelta
from typing import List, Tuple
import numpy as np
import bittensor as bt
import httpx
import torch
import wandb
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from pydantic import ValidationError
from substrateinterface import SubstrateInterface, Keypair

from detection import __version__, WANDB_PROJECT, WANDB_ENTITY, MAX_RUN_STEPS_PER_WANDB_RUN
from detection.attacks.data_augmentation import DataAugmentator
from detection.protocol import TextRequest
from detection.utils.uids import get_random_uids, get_random_nodes
from detection.validator.generate_version import generate_random_version
from detection.validator.models import ValDataRow
from detection.validator.segmentation_processer import SegmentationProcesser
from validator.reward import get_rewards
from fiber.chain import chain_utils
from fiber.chain.metagraph import Metagraph
from fiber.chain.interface import get_substrate
from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.weights import set_node_weights, process_weights_for_netuid, convert_weights_and_uids_for_emit
from fiber.chain.post_ip_to_chain import post_node_ip_to_chain
from fiber.chain.models import Node
from fiber.logging_utils import get_logger
from fiber.validator import client as vali_client
from fiber.validator import handshake
from fiber.validator import handshake, client

# Load environment variables
load_dotenv()

# Initialize logger
logger = get_logger(__name__)

async def send_query_to_nodes(self, nodes: list[Node], synapse: TextRequest, timeout: float) -> List[TextRequest]:
    res: List[TextRequest | None] = [None] * len(nodes)
    wallet_name = self.wallet_name
    hotkey_name = self.hotkey_name
    payload = synapse.dict()
    for i, node in enumerate(nodes):
        httpx_client = httpx.AsyncClient()
        server_address=client.construct_server_address(
            node,
        )
        miner_hotkey_ss58_address = node.hotkey
        symmetric_key_str, symmetric_key_uuid = await handshake.perform_handshake(
            keypair=self.keypair,
            httpx_client=httpx_client,
            server_address=server_address,
            miner_hotkey_ss58_address=miner_hotkey_ss58_address,
        )
        if symmetric_key_str is None or symmetric_key_uuid is None:
            raise ValueError("Symmetric key or UUID is None :-(")
        else:
            logger.info("Wohoo - handshake worked! :)")
        fernet = Fernet(symmetric_key_str)
        
        resp = await vali_client.make_non_streamed_post(
            httpx_client=httpx_client,
            server_address=server_address,
            fernet=fernet,
            keypair=self.keypair,
            symmetric_key_uuid=symmetric_key_uuid,
            validator_ss58_address=self.hotkey,
            miner_ss58_address=miner_hotkey_ss58_address,
            payload=payload,
            endpoint="/detection-request",
        )
        
        resp.raise_for_status()
        logger.info(f"Example request sent! Response: {resp.text}")            
        res[i] = resp    
        
    # assert all([el is not None for el in res])
    return res


async def get_all_responses(self, seleted_nodes: list[Node], queries: List[ValDataRow], check_ids, timeout, step=25, min_text_length=250):
    all_responses = []
    version_responses = []
    check_responses = []
    final_labels = []
    augmentator = DataAugmentator()
    segmentation_processer = SegmentationProcesser()

    for i in range(0, len(seleted_nodes), step):
        logger.info(f"Sending challenges to the #{i} subset of miners with size {step}")
        subset_nodes = seleted_nodes[i:i + step]

        auged_texts = []
        auged_labels = []
        for el in queries:
            text, labels = segmentation_processer.subsample_words(el.text, sum([ell == 0 for ell in el.segmentation_labels]))
            new_text, augs, new_labels = augmentator(text, labels)

            if len(new_text) >= min_text_length:
                auged_texts.append(new_text)
                auged_labels.append(new_labels)
            else:
                auged_texts.append(el.text_auged)
                auged_labels.append(el.auged_segmentation_labels)

        final_labels += [auged_labels] * len(subset_nodes)

        logger.info("Quering check_ids")
        responses: List[TextRequest] = await send_query_to_nodes(
            self,
            nodes=subset_nodes,
            synapse=TextRequest(
                texts=[auged_texts[idx] for idx in check_ids],
                predictions=[],
                version=__version__
            ),
            timeout=timeout,
        )
        check_responses.extend(responses)

        if random.random() < 0.2:
            logger.info("Quering random_version")
            random_version = generate_random_version(
                self.version, self.least_acceptable_version)

            responses: List[TextRequest] = await send_query_to_nodes(
                self,
                nodes = subset_nodes,
                synapse=TextRequest(
                    texts=auged_texts,
                    predictions=[],
                    version=random_version
                ),
                timeout=timeout,
            )
            version_responses.extend(responses)
        else:
            version_responses.extend([TextRequest(predictions=[], texts=[]) for _ in range(len(subset_nodes))])

        logger.info("Quering predictions")
        responses: List[TextRequest] = await send_query_to_nodes(
            self,
            nodes = subset_nodes,
            synapse=TextRequest(
                texts=auged_texts,
                predictions=[],
                version=__version__
            ),
            timeout=timeout,
        )
        all_responses.extend(responses)

        # Log the results for monitoring purposes.
        logger.info(f"Received responses: {len(responses)}")
        logger.info(f"Overall amount of responses: {len(all_responses)}")
    return all_responses, check_responses, version_responses, final_labels


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    logger.info("Updating and querying available uids")
    # Define how the validator selects a miner to query, how often, etc.
    logger.info(f"STEPS {self.step} {self.step%300} {not (self.step % 300)}")

    available_axon_size = len(self.metagraph.nodes) - 1  # Except our own
    miner_selection_size = min(available_axon_size, self.subnet_config.neuron.sample_size)
    seleted_nodes = get_random_nodes(self, miner_selection_size, self.metagraph.nodes)
    # miner_uids = get_random_uids(self, k=miner_selection_size)
    # axons = [self.metagraph.axons[uid] for uid in miner_uids]
    miner_uids = [node.node_id for node in seleted_nodes]
    start_time = time.time()
    queries, labels = await self.build_queries()
    out_of_domain_ids = np.where([el.data_source == 'common_crawl' for el in queries])[0]
    end_time = time.time()
    logger.info(f"Time to generate challenges: {int(end_time - start_time)}")

    cnt_challenges_for_check = random.randint(1, min(10, len(queries)))
    check_ids = np.random.choice(np.arange(len(queries)).astype(int), size=cnt_challenges_for_check, replace=False)
    check_ids = np.array(sorted(check_ids))

    all_responses, check_responses, version_responses, final_labels = await get_all_responses(
        self, seleted_nodes, queries, check_ids, self.subnet_config.neuron.timeout)

    rewards, metrics = get_rewards(self,
                                   labels=final_labels,
                                   responses=all_responses,
                                   check_responses=check_responses,
                                   version_responses=version_responses,
                                   check_ids=check_ids,
                                   out_of_domain_ids=out_of_domain_ids,
                                   update_out_of_domain=True)
    logger.info("Miner uids: {}".format(miner_uids))
    logger.info("Rewards: {}".format(rewards))
    logger.info("Metrics: {}".format(metrics))

    rewards_tensor = torch.tensor(rewards).to(self.device)

    m = torch.nn.Softmax()
    rewards_tensor = m(rewards_tensor * 100)

    logger.info("Normalized rewards: {}".format(rewards_tensor))

    uids_tensor = torch.tensor(miner_uids).to(self.device)
    self.update_scores(rewards_tensor, uids_tensor)

    self.log_step(miner_uids, metrics, rewards)
    
    
    
    