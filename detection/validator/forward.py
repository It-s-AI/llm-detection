# The MIT License (MIT)
 # Copyright © 2024 It's AI
import random

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
import numpy as np

from detection.protocol import TextSynapse
from detection.validator.reward import get_rewards
from detection.utils.uids import get_random_uids

import time
from typing import List
import torch


async def get_all_responses(self, axons, texts, timeout, step=35):
    all_responses = []
    for i in range(0, len(axons), step):
        bt.logging.info(f"Sending challenges to the #{i} subset of miners with size {step}")
        subset_axons = axons[i:i + step]

        responses: List[TextSynapse] = await self.dendrite(
            axons=subset_axons,
            synapse=TextSynapse(texts=texts, predictions=[]),
            deserialize=True,
            timeout=timeout,
        )

        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {len(responses)}")
        all_responses.extend(responses)
        bt.logging.info(f"Overall amount of responses: {len(all_responses)}")
    return all_responses


async def forward(self):
    """
    The forward function is called by the validator every time step.

    It is responsible for querying the network and scoring the responses.

    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

    """
    bt.logging.info("Updating and querying available uids")
    # Define how the validator selects a miner to query, how often, etc.
    # bt.logging.info(f"STEPS {self.step} {self.step%300} {not (self.step % 300)}")

    available_axon_size = len(self.metagraph.axons) - 1 # Except our own
    miner_selection_size = min(available_axon_size, self.config.neuron.sample_size)
    miner_uids = get_random_uids(self, k=miner_selection_size)
    axons = [self.metagraph.axons[uid] for uid in miner_uids]

    start_time = time.time()
    texts, labels = await self.build_queries()
    end_time = time.time()
    bt.logging.info(f"Time to generate challenges: {int(end_time - start_time)}")

    cnt_challenges_for_check = random.randint(1, min(10, len(texts)))
    check_ids = np.random.choice(np.arange(len(texts)), size=cnt_challenges_for_check, replace=False)
    check_responses = await get_all_responses(self, axons, texts[check_ids], self.config.neuron.timeout_10)
    all_responses = await get_all_responses(self, axons, texts, self.config.neuron.timeout_300)

    rewards, metrics = get_rewards(self, labels=labels, responses=all_responses, check_responses=check_responses, check_ids=check_ids)
    bt.logging.info("Miner uids: {}".format(miner_uids))
    bt.logging.info("Rewards: {}".format(rewards))
    bt.logging.info("Metrics: {}".format(metrics))

    rewards_tensor = torch.tensor(rewards).to(self.device)
    uids_tensor = torch.tensor(miner_uids).to(self.device)
    self.update_scores(rewards_tensor, uids_tensor)

    self.log_step(miner_uids, metrics, rewards)
