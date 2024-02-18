# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

from detection.protocol import TextSynapse
from detection.validator.reward import get_rewards
from detection.utils.uids import get_random_uids

import time
from typing import List


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

    # if self.step % 100:
    #     return
    
    start_time = time.time()
    bt.logging.info(f"ALL axons {self.metagraph.axons}")

    available_axon_size = len(self.metagraph.axons) - 1 # Except our own
    bt.logging.info(f"available_axon_size {available_axon_size}")

    miner_selection_size = min(available_axon_size, self.config.neuron.sample_size)
    bt.logging.info(f"miner_selection_size {miner_selection_size}")

    miner_uids = get_random_uids(self, k=miner_selection_size)
    bt.logging.info(f"miner_uids {miner_uids}")

    axons = [self.metagraph.axons[uid] for uid in miner_uids]
    bt.logging.info(f"axons {axons}")

    texts, labels = await self.build_queries()
    bt.logging.info(f"texts {texts}")
    bt.logging.info(f"labels {labels}")


    responses: List[TextSynapse]  = await self.dendrite(
        axons=axons,
        synapse=TextSynapse(texts=texts, predictions=[]),
        deserialize=True,
        timeout=self.config.neuron.timeout,
    )

    # Log the results for monitoring purposes.
    bt.logging.info(f"Received responses: {responses}")

    # TODO(developer): Define how the validator scores responses.
    # Adjust the scores based on responses from miners.
    rewards = get_rewards(self, labels=labels, responses=responses)

    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    self.update_scores(rewards, miner_uids)
