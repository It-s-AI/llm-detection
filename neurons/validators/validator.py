# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>
import os
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


import time

# Bittensor
import bittensor as bt
import numpy as np
import torch

# Bittensor Validator Template:
import template
from neurons.validators.data_generator import DataGenerator
from neurons.validators.text_completion import OpenAiModel, OllamaModel
from template.validator import forward

# import base validator class which takes care of most of the boilerplate
from template.base.validator import BaseValidatorNeuron
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, average_precision_score
import openai


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        openai.api_key = os.getenv("OPENAI_API_KEY")
        models = [OpenAiModel(model_name='gpt-3.5-turbo'),
                  OpenAiModel(model_name='gpt-4-turbo-preview'),
                  OllamaModel(model_name='vicuna'),
                  OllamaModel(model_name='mistral')]
        self.generator = DataGenerator(models, [0.25, 0.25, 0.25, 0.25])

    async def build_queries(self) -> (list[str], np.array):
        data = self.generator.generate_data(n_samples=200)
        return [el.text for el in data], np.array([int(el.label) for el in data])

    async def count_reward(self, y_true: np.array, y_pred: np.array) -> float:
        preds = y_pred.astype(int)

        # accuracy = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        f1 = f1_score(y_true, preds)
        ap_score = average_precision_score(y_true, y_pred)

        res = {'fp_score': 1 - fp / len(y_pred),
               'f1_score': f1,
               'ap_score': ap_score}
        reward = sum([v for v in res.values()]) / len(res)
        return reward

    async def count_penalty(self, y_pred: np.array, y_true: np.array) -> float:
        bad = np.any((y_pred < 0) | (y_pred > 1))
        return 0 if bad else 1

    async def get_uids(self):
        # return miners uids, which we want to validate
        return []

    async def query_miner(self, uid, text: str) -> float:
        return random.random()

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """

        uids = await self.get_uids()
        texts, y_true = await self.build_queries()

        rewards = []
        for uid in uids:
            y_pred = np.array([await self.query_miner(uid, text) for text in texts])
            reward = await self.count_reward(y_true, y_pred)
            reward *= await self.count_penalty(y_true, y_pred)
            rewards.append(reward)

        self.update_scores(torch.FloatTensor(rewards), uids)

        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Validator running...", time.time())
            time.sleep(5)
