# The MIT License (MIT)
# Copyright © 2024 It's AI

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
import os
import random
import traceback
from typing import List

import bittensor as bt
import torch
import numpy as np

import detection
from detection.validator import forward
from detection.base.validator import BaseValidatorNeuron

from detection.validator.data_generator import DataGenerator
from detection.validator.models import ValDataRow
from detection.validator.text_completion import OllamaModel


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        bt.logging.info("Initializing Validator")

        super(Validator, self).__init__(config=config)

        ollama_url = self.config.neuron.ollama_url

        models = [OllamaModel(model_name='llama2:13b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='llama3:text', base_url=ollama_url),
                   OllamaModel(model_name='llama3:70b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='llama3.1:70b-text-q4_0', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='llama3.2', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='llama3.3:70b', base_url=ollama_url),

                   OllamaModel(model_name='qwen:32b-text-v1.5-q4_0', base_url=ollama_url),
                   OllamaModel(model_name='qwen2.5:14b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='qwen2.5-coder:32b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='qwen2.5:72b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='qwen3:32b', base_url=ollama_url, in_the_middle_generation=True),

                   OllamaModel(model_name='command-r', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='command-r', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),

                   OllamaModel(model_name='gemma2:27b-text-q4_0', base_url=ollama_url),

                   OllamaModel(model_name='mistral-nemo:12b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='mistral-small:22b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='mistral-large:123b', base_url=ollama_url, in_the_middle_generation=True),

                   OllamaModel(model_name='internlm2:7b', base_url=ollama_url),
                   OllamaModel(model_name='internlm2:20b', base_url=ollama_url),
                   OllamaModel(model_name='internlm/internlm2.5:20b-chat', base_url=ollama_url),
                   OllamaModel(model_name='internlm/internlm2.5:latest', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='internlm/internlm2.5:20b-chat', base_url=ollama_url, in_the_middle_generation=True),

                   OllamaModel(model_name='deepseek-v2:16b', base_url=ollama_url),
                   OllamaModel(model_name='deepseek-r1:14b', base_url=ollama_url),
                   OllamaModel(model_name='phi4:14b', base_url=ollama_url),
                   OllamaModel(model_name='aya-expanse:32b', base_url=ollama_url),
                   OllamaModel(model_name='yi:34b-chat', base_url=ollama_url),
                   OllamaModel(model_name='athene-v2:72b', base_url=ollama_url),
                ]

        bt.logging.info(f"Models loaded{models}")

        self.generator = DataGenerator(models, device=self.config.neuron.device)
        bt.logging.info(f"Generator initialized {self.generator}")

        self.out_of_domain_f1_scores = np.ones(257)
        self.out_of_domain_alpha = 0.2

    async def build_queries(self) -> tuple[List[ValDataRow], np.array]:
        bt.logging.info(f"Generating texts for challenges...")
        data = self.generator.generate_data(n_human_samples=30, n_ai_samples=90)
        texts = [el for el in data]
        labels = [el.segmentation_labels for el in data]
        return texts, labels,

    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """

        try:
            res = await forward(self)
            return res
        except Exception as e:
            bt.logging.error("Got error in forward function")
            bt.logging.info(traceback.format_exc())
            return None


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            # bt.logging.info("Validator running...", time.time())
            time.sleep(60)
