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


import time
import os
import random
from typing import List

import bittensor as bt
import torch
import numpy as np


import detection
from detection.validator import forward
from detection.base.validator import BaseValidatorNeuron

from detection.validator.data_generator import DataGenerator
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

        bt.logging.info("load_state()")
        self.load_state()

        models = [
            OllamaModel(model_name='vicuna'),
            OllamaModel(model_name='mistral')
        ]
        bt.logging.info(f"Models {models}")

        self.generator = DataGenerator(models, [0.5, 0.5])
        bt.logging.info(f"Generator {self.generator}")


    async def build_queries(self) -> tuple[List[str], np.array]:
        data = self.generator.generate_data(n_human_samples=1, n_ai_samples=1)        
        texts = [el.text for el in data]
        labels = np.array([int(el.label) for el in data])

        # texts = ['\n\n§ INTRODUCTION\n The inflationary universe scenario  in which the early universe undergoes a rapid expansion has been generally accepted as a solution to the horizon problem and some other related problems of the standard big-bang cosmology. The origin of the field that drives inflation is still unknown and is subject to speculations. Among many models of inflation a popular class comprise tachyon inflation models . These models are of particular interest as in these models inflation is driven by the tachyon field originating in string theory. The tachyon potential is derived from string theory and has to satisfy some definite properties to describe tachyon condensation and other requirements in string theory. However, Kofman and Linde have shown  that the slow-roll conditions are not compatible with a string coupling much smaller than one, and the compactification length scale much larger than the Planck length. This leads to the density fluctuations produced during inflation being incompatible with observational constraint on the amplitude of the scalar perturbations. This criticism is based on the string theory motivated values of the parameters in the tachyon potential, i.e., the brane tension and the parameters in the four-dimensional Newton constant obtained via conventional string compactification. Of course, if one relaxes the string theory constraints on the above mentioned parameters, the effective tachyon theory will naturally lead to a type of inflation which will slightly deviate from the conventional inflation based on the canonical scalar field theory. Steer and Vernizzi  have noted a deviation from the standard single field inflation in the second order consistency relations. Based on their analysis they concluded that the tachyon inflation could not be ruled out by the then available observations. It seems like the present observations  could perhaps discriminate between different tachyon models and disfavor or rule out some of these models (for a recent discussion on phenomenological constraints imposed by Planck 2015, see, e.g., ref ).', 'In the United States, the term "middle class" refers to a broad socio-economic group of individuals and families who possess a certain level of economic stability and resources. Middle-class Americans typically have incomes that fall between those of the upper and lower classes, although there is no definitive cutoff for what constitutes middle-class income.\n\nThe US Census Bureau uses various measures to define middle class based on income levels. For example, the middle 60% of households are generally considered to be middle class, which in 2019 corresponded to an annual household income between approximately $48,500 and $137,000. However, this definition may vary depending on the specific economic context and the particular socio-economic indicators being used.\n\nMiddle-class Americans often enjoy a standard of living that allows them to afford basic necessities like housing, food, healthcare, education, and transportation, as well as some discretionary spending for entertainment or savings. They may also have access to certain social and cultural resources, such as stable employment, good schools, and community networks, which contribute to their overall economic security and sense of opportunity.\n\nIt\'s important to note that the concept of the middle class is socially and politically constructed, and its definition can vary over time and across different contexts. Additionally, the economic landscape in the US has shifted significantly in recent decades, with rising inequality and stagnant wages for many workers. As a result, some experts have questioned whether the traditional notion of the middle class remains relevant or accurate in describing contemporary American society.']
        # labels = np.array([0, 1])
        bt.logging.info(f"texts {len(texts)}")
        bt.logging.info(f"labels {labels}")
        return texts, labels


    async def forward(self):
        """
        Validator forward pass. Consists of:
        - Generating the query
        - Querying the miners
        - Getting the responses
        - Rewarding the miners
        - Updating the scores
        """
        return await forward(self)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            # bt.logging.info("Validator running...", time.time())
            time.sleep(36)
