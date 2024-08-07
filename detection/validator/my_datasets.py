import logging
import random
import time

import bittensor as bt
import numpy as np
from datasets import load_dataset
from collections.abc import Iterator
from pylatexenc.latex2text import LatexNodes2Text

from detection.validator.prompt_generator import PromptGenerator


class HumanDataset(Iterator):
    def __init__(self):
        super().__init__()
        self.pile_prompt_dataset = PilePromptDataset(1500)

    def __next__(self) -> dict:
        el = next(self.pile_prompt_dataset)
        return {'text': el['real_completion'], 'data_source': 'pile_human', 'topic': el['topic']}


class PilePromptDataset(Iterator):
    def __init__(self, max_prompt_len):
        super().__init__()
        self.pile = self.init_dataset()
        self.max_prompt_len = max_prompt_len

    def init_dataset(self):
        try:
            seed = random.randint(0, 1000)
            dataset = iter(
                load_dataset("monology/pile-uncopyrighted", streaming=True)['train'].shuffle(
                    seed=seed, buffer_size=100000
                )
            )
            return dataset
        except Exception as e:
            logging.error("Got exception during Pile dataset initializing: {}, retrying...".format(e))
            time.sleep(60)
            return self.init_dataset()

    def __next__(self):
        while True:
            try:
                el = next(self.pile)
                if random.random() > 0.01:
                    continue

                document_text = el['text'][:int(self.max_prompt_len * 1.25)]
                context_len = int(len(document_text) * np.random.uniform(0.25, 0.75))
                # prompt = self.generate_prompt(document_text[:context_len])[:self.max_prompt_len]
                prompt = document_text[:context_len]
                return {'prompt': prompt, 'topic': el['meta']['pile_set_name'], 'real_completion': el['text'][context_len:]}
            except Exception as e:
                if type(e) == StopIteration:
                    bt.logging.info('PilePromptDataset ended: reinitializing it')
                else:
                    bt.logging.error("Got exception during loading data from PilePromptDataset, reinitializing it: {}".format(e))
                    bt.logging.exception(e)

                self.pile = self.init_dataset()
                continue


class PromptDataset(Iterator):
    def __init__(self, max_prompt_len=1500):
        super().__init__()
        self.pile_prompt_dataset = PilePromptDataset(max_prompt_len)
        self.prompt_generator = PromptGenerator()
        self.max_prompt_len = max_prompt_len

    def __next__(self) -> dict:
        while True:
            res = {}
            el = next(self.pile_prompt_dataset)
            res['data_source'] = 'pile'

            if len(el['prompt']) > self.max_prompt_len:
                bt.logging.info("Prompt has len {}, truncating it to {} chars".format(len(el['prompt']), self.max_prompt_len))

            res['prompt'] = el["prompt"][:self.max_prompt_len]
            res['topic'] = el['task'] if res['data_source'] == 'prompt_generator' else el['topic']

            # Check if the text is not empty or does not consist only of newline characters
            if res['prompt'].strip():
                return res


if __name__ == '__main__':
    dataset = HumanDataset()
    print(next(dataset))

    dataset = PromptDataset()
    for i in range(2):
        print(next(dataset))
