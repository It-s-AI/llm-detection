import logging
import random
import bittensor as bt
from datasets import load_dataset
from collections.abc import Iterator
from pylatexenc.latex2text import LatexNodes2Text

from detection.validator.prompt_generator import PromptGenerator


class HumanDataset(Iterator):
    def __init__(self):
        super().__init__()
        self.c4 = self.init_dataset()

    def init_dataset(self):
        seed = random.randint(0, 1000)

        c4 = iter(
            load_dataset("allenai/c4", 'en',  streaming=True)['train'].shuffle(
                seed=seed, buffer_size=1000
            )
        )
        return c4

    def __next__(self) -> dict:
        while True:
            try:
                el = next(self.c4)
            except Exception as e:
                if type(e) == StopIteration:
                    bt.logging.info('Human dataset ended: reinitializing it')
                else:
                    bt.logging.error("Got exception during loading data from human dataset, reinitializing it")
                    bt.logging.exception(e)

                self.c4 = self.init_dataset()
                continue

            res = {'text': el['text'], 'data_source': 'c4_en'}
            return res


class PromptDataset(Iterator):
    def __init__(self):
        super().__init__()
        self.hc3 = self.init_dataset()
        self.prompt_generator = PromptGenerator()

    def init_dataset(self):
        seed = random.randint(0, 1000)
        hc3 = iter(
            load_dataset("Hello-SimpleAI/HC3", name="all", streaming=True)['train'].shuffle(
                seed=seed, buffer_size=1000
            )
        )
        return hc3

    def get_hc3_prompt(self):
        while True:
            try:
                el = next(self.hc3)
                if random.random() < 0.5:
                    while el['source'] == 'reddit_eli5':
                        el = next(self.hc3)
                else:
                    while el['source'] != 'reddit_eli5':
                        el = next(self.hc3)
            except Exception as e:
                if type(e) == StopIteration:
                    bt.logging.info('Prompt dataset ended: reinitializing it')
                else:
                    bt.logging.error("Got exception during loading data from prompt dataset, reinitializing it")
                    bt.logging.exception(e)

                self.hc3 = self.init_dataset()
                continue

            el['prompt'] = el['question']
            return el

    def __next__(self) -> dict:
        while True:
            # bt.logging.debug("Retrieving data from PromptDataset...")
            res = {}
            if random.random() < 0.3:
                bt.logging.debug("Getting prompt from hc3")
                el = self.get_hc3_prompt()
                res['data_source'] = 'hc3'
            else:
                bt.logging.debug("Getting prompt from prompt_generator")
                el = self.prompt_generator.get_challenge(None)
                res['data_source'] = 'prompt_generator'

            if len(el['prompt']) > 400:
                bt.logging.info("Prompt has len {}, truncating it to 400 chars".format(len(el['prompt'])))

            res['prompt'] = el["prompt"][:400]
            res['task_name'] = el['task'] if res['data_source'] == 'prompt_generator' else el['source']

            # Check if the text is not empty or does not consist only of newline characters
            if res['prompt'].strip():
                return res


if __name__ == '__main__':
    dataset = HumanDataset()
    print(next(dataset))

    dataset = PromptDataset()
    for i in range(15):
        print(next(dataset))
