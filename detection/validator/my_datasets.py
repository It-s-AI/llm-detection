import logging
import random
import bittensor as bt
from datasets import load_dataset
from collections.abc import Iterator
from pylatexenc.latex2text import LatexNodes2Text


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

    def init_dataset(self):
        seed = random.randint(0, 1000)
        hc3 = iter(
            load_dataset("Hello-SimpleAI/HC3", name="all", streaming=True)['train'].shuffle(
                seed=seed, buffer_size=1000
            )
        )
        return hc3

    def __next__(self) -> dict:
        while True:
            # bt.logging.debug("Retrieving data from PromptDataset...")
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

            if len(el['question']) > 400:
                bt.logging.info("Prompt has len {}, truncating it to 400 chars".format(len(el['question'])))

            res = {'prompt': el["question"][:400], 'data_source': el['source']}

            # Check if the text is not empty or does not consist only of newline characters
            if res['prompt'].strip():
                return res


if __name__ == '__main__':
    dataset = PromptDataset()
    print(next(dataset))

    dataset = HumanDataset()
    for i in range(5):
        print(next(dataset))
