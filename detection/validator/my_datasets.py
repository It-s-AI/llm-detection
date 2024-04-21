import logging
import random
import bittensor as bt
import numpy as np
from datasets import load_dataset
from collections.abc import Iterator
from pylatexenc.latex2text import LatexNodes2Text

from detection.validator.prompt_generator import PromptGenerator


class HumanDataset(Iterator):
    def __init__(self):
        super().__init__()
        self.pile = self.init_dataset()
        self.pile_prompt_dataset = PilePromptDataset(2048)

    def init_dataset(self):
        seed = random.randint(0, 1000)
        pile = iter(
            load_dataset("monology/pile-uncopyrighted", streaming=True)['train'].shuffle(
                seed=seed, buffer_size=1000
            )
        )
        return pile

    def get_next_pile(self):
        while True:
            try:
                el = next(self.pile)
            except Exception as e:
                if type(e) == StopIteration:
                    bt.logging.info('Human dataset ended: reinitializing it')
                else:
                    bt.logging.error("Got exception during loading data from human dataset, reinitializing it")
                    bt.logging.exception(e)

                self.pile = self.init_dataset()
                continue

            res = {'text': el['text'], 'data_source': 'pile_human', 'topic': el['meta']['pile_set_name']}
            return res

    def __next__(self) -> dict:
        el = next(self.pile_prompt_dataset)
        return {'text': el['real_completion'], 'data_source': 'pile_human', 'topic': el['topic']}


class PilePromptDataset(Iterator):
    def __init__(self, max_prompt_len):
        super().__init__()
        self.pile = self.init_dataset()
        self.max_prompt_len = max_prompt_len

    def init_dataset(self):
        seed = random.randint(0, 1000)
        dataset = iter(
            load_dataset("monology/pile-uncopyrighted", streaming=True)['train'].shuffle(
                seed=seed, buffer_size=1000
            )
        )
        return dataset

    def generate_prompt(self, context):
        payload_cuttoff = int(len(context) * np.random.uniform(0.1, 0.9))
        prompt_payload = context[:payload_cuttoff]

        user_request = np.random.choice([
            f'Complete the following document.\n\n"{prompt_payload}"',
            f'Finish writing the following document.\n\n"{prompt_payload}"',
            f'Help me finish writing the following.\n\n"{prompt_payload}"',
            f'Help me complete this: "{prompt_payload}"',
            f'Finish writing the following (be careful not to stop prematurely): "{prompt_payload}"',
        ])

        leading_words = np.random.choice([
            context[payload_cuttoff:],
            f"Here's a plausible continuation for the document: {context[payload_cuttoff:]}",
            f"Sure! Here's the rest of the document: \"{context[payload_cuttoff:]}"
        ])

        return f"A chat.\nUSER: {user_request}\nASSISTANT: {leading_words}"

    def __next__(self):
        while True:
            # if np.random.random() > 0.001:
            #     continue

            try:
                el = next(self.pile)
                document_text = el['text'][:int(self.max_prompt_len * 1.25)]
                context_len = int(len(document_text) * np.random.uniform(0.25, 0.75))
                prompt = self.generate_prompt(document_text[:context_len])[:self.max_prompt_len]
                return {'prompt': prompt, 'topic': el['meta']['pile_set_name'], 'real_completion': el['text'][context_len:]}
            except Exception as e:
                if type(e) == StopIteration:
                    bt.logging.info('PilePromptDataset ended: reinitializing it')
                else:
                    bt.logging.error("Got exception during loading data from PilePromptDataset, reinitializing it")
                    bt.logging.exception(e)

                self.pile = self.init_dataset()
                continue


class HC3PromptDataset(Iterator):
    def __init__(self, max_prompt_len):
        super().__init__()
        self.hc3 = self.init_dataset()
        self.max_prompt_len = max_prompt_len

    def init_dataset(self):
        seed = random.randint(0, 1000)
        hc3 = iter(
            load_dataset("Hello-SimpleAI/HC3", name="all", streaming=True)['train'].shuffle(
                seed=seed, buffer_size=1000
            )
        )
        return hc3

    def __next__(self):
        while True:
            try:
                el = next(self.hc3)
                if random.random() < 0.7:
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

            el['question'] = el['question'].replace("Explain like I'm five.", '').replace("Please explain like I'm five.", '')
            el['question'] = el['question'][:self.max_prompt_len]
            return {'prompt': el['question'], 'topic': el['source']}


class PromptDataset(Iterator):
    def __init__(self, max_prompt_len=2048):
        super().__init__()
        self.hc3_prompt_dataset = HC3PromptDataset(max_prompt_len)
        self.pile_prompt_dataset = PilePromptDataset(max_prompt_len)
        self.prompt_generator = PromptGenerator()
        self.max_prompt_len = max_prompt_len

    def __next__(self) -> dict:
        while True:
            # bt.logging.debug("Retrieving data from PromptDataset...")
            res = {}
            p = random.random()
            if p < 0.2:
                bt.logging.debug("Getting prompt from hc3")
                el = next(self.hc3_prompt_dataset)
                res['data_source'] = 'hc3'
            elif p < 0.5:
                bt.logging.debug("Getting prompt from prompt_generator")
                el = self.prompt_generator.get_challenge(None)
                res['data_source'] = 'prompt_generator'
            else:
                bt.logging.debug("Getting prompt from pile")
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
    for i in range(15):
        print(next(dataset))
