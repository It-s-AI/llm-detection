import json
import logging
import random
import time
from abc import abstractmethod

import bittensor as bt
import numpy as np
from datasets import load_dataset
from collections.abc import Iterator


class TextDataset(Iterator):
    def __init__(self, max_prompt_len, text_field):
        super().__init__()
        self.pile = self.init_dataset()
        self.max_prompt_len = max_prompt_len
        self.text_field = text_field

    @abstractmethod
    def get_iter(self):
        ...

    def filter_rules_pass(self, sample):
        if random.random() > 0.01:
            return False
        return True

    def init_dataset(self):
        try:
            dataset = self.get_iter()
            return dataset
        except Exception as e:
            logging.error("Got exception during Pile dataset initializing: {}, retrying...".format(e))
            time.sleep(60)
            return self.init_dataset()

    def __next__(self):
        while True:
            try:
                el = next(self.pile)
                if not self.filter_rules_pass(el):
                    continue

                document_text = el[self.text_field][:int(self.max_prompt_len * 1.25)]
                context_len = int(len(document_text) * np.random.uniform(0.25, 0.75))
                prompt = document_text[:context_len]
                return {'prompt': prompt, 'real_completion': el[self.text_field][context_len:]}
            except Exception as e:
                if type(e) == StopIteration:
                    bt.logging.info('PilePromptDataset ended: reinitializing it')
                else:
                    bt.logging.error("Got exception during loading data from PilePromptDataset, reinitializing it: {}".format(e))
                    bt.logging.exception(e)

                self.pile = self.init_dataset()
                continue


class PileDataset(TextDataset):
    def __init__(self, max_prompt_len):
        super().__init__(max_prompt_len, 'text')

    def get_iter(self):
        seed = random.randint(0, 1000)
        dataset = iter(
            load_dataset("monology/pile-uncopyrighted", streaming=True)['train'].shuffle(
                seed=seed, buffer_size=100000
            )
        )
        return dataset


class RedPajamaDataset(TextDataset):
    def __init__(self, max_prompt_len):
        super().__init__(max_prompt_len, 'raw_content')

    def get_iter(self):
        seed = random.randint(0, 1000)
        dataset = iter(
            load_dataset(
                "togethercomputer/RedPajama-Data-V2",
                snapshots=["2023-14"],
                languages=["en"],
                name="default",
                streaming=True,
            ).shuffle(seed=seed, buffer_size=100000)
        )
        return dataset

    def filter_rules_pass(self, sample):
        if random.random() > 0.01 or not self.c4_rules_pass(sample):
            return False
        return True

    def c4_rules_pass(self, sample) -> bool:
        """ function returns True if the sample complies with the filtering rules used in C4 """
        signals = json.loads(sample["quality_signals"])

        # rule 1: at least 3 sentences
        num_sentences = signals["rps_doc_num_sentences"][0][2]
        if num_sentences < 3:
            return False

        # rule 2: page may not contain bad words
        n_bad_words = signals["rps_doc_ldnoobw_words"][0][2]
        if n_bad_words > 0:
            return False

        # rule 3: page may not contain placeholder "lorem ipsum" text
        lorem_ipsum = signals["rps_doc_lorem_ipsum"][0][2]
        if lorem_ipsum > 0:
            return False

        return True


class HumanDataset(Iterator):
    def __init__(self, max_prompt_len=1500):
        super().__init__()
        self.pile_dataset = PileDataset(max_prompt_len)
        self.red_pajama_dataset = RedPajamaDataset(max_prompt_len)

    def __next__(self) -> dict:
        res = {}
        if random.random() > 0.5:
            el = next(self.pile_dataset)
            res['data_source'] = 'pile'
        else:
            el = next(self.red_pajama_dataset)
            res['data_source'] = 'red_pajama'

        res['text'] = el['real_completion']
        return res


class PromptDataset(Iterator):
    def __init__(self, max_prompt_len=1500):
        super().__init__()
        self.pile_dataset = PileDataset(max_prompt_len)
        self.red_pajama_dataset = RedPajamaDataset(max_prompt_len)
        self.max_prompt_len = max_prompt_len

    def __next__(self) -> dict:
        while True:
            res = {}
            if random.random() > 0.5:
                el = next(self.pile_dataset)
                res['data_source'] = 'pile'
            else:
                el = next(self.red_pajama_dataset)
                res['data_source'] = 'red_pajama'

            if len(el['prompt']) > self.max_prompt_len:
                bt.logging.info("Prompt has len {}, truncating it to {} chars".format(len(el['prompt']), self.max_prompt_len))

            res['prompt'] = el["prompt"][:self.max_prompt_len]
            if res['prompt'].strip():
                return res


if __name__ == '__main__':
    dataset = HumanDataset()
    print(next(dataset))

    dataset = PromptDataset()
    for i in range(2):
        print(next(dataset))
