import json
import logging
import random
import time
from abc import abstractmethod
from pathlib import Path

import bittensor as bt
import numpy as np
from datasets import load_dataset
from collections.abc import Iterator

from detection.validator.cc_dataset import CCDataset, get_2023_dumps
from neurons.miners.deberta_classifier import DebertaClassifier


class TextDataset(Iterator):
    def __init__(self, max_prompt_len, text_field):
        super().__init__()
        self.max_prompt_len = max_prompt_len
        self.text_field = text_field
        self.name = 'CommonCrawlDataset' if text_field == 'raw_content' else 'PileDataset'
        self.dataset = self.init_dataset()

    @abstractmethod
    def get_iter(self):
        ...

    def filter_rules_pass(self, prompt, completion):
        if random.random() > 0.01:
            return False
        return True

    def init_dataset(self):
        try:
            dataset = self.get_iter()
            return dataset
        except Exception as e:
            logging.error("Got exception during {} dataset initializing: {}, retrying...".format(self.name, e))
            time.sleep(60)
            return self.init_dataset()

    def __next__(self):
        while True:
            try:
                el = next(self.dataset)

                document_text = el[self.text_field][:int(self.max_prompt_len * 1.25)]
                context_len = int(len(document_text) * np.random.uniform(0.25, 0.75))
                prompt = document_text[:context_len]
                completion = el[self.text_field][context_len:]

                if not self.filter_rules_pass(prompt, completion):
                    continue

                return {'prompt': prompt, 'real_completion': completion}
            except Exception as e:
                if type(e) == StopIteration:
                    bt.logging.info(f'{self.name} with ended: reinitializing it')
                else:
                    bt.logging.error("Got exception during loading data from {}, reinitializing it: {}".format(self.name, e))
                    bt.logging.exception(e)

                self.dataset = self.init_dataset()
                continue


class PileDataset(TextDataset):
    def __init__(self, max_prompt_len):
        super().__init__(max_prompt_len, 'text')

    def get_iter(self):
        seed = int(time.time())
        dataset = iter(
            load_dataset("monology/pile-uncopyrighted", streaming=True)['train'].shuffle(
                seed=seed, buffer_size=100000
            )
        )
        return dataset


class CommonCrawlDataset(TextDataset):
    def __init__(self, max_prompt_len):
        self.dumps_2023 = get_2023_dumps()
        logging.info(f"Found {len(self.dumps_2023)} dumps from 2023: {self.dumps_2023}")
        super().__init__(max_prompt_len, 'raw_content')
        # self.classifier = DebertaClassifier(foundation_model_path="models/deberta-v3-large-hf-weights",
        #                                     model_path="models/deberta-cc-classifier",
        #                                     device="cuda")

    def get_iter(self):
        seed = int(time.time())
        random.seed(seed)
        logging.info('Using seed {}'.format(seed))
        dataset = CCDataset(
            dumps=self.dumps_2023,
            num_segments=10,
            lang_model=Path("cc_processor/bin/lid.bin"),
            lm_dir=Path("cc_processor/data/lm_sp/"),
            lang_whitelist=['en'],
            lang_threshold=0.5,
            min_len=300,
            cache_dir=None,
            tmp_dir=Path("cc_processor/tmp_segments"),
        )
        return dataset

    def filter_rules_pass(self, prompt, completion):
        if random.random() > 0.1:
            return False
        return True


class HumanDataset(Iterator):
    def __init__(self, max_prompt_len=1500):
        super().__init__()
        self.pile_dataset = PileDataset(max_prompt_len)
        self.red_pajama_dataset = CommonCrawlDataset(max_prompt_len)

    def __next__(self) -> dict:
        res = {}
        if random.random() > 0.5:
            el = next(self.pile_dataset)
            res['data_source'] = 'pile'
        else:
            el = next(self.red_pajama_dataset)
            res['data_source'] = 'common_crawl'

        res['text'] = el['real_completion']
        return res


class PromptDataset(Iterator):
    def __init__(self, max_prompt_len=1500):
        super().__init__()
        self.pile_dataset = PileDataset(max_prompt_len)
        self.red_pajama_dataset = CommonCrawlDataset(max_prompt_len)
        self.max_prompt_len = max_prompt_len

    def __next__(self) -> dict:
        while True:
            res = {}
            if random.random() > 0.5:
                el = next(self.pile_dataset)
                res['data_source'] = 'pile'
            else:
                el = next(self.red_pajama_dataset)
                res['data_source'] = 'common_crawl'

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
