import logging
import random

import pandas as pd
# import bittensor as bt
from datasets import load_dataset
from collections.abc import Iterator
from pylatexenc.latex2text import LatexNodes2Text
from tqdm import tqdm


class HumanDataset(Iterator):
    def __init__(self):
        super().__init__()
        seed = random.randint(0, 1000)
        # self.openwebtext = iter(
        #     load_dataset("openwebtext", split="train", streaming=True).shuffle(
        #         seed=seed, buffer_size=1000
        #     )
        # )

        self.c4 = iter(
            load_dataset("allenai/c4", 'en',  streaming=True)['train'].shuffle(
                seed=seed, buffer_size=1000
            )
        )

    def __next__(self) -> dict:
        while True:
            el = next(self.c4)
            res = {'text': el['text'], 'data_source': 'c4_en'}

            # cnt_words = random.randint(25, 500)
            # if len(res['text'].split()) < cnt_words:
            #     logging.info('skipping, due to small amout of words')
            #     continue

            # res['text'] = ' '.join(res['text'].split()[:cnt_words])
            # res['text'] = res['text'][:res['text'].rfind('.') + 1]
            # res['text'] = LatexNodes2Text().latex_to_text(res['text'])
            return res


class PromptDataset(Iterator):
    def __init__(self):
        super().__init__()
        seed = random.randint(0, 1000)
        self.hc3 = iter(
            load_dataset("Hello-SimpleAI/HC3", name="all", streaming=True)['train'].shuffle(
                seed=seed, buffer_size=1000
            )
        )

    def __next__(self) -> dict:
        while True:
            # bt.logging.debug("Retrieving data from PromptDataset...")
            el = next(self.hc3)
            if random.random() < 0.5:
                while el['source'] == 'reddit_eli5':
                    el = next(self.hc3)
            else:
                while el['source'] != 'reddit_eli5':
                    el = next(self.hc3)

            res = {'prompt': el["question"], 'data_source': el['source']}

            # Check if the text is not empty or does not consist only of newline characters
            if res['prompt'].strip():
                return res


if __name__ == '__main__':
    # dataset = PromptDataset()
    # print(next(dataset))

    dataset = HumanDataset()
    data = []
    for i in tqdm(range(200)):
        data.append(next(dataset))
    print(data[:5])
    df = pd.DataFrame(data)
    df['label'] = 0
    df.to_csv('../../../llm-detection-sergak0/notebooks/val_data_human_allenai_c4_1.csv', index=False)
