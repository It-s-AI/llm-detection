import random
import time

import bittensor as bt
import pandas as pd
from tqdm import tqdm

from detection.validator.models import ValDataRow
from detection.validator.my_datasets import HumanDataset, PromptDataset
from detection.validator.text_completion import OllamaModel


class DataGenerator:
    def __init__(self, models: list, model_probs: list | None):
        bt.logging.info(f"DataGenerator initializing...")
        bt.logging.info(f"Models {models}")
        bt.logging.info(f"model_probs {model_probs}")

        self.models = models
        self.model_names = [el.model_name for el in models]

        if model_probs is None:
            self.model_probs = [1 / len(self.models) for i in range(len(self.models))]
        else:
            self.model_probs = model_probs
            assert sum(model_probs) == 1

        self.human_dataset = HumanDataset()
        self.prompt_dataset = PromptDataset()

        assert len(self.models) == len(self.model_names) == len(self.model_probs)
        assert len(self.models) != 0

        bt.logging.info(f"DataGenerator initialized")

    def generate_ai_data(self, n_samples) -> list[ValDataRow]:
        bt.logging.info(f"Generating {n_samples} samples of AI data")

        res = []
        processed = 0
        for i in tqdm(range(len(self.models)), desc=f"Generating AI data"):
            cnt_samples = int(n_samples * self.model_probs[i]) if i != len(self.models) - 1 else n_samples - processed
            model = self.models[i]
            model_name = self.model_names[i]

            for j in tqdm(range(cnt_samples), desc=f"Generating with {model_name} model"):
                while True:
                    el = next(self.prompt_dataset)
                    el['text'] = model(el['prompt'])
                    el['model_name'] = model_name
                    if el['text'] is not None:
                        break
                    time.sleep(1)

                res.append(ValDataRow(**el, label=True))

            processed += cnt_samples
        return res

    def generate_human_data(self, n_samples) -> list[ValDataRow]:
        bt.logging.info(f"Generating {n_samples} samples of Human data")

        res = []
        for i in tqdm(range(n_samples), desc="Generating Humand Data"):
            res.append(ValDataRow(**next(self.human_dataset), label=False))
        return res

    def generate_data(self, n_human_samples, n_ai_samples) -> list[ValDataRow]:
        res = self.generate_human_data(n_human_samples) + self.generate_ai_data(n_ai_samples)
        random.shuffle(res)
        return res


if __name__ == '__main__':
    models = [OllamaModel(model_name='vicuna'), OllamaModel(model_name='mistral')]
    generator = DataGenerator(models, [0.5, 0.5])

    epoch = 0
    full_data = []
    while True:
        start_time = time.time()
        data = generator.generate_data(n_ai_samples=25, n_human_samples=25)
        # full_data += [el.dict() for el in data]
        bt.logging.info('Generated epoch {} in {} seconds'.format(epoch, round(time.time() - start_time, 3)))

        # if epoch % 5 == 0:
        #     df = pd.DataFrame(full_data)
        #     df.to_csv("generated_data.csv", index=False)
        #     bt.logging.info("Saved {} samples into generated_data.csv".format(len(full_data)))

        epoch += 1
        time.sleep(1)
