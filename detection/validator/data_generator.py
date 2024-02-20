import bittensor as bt
import os

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

    def generate_human_data(self, n_samples) -> list[ValDataRow]:
        bt.logging.info(f"Generating {n_samples} samples of Human data")

        res = []
        for i in tqdm(range(n_samples), desc="Generating Humand Data"):
            res.append(ValDataRow(**next(self.human_dataset), label=False))
        return res

    def generate_ai_data(self, n_samples) -> list[ValDataRow]:
        bt.logging.info(f"Generating {n_samples} samples of AI data")

        res = []
        processed = 0
        for i in tqdm(range(len(self.models)), desc=f"Generating AI data"):
            cnt_samples = int(n_samples * self.model_probs[i]) if i != len(self.models) - 1 else n_samples - processed
            model = self.models[i]
            model_name = self.model_names[i]

            for j in tqdm(range(cnt_samples), desc=f"Generating with {model_name} model"):
                el = next(self.prompt_dataset)

                el['text'] = model(el['prompt'])
                el['model_name'] = model_name
                res.append(ValDataRow(**el, label=True))

            processed += cnt_samples
        return res

    def generate_data(self, n_samples) -> list[ValDataRow]:
        human_samples = n_samples // 2
        ai_samples = n_samples - human_samples
        return self.generate_human_data(human_samples) + self.generate_ai_data(ai_samples)


if __name__ == '__main__':
    models = [OllamaModel(model_name='vicuna'), OllamaModel(model_name='mistral')]
    generator = DataGenerator(models, [0.5, 0.5])

    data = generator.generate_data(n_samples=10)
    data = pd.DataFrame([el.dict() for el in data])
    print(data)
    # data.to_csv('val_data.csv', index=False)