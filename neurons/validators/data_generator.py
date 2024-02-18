import logging
import os

import openai
import pandas as pd
from tqdm import tqdm

from neurons.models import ValDataRow
from neurons.validators.my_datasets import HumanDataset, PromptDataset
from neurons.validators.text_completion import OllamaModel, OpenAiModel


class DataGenerator:
    def __init__(self, models: list, model_probs: list | None):
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

        logging.info('Init completed')

    def generate_human_data(self, n_samples) -> list[ValDataRow]:
        logging.info('Generating {} samples of Human data'.format(n_samples))
        res = []
        for i in range(n_samples):
            res.append(ValDataRow(**next(self.human_dataset), label=False))
        return res

    def generate_ai_data(self, n_samples) -> list[ValDataRow]:
        logging.info('Generating {} samples of AI data'.format(n_samples))
        res = []
        processed = 0
        for i in range(len(self.models)):
            cnt_samples = int(n_samples * self.model_probs[i]) if i != len(self.models) - 1 else n_samples - processed
            model = self.models[i]
            model_name = self.model_names[i]
            logging.info('Generating {} samples with {} model'.format(n_samples, model_name))

            for j in tqdm(range(cnt_samples)):
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
    openai.api_key = os.getenv("OPENAI_API_KEY")
    models = [OpenAiModel(model_name='gpt-3.5-turbo'), OpenAiModel(model_name='gpt-4-turbo-preview'),
              OllamaModel(model_name='vicuna'), OllamaModel(model_name='mistral')]
    generator = DataGenerator(models, [0.25, 0.25, 0.25, 0.25])

    data = generator.generate_data(n_samples=10)
    data = pd.DataFrame([el.dict() for el in data])
    data.to_csv('val_data.csv', index=False)
