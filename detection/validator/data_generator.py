import random
import time

import bittensor as bt
import click
import pandas as pd
from tqdm import tqdm

from detection.validator.models import ValDataRow
from detection.validator.my_datasets import HumanDataset, PromptDataset
from detection.validator.text_completion import OllamaModel
from detection.validator.data_augmentation import DataAugmentator


class DataGenerator:
    def __init__(self, models: list, model_probs: list | None, min_text_length=250):
        bt.logging.info(f"DataGenerator initializing...")
        bt.logging.info(f"Models {models}")
        bt.logging.info(f"model_probs {model_probs}")

        self.min_text_length = min_text_length
        self.models = models
        self.model_names = [el.model_name for el in models]
        self.augmentator = DataAugmentator()

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
            self.models[i].init_model()
            model = self.models[i]
            model_name = self.model_names[i]

            bt.logging.info(f"Generating with {model_name} model and params {model.params}")
            for j in range(cnt_samples):
                while True:
                    el = next(self.prompt_dataset)
                    el['text'] = model(el['prompt'], text_completion_mode=True)
                    el['model_name'] = model_name
                    el['model_params'] = model.params

                    text, augs = self.augmentator(el['text'])
                    el['text'] = text
                    el['augmentations'] = augs

                    if len(el['text']) > self.min_text_length:
                        break

                res.append(ValDataRow(**el, label=True))

            processed += cnt_samples
        return res

    def generate_human_data(self, n_samples) -> list[ValDataRow]:
        bt.logging.info(f"Generating {n_samples} samples of Human data")

        res = []
        for i in tqdm(range(n_samples), desc="Generating Humand Data"):
            while True:
                el = next(self.human_dataset)

                text, augs = self.augmentator(el['text'])
                el['text'] = text
                el['augmentations'] = augs

                if len(el['text']) > self.min_text_length:
                    break

            res.append(ValDataRow(**el, label=False))
        return res

    def generate_data(self, n_human_samples, n_ai_samples) -> list[ValDataRow]:
        res = self.generate_human_data(n_human_samples) + self.generate_ai_data(n_ai_samples)
        random.shuffle(res)
        return res


@click.command()
@click.option("--input_path", default=None)
@click.option("--output_path", default='generated_data.csv')
@click.option("--n_samples", default=None)
@click.option("--ai_batch_size", default=100)
@click.option("--human_batch_size", default=0)
def main(input_path, output_path, n_samples, ai_batch_size, human_batch_size):
    # models = [OllamaModel(model_name='neural-chat'),
    #           OllamaModel(model_name='vicuna'),
    #           OllamaModel(model_name='gemma:7b'),
    #           OllamaModel(model_name='mistral'),
    #           OllamaModel(model_name='zephyr:7b-beta'),
    #
    #           OllamaModel(model_name='llama3'),
    #           # OllamaModel(model_name='command-r'),
    #           OllamaModel(model_name='wizardlm2'),
    #           OllamaModel(model_name='openhermes'),
    #           # OllamaModel(model_name='mixtral'),
    #           OllamaModel(model_name='starling-lm'),
    #           OllamaModel(model_name='openchat'),
    #           # OllamaModel(model_name='nous-hermes2'),
    #           OllamaModel(model_name='wizardcoder'), ]

    text_models = [OllamaModel(model_name='mistral:text'),
                   OllamaModel(model_name='llama3:text'),
                   OllamaModel(model_name='mixtral:text'),
                   OllamaModel(model_name='gemma:text'),

                   OllamaModel(model_name='command-r'),
                   OllamaModel(model_name='neural-chat'),
                   OllamaModel(model_name='zephyr:7b-beta'),
                   OllamaModel(model_name='openhermes'),
                   OllamaModel(model_name='wizardcoder'),
                   OllamaModel(model_name='starling-lm'),
                   ]

    generator = DataGenerator(text_models, None)

    if input_path is not None:
        # path_to_prompts = 'prompts.csv'
        data = pd.read_csv(input_path)
        generator.prompt_dataset = iter(data.to_dict('records'))
        n_samples = len(data)

    if n_samples is not None:
        assert human_batch_size == 0, "You cant set n_samples and human_batch_size at the same time"

    epoch = 0
    full_data = []
    while True:
        start_time = time.time()
        if len(full_data) == n_samples:
            bt.logging.info("Successfully generated {} samples, finishing".format(n_samples))
            break

        cur_ai_batch_size = ai_batch_size if n_samples is None else min(ai_batch_size, n_samples - len(full_data))
        data = generator.generate_data(n_ai_samples=cur_ai_batch_size, n_human_samples=human_batch_size)
        full_data += [el.dict() for el in data]
        bt.logging.info('Generated epoch {} in {} seconds'.format(epoch, round(time.time() - start_time, 3)))

        if epoch % 1 == 0 or len(full_data) == n_samples:
            df = pd.DataFrame(full_data)
            try:
                df.to_csv(output_path, index=False)
                bt.logging.info("Saved {} samples into {}".format(len(full_data), output_path))
            except:
                bt.logging.error("Coudnt save data into file")

        epoch += 1
        time.sleep(1)


if __name__ == '__main__':
    main()

# nohup python3 detection/validator/data_generator.py --ai_batch_size=150 --human_batch_size=150 > generator.log &
