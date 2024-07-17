import logging
import random
import time
import traceback

import bittensor as bt
import click
import pandas as pd
from tqdm import tqdm

from detection.validator.models import ValDataRow
from detection.validator.my_datasets import HumanDataset, PromptDataset
from detection.validator.segmentation_processer import SegmentationProcesser
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
        self.segmentation_processer = SegmentationProcesser()

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

                    good = False
                    for _ in range(10):
                        text, cnt_first_human = self.segmentation_processer.merge_prompt_text(el['prompt'], el['text'])
                        text, labels = self.segmentation_processer.subsample_words(text, cnt_first_human)
                        try:
                            text_auged, augs, labels_auged = self.augmentator(text, labels)
                        except:
                            logging.error("Got error during augmentations for text: {} \n and labels: {}".format(text, labels))
                            logging.info(traceback.format_exc())
                            continue

                        if self.min_text_length <= len(text_auged):
                            el['text_auged'] = text_auged
                            el['augmentations'] = augs
                            el['segmentation_labels'] = labels
                            el['auged_segmentation_labels'] = labels_auged
                            good = True
                            break

                    if good:
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

                good = False
                for _ in range(10):
                    text, cnt_first_human = el['text'], len(el['text'].split())
                    text, labels = self.segmentation_processer.subsample_words(text, cnt_first_human)
                    text_auged, augs, labels_auged = self.augmentator(text, labels)

                    if self.min_text_length <= len(text_auged):
                        el['text_auged'] = text
                        el['augmentations'] = augs
                        el['segmentation_labels'] = labels
                        el['auged_segmentation_labels'] = labels_auged
                        good = True
                        break

                if good:
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
@click.option("--n_ai_samples", default=100)
@click.option("--n_human_samples", default=0)
def main(input_path, output_path, n_samples, n_ai_samples, n_human_samples):
    text_models = [OllamaModel(model_name='mistral:text'),
                   OllamaModel(model_name='llama3:text'),
                   OllamaModel(model_name='mixtral:text'),

                   OllamaModel(model_name='gemma:7b'),
                   OllamaModel(model_name='command-r'),
                   OllamaModel(model_name='neural-chat'),
                   OllamaModel(model_name='zephyr:7b-beta'),
                   OllamaModel(model_name='openhermes'),
                   OllamaModel(model_name='wizardcoder'),
                   OllamaModel(model_name='starling-lm:7b-beta'),
                   OllamaModel(model_name='yi:34b'),
                   OllamaModel(model_name='openchat:7b'),
                   OllamaModel(model_name='dolphin-mistral'),
                   OllamaModel(model_name='solar'),
                   OllamaModel(model_name='llama2:13b'),]

    generator = DataGenerator(text_models, None)

    if input_path is not None:
        # path_to_prompts = 'prompts.csv'
        data = pd.read_csv(input_path)
        generator.prompt_dataset = iter(data.to_dict('records'))
        n_samples = len(data)

    epoch = 0
    full_data = []
    while True:
        start_time = time.time()
        if n_samples is not None and len(full_data) >= n_samples:
            bt.logging.info("Successfully generated {} samples, finishing".format(n_samples))
            break

        data = generator.generate_data(n_ai_samples=n_ai_samples, n_human_samples=n_human_samples)
        full_data += [el.dict() for el in data]
        bt.logging.info('Generated epoch {} in {} seconds'.format(epoch, round(time.time() - start_time, 3)))

        if epoch % 1 == 0 or (n_samples is not None and len(full_data) >= n_samples):
            df = pd.DataFrame(full_data)
            try:
                start_ind = len(full_data) // 10000 * 10000
                cur_path = output_path[:-4] + '_{}'.format(start_ind) + '.csv'
                df[start_ind:].to_csv(cur_path, index=False)
                bt.logging.info("Saved {} samples into {}".format(len(df[start_ind:]), cur_path))
            except:
                bt.logging.error("Coudnt save data into file")

        epoch += 1
        time.sleep(1)


if __name__ == '__main__':
    main()

# nohup python3 detection/validator/data_generator.py --n_ai_samples=75 --n_human_samples=25 --output_path "data/generated_data.csv" > generator.log &
