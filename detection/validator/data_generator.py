import logging
import random
import time
import traceback

import bittensor as bt
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk

from detection.validator.models import ValDataRow
from detection.validator.my_datasets import HumanDataset, PromptDataset
from detection.validator.segmentation_processer import SegmentationProcesser
from detection.validator.text_completion import OllamaModel
from detection.attacks.data_augmentation import DataAugmentator


AI_THEN_HUMAN_PERCENT = 10
HUMAN_THEN_AI_PERCENT = 40
AI_PERCENT = 25
HUMAN_PERCENT = 25

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def get_sentences(text):
    spans = list(sent_tokenizer.span_tokenize(text))

    sentences_with_trailing = []
    for i, (start, end) in enumerate(spans):
        if i < len(spans) - 1:
            next_start = spans[i + 1][0]
        else:
            next_start = len(text)
        expanded_end = end
        while expanded_end < next_start and expanded_end < len(text):
            if text[expanded_end].isspace():
                expanded_end += 1
            else:
                break
        sentence_text = text[start:expanded_end]
        sentences_with_trailing.append(sentence_text)

    return sentences_with_trailing


summary_prompts = [
    "Summarize the text in your own words, highlighting the key points. Do not generate anything else.",
    "Provide a concise summary of the text, focusing on its main argument. Refrain from generating anything else.",
    "In a few sentences, capture the core ideas of the text. Ensure you do not produce anything else.",
    "Write a short overview of the text, emphasizing the primary takeaways. Do not include anything else beyond the summary.",
    "Condense the text into a brief summary, touching on the essential details. Do not provide anything else in your response.",
    "Explain the text’s main points in a summarized format. Nothing else should be generated.",
    "Give me a succinct summary of the text’s content. Do not produce additional information.",
    "What is the most important information to include in a summary of this text? Only produce the summary, nothing else.",
    "Craft a concise review of the text, highlighting the central message. No other content should be added.",
    "Generate a quick summary that identifies the text’s key themes. Provide only the summary, with nothing else included.",
    "Offer a short synopsis of the text, noting the critical arguments. Please do not add anything else.",
    "Provide an executive summary of the text’s main findings. Avoid including extra information.",
    "Distill the text into a paragraph covering the core ideas. Refrain from adding any additional content.",
    "Summarize the text with an emphasis on its conclusion and supporting points. Do not provide anything beyond the summary.",
    "In just a few sentences, outline the text’s primary purpose and insights. Do not generate anything else."
]

generation_prompts = [
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will be given the start and finish of a text plus a summary of its middle. Your job is to compose only the middle portion, making sure it aligns with both the beginning and the end. Do not provide a summary; preserve any existing warnings by rephrasing them, and write nothing else. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You receive the opening and closing paragraphs of a text, as well as a synopsis of the central section. Your task is to generate the text for the middle part alone, ensuring coherence with the given beginning and end. Keep any cautions or alerts by rewording them, and do not include any summarizing. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You are provided with a text’s first and final segments along with a brief outline of what occurs in the middle. Your job is to fill in only the middle content. The final text should flow naturally, so do not insert a summary. Retain all warnings by rephrasing, and write nothing else. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You have the initial and concluding parts of a text, plus a summary that describes the middle portion. Construct only the middle section so that it fits seamlessly from start to end. Rephrase and keep any warnings, and do not add a summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will see the beginning and ending of a text and a concise description of its midpoint. Your role is to write only the middle paragraphs, ensuring coherence with the provided segments. Maintain any disclaimers by rephrasing them, and avoid including any summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You get the start and finish of a text, as well as a summary of what happens in between. Create only the central portion, ensuring logical flow without adding a recap. Retain all cautions by rephrasing them as needed, and do not write anything else. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. The opening and concluding lines of a text, plus a synopsis of the middle, will be given to you. Your aim is to produce only the middle content. Preserve existing warnings in a rephrased form, and refrain from including any summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You have the introduction and conclusion of a text, together with an overview of the midsection. Craft only the missing middle text so that the entire piece remains coherent. Keep any alerts or disclaimers by rewording them, and omit any summarizing. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. Given the first and last parts of a text, plus a short summary of the middle, your job is to write only the central portion. Maintain coherence with the given sections, preserve any warnings by rephrasing them, and do not summarize anything. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will be shown the beginning and ending of a text along with a high-level summary of its midpoint. Only generate the middle content to ensure a continuous flow. Any existing notices or cautions must be included but reworded, and avoid all summarization. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You receive the text’s start and end segments, plus a brief overview of the middle. Construct only the middle text, ensuring it aligns with the summary and merges naturally with the given parts. Rephrase and keep warnings intact, without including a summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You are given the initial and final parts of a text, along with a concise account of what takes place in the middle. Write only the middle paragraphs to form a cohesive piece. Adjust any existing warnings but keep them, and exclude all summaries. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. The beginning and ending sections of a text, along with an abstract of the middle, are provided. Your task: generate only the missing central portion so the entire text reads coherently. Reword any cautions, and do not include any summarizing. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You have the introduction and final segment of a text, plus a summary of the events in between. Craft the middle portion only, preserving flow. Keep warnings by restating them in your own words, and do not add any form of summary. Do not generate anything else (Only middle part) - you're output will be concatenated with begin and end.",
    "You should only provide exact answers. Do not add any reasoning, disclaimers, or filler text. You will see the first and last paragraphs of a text plus a synopsis of the middle. Provide only the middle text, ensuring it fits seamlessly. Retain any disclaimers by rephrasing them, and avoid providing any additional summarization. Do not generate anything else (Only middle part - you're output will be concatenated with begin and end)."
]


def regenerated_in_the_middle(model: OllamaModel, text, summary_prompt, generation_prompt):
    sentences = get_sentences(text)
    lens = [len(x) for x in sentences]
    first_part = len(sentences) // 3
    second_part = 2 * len(sentences) // 3
    third_part = len(sentences)

    first_size = sum(lens[:first_part])
    second_size = sum(lens[first_part:second_part])
    third_size = sum(lens[second_part:])
    for i in range(10):
        if first_size - lens[first_part - 1] > second_size + lens[first_part - 1]:
            first_part -= 1
            first_size = sum(lens[:first_part])
            second_size = sum(lens[first_part:second_part])
        elif second_size - lens[second_part - 1] > third_size + lens[second_part - 1]:
            second_part -= 1
            second_size = sum(lens[first_part:second_part])
            third_size = sum(lens[second_part:])
        elif first_size + lens[first_part] < second_size - lens[first_part]:
            first_part += 1
            first_size = sum(lens[:first_part])
            second_size = sum(lens[first_part:second_part])
        elif second_size + lens[second_part] < third_size - lens[second_part]:
            second_part += 1
            second_size = sum(lens[first_part:second_part])
            third_size = sum(lens[second_part:])
        else:
            break

    begin = ''.join(sentences[:first_part])
    middle = ''.join(sentences[first_part:second_part])
    end = ''.join(sentences[second_part:])

    middle_stripped = middle.rstrip()
    diff = len(middle) - len(middle_stripped)
    end = middle[-diff:] + end
    middle = middle_stripped

    assert model.in_the_middle_generation
    summary = model.classic_invoke([
        {"role": "system", "content": summary_prompt},
        {"role": "user", "content": middle}
    ])
    middle_size = len(middle.split())
    generated_middle = model.classic_invoke([
        {"role": "system", "content": generation_prompt + f" The middle should be about {middle_size} words long"},
        {"role": "user", "content": f"begin: {begin}\nend: {end}\nsummary: {summary}"}
    ])
    labels = [0] * len(begin.split()) + [1] * len(generated_middle.strip().split()) + [0] * len(end.split())
    return begin + generated_middle.strip() + end, labels


class DataGenerator:
    def __init__(self, models: list, min_text_length=250, device=0):
        bt.logging.info(f"DataGenerator initializing...")
        bt.logging.info(f"Models {models}")

        self.min_text_length = min_text_length
        self.models = models
        self.model_names = [el.model_name for el in models]
        self.n_models = len(self.models)
        self.n_models_with_in_the_middle = sum([el.in_the_middle_generation for el in models])
        self.models_with_in_the_middle = [i for i in range(self.n_models) if self.models[i].in_the_middle_generation]
        self.augmentator = DataAugmentator(device)
        self.segmentation_processer = SegmentationProcesser()

        self.human_dataset = HumanDataset()
        self.prompt_dataset = PromptDataset()

        assert len(self.models) != 0

        bt.logging.info(f"DataGenerator initialized")

    def generated_ai_in_the_middle(self, n_samples) -> list[ValDataRow]:
        bt.logging.info(f"Generating {n_samples} samples of AI data")

        res = []
        processed = 0
        generations_per_model = n_samples // self.n_models_with_in_the_middle
        additional_gen = np.random.choice(np.arange(self.n_models_with_in_the_middle), n_samples - generations_per_model * self.n_models_with_in_the_middle, replace=False)
        for i in tqdm(range(self.n_models_with_in_the_middle), desc=f"Generating AI data"):
            cnt_samples = generations_per_model + int(i in additional_gen)
            self.models[self.models_with_in_the_middle[i]].init_model()
            model = self.models[self.models_with_in_the_middle[i]]
            model_name = self.model_names[self.models_with_in_the_middle[i]]

            for j in range(cnt_samples):
                while True:
                    try:
                        el = None
                        while el is None:
                            el = next(self.prompt_dataset)
                            if len(nltk.sent_tokenize(el['prompt'])) < 3 or len(el['prompt'].split()) < self.min_text_length * 0.75:
                                el = None
                                continue
                        summary_idx = np.random.randint(len(summary_prompts))
                        generation_idx = np.random.randint(len(generation_prompts))
                        text, labels = regenerated_in_the_middle(model, el['prompt'], summary_prompts[summary_idx], generation_prompts[generation_idx])
                        el['text'] = text
                        el['segmentation_labels'] = labels
                        el['model_name'] = model_name
                        el['model_params'] = model.params

                        good = False
                        for _ in range(10):
                            text, labels = self.segmentation_processer.subsample_words(text, labels)
                            if len(labels) == 0:
                                continue

                            try:
                                text_auged, augs, labels_auged = self.augmentator(text, labels)
                                assert len(text_auged.split()) == len(labels_auged)
                            except:
                                bt.logging.error("Got error during augmentations for text: {} \n and labels: {}".format(text, labels))
                                logging.info(traceback.format_exc())
                                continue

                            if self.min_text_length <= len(text_auged):
                                el['text_auged'] = text_auged
                                el['augmentations'] = augs
                                el['auged_segmentation_labels'] = labels_auged
                                good = True
                                break

                        if good:
                            break

                    except Exception as e:
                        bt.logging.error(f"Error during generation with {model_name} model: {e}")
                        logging.info(traceback.format_exc())
                        continue

                res.append(ValDataRow(**el, label=True))

            processed += cnt_samples
        return res

    def generate_ai_data(self, n_samples) -> list[ValDataRow]:
        bt.logging.info(f"Generating {n_samples} samples of AI data")

        res = []
        processed = 0
        generations_per_model = n_samples // self.n_models
        additional_gen = np.random.choice(np.arange(self.n_models), n_samples - generations_per_model * self.n_models, replace=False)
        for i in tqdm(range(self.n_models), desc=f"Generating AI data"):
            cnt_samples = generations_per_model + int(i in additional_gen)
            self.models[i].init_model()
            model = self.models[i]
            model_name = self.model_names[i]

            bt.logging.info(f"Generating with {model_name} model and params {model.params}")
            for j in range(cnt_samples):
                while True:
                    el = next(self.prompt_dataset)
                    el['completion'] = model(el['prompt'], text_completion_mode=True)
                    el['model_name'] = model_name
                    el['model_params'] = model.params

                    good = False
                    for _ in range(10):
                        text, cnt_first_human = self.segmentation_processer.merge_prompt_text(el['prompt'], el['completion'])
                        labels = [0] * cnt_first_human + [1] * (len(text.split()) - cnt_first_human)
                        el['text'] = text
                        el['segmentation_labels'] = labels

                        text, labels = self.segmentation_processer.subsample_words(text, labels)
                        if len(labels) == 0:
                            continue

                        try:
                            text_auged, augs, labels_auged = self.augmentator(text, labels)
                            assert len(text_auged.split()) == len(labels_auged)
                        except:
                            bt.logging.error("Got error during augmentations for text: {} \n and labels: {}".format(text, labels))
                            logging.info(traceback.format_exc())
                            continue

                        if self.min_text_length <= len(text_auged):
                            el['text_auged'] = text_auged
                            el['augmentations'] = augs
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
                    el['segmentation_labels'] = cnt_first_human * [0]
                    labels = el['segmentation_labels']

                    text, labels = self.segmentation_processer.subsample_words(text, labels)
                    if len(labels) == 0:
                        continue

                    text_auged, augs, labels_auged = self.augmentator(text, labels)

                    if self.min_text_length <= len(text_auged):
                        el['text_auged'] = text_auged
                        el['augmentations'] = augs
                        el['auged_segmentation_labels'] = labels_auged
                        good = True
                        break

                if good:
                    break

            res.append(ValDataRow(**el, label=False))
        return res

    def generate_data(self, n_human_samples, n_ai_samples) -> list[ValDataRow]:
        # ai probabilities are: 10/75 for ai then human, 40/75 for human then ai, 25/75 for full ai
        ai_in_the_middle_mask = np.random.random(n_ai_samples) < 10/75
        ai_in_the_middle_count = np.sum(ai_in_the_middle_mask)
        res = self.generated_ai_in_the_middle(ai_in_the_middle_count) + self.generate_human_data(n_human_samples) + self.generate_ai_data(n_ai_samples - ai_in_the_middle_count)
        random.shuffle(res)
        return res


@click.command()
@click.option("--input_path", default=None)
@click.option("--output_path", default='generated_data.csv')
@click.option("--n_samples", default=None)
@click.option("--n_ai_samples", default=75)
@click.option("--n_human_samples", default=25)
@click.option("--ollama_url", default="http://127.0.0.1:11434")
def main(input_path, output_path, n_samples, n_ai_samples, n_human_samples, ollama_url):
    text_models = [OllamaModel(model_name='llama2:13b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='llama3:text', base_url=ollama_url),
                   OllamaModel(model_name='llama3:70b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='llama3.1:70b-text-q4_0', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='llama3.2', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='llama3.3:70b', base_url=ollama_url),

                   OllamaModel(model_name='qwen:32b-text-v1.5-q4_0', base_url=ollama_url),
                   OllamaModel(model_name='qwen2:72b-text-q4_0', base_url=ollama_url),
                   OllamaModel(model_name='qwen2.5:14b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='qwen2.5-coder:32b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='qwen2.5:72b', base_url=ollama_url, in_the_middle_generation=True),

                   OllamaModel(model_name='command-r', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='command-r', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='command-r-plus:104b', base_url=ollama_url, in_the_middle_generation=True),

                   OllamaModel(model_name='gemma2:9b-instruct-q4_0', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='gemma2:27b-text-q4_0', base_url=ollama_url),
                   OllamaModel(model_name='gemma2:27b', base_url=ollama_url, in_the_middle_generation=True),

                   OllamaModel(model_name='mistral:text', base_url=ollama_url),
                   OllamaModel(model_name='mistral-nemo:12b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='mistral-small:22b', base_url=ollama_url, in_the_middle_generation=True),
                   OllamaModel(model_name='mistral-large:123b', base_url=ollama_url, in_the_middle_generation=True),

                   OllamaModel(model_name='internlm2:7b', base_url=ollama_url),
                   OllamaModel(model_name='internlm2:20b', base_url=ollama_url),
                   OllamaModel(model_name='internlm/internlm2.5:20b-chat', base_url=ollama_url),

                   OllamaModel(model_name='deepseek-v2:16b', base_url=ollama_url),
                   OllamaModel(model_name='deepseek-r1:14b', base_url=ollama_url),
                   OllamaModel(model_name='phi4:14b', base_url=ollama_url),
                   OllamaModel(model_name='aya-expanse:32b', base_url=ollama_url),
                   OllamaModel(model_name='yi:34b-chat', base_url=ollama_url),
                   OllamaModel(model_name='athene-v2:72b', base_url=ollama_url),
                   ]

    generator = DataGenerator(text_models)

    if input_path is not None:
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
                cur_path = output_path[:output_path.rfind('.')] + '_{}'.format(start_ind) + output_path[output_path.rfind('.'):]
                df[start_ind:].to_csv(cur_path)
                bt.logging.info("Saved {} samples into {}".format(len(df[start_ind:]), cur_path))
            except:
                bt.logging.error("Coudnt save data into file: {}".format(traceback.format_exc()))

        epoch += 1
        time.sleep(1)


if __name__ == '__main__':
    main()

# nohup python3 detection/validator/data_generator.py --output_path "data/generated_data_v3.5.csv" > generator.log &
