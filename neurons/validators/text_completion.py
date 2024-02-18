import logging
import time

import openai
from langchain_community.llms import Ollama
from text_postprocessing import TextCleaner


class OpenAiModel:
    def __init__(self, model_name='gpt-3.5-turbo', temperature=0.7, max_tokens=500, presence_penalty=0.1, frequency_penalty=0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    def __call__(self, prompt: str) -> str:
        messages = [
            {"role": 'user', "content": prompt}
        ]

        time.sleep(1)
        resp = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )["choices"][0]["message"]["content"]

        return resp


class OllamaModel:
    def __init__(self, model_name='llama2'):
        """
        available models you can find on https://github.com/ollama/ollama
        before running model <model_name> install ollama and run 'ollama pull <model_name>'
        """
        logging.info('Initializing OllamaModel {}'.format(model_name))
        self.model_name = model_name
        self.model = Ollama(model=model_name)
        self.text_cleaner = TextCleaner()

    def __call__(self, prompt: str) -> str:
        text = self.model.invoke(prompt)
        return self.text_cleaner.clean_text(text)

