import logging
import time

import bittensor as bt
import numpy as np
from langchain_community.llms import Ollama

from detection.validator.text_postprocessing import TextCleaner
import ollama


class OllamaModel:
    def __init__(self, model_name, num_predict=1000):
        """
        available models you can find on https://github.com/ollama/ollama
        before running model <model_name> install ollama and run 'ollama pull <model_name>'
        """
        bt.logging.info(f'Initializing OllamaModel {model_name}')
        if num_predict > 1000:
            raise Exception("You're trying to set num_predict to more than 1000, it can lead to context overloading and Ollama hanging")

        pulled_models = [el['name'] for el in ollama.list()['models']] if ollama.list() is not None else []
        if model_name not in pulled_models and model_name + ':latest' not in pulled_models:
            bt.logging.info("Model {} cannot be found locally - downloading it...".format(model_name))
            ollama.pull(model_name)
            bt.logging.info("Successfully downloaded {}".format(model_name))

        self.model_name = model_name
        self.num_predict = num_predict
        self.model = None
        self.params = {}
        self.init_model()

        self.text_cleaner = TextCleaner()

    def init_model(self):
        sampling_temperature = np.clip(np.random.normal(loc=1, scale=0.2), a_min=0, a_max=2)
        # Centered around 1 because that's what's hardest for downstream classification models.
        frequency_penalty = np.random.uniform(low=0.9, high=1.5)
        top_k = int(np.random.choice([-1, 20, 40]))
        top_k = top_k if top_k != -1 else None
        top_p = np.random.uniform(low=0.5, high=1)

        self.model = Ollama(model=self.model_name,
                            timeout=200,
                            num_thread=1,
                            num_predict=self.num_predict,
                            temperature=sampling_temperature,
                            repeat_penalty=frequency_penalty,
                            top_p=top_p,
                            top_k=top_k)
        self.params = {'top_k': top_k, 'top_p': top_p, 'temperature': sampling_temperature, 'repeat_penalty': frequency_penalty}

    def __call__(self, prompt: str, text_completion_mode=False) -> str | None:
        while True:
            try:
                if text_completion_mode:
                    if ':text' not in self.model_name:
                        system_message = "You're a text completion model, just complete text that user sended you" #. Return text without any supportive - we write add your result right after the user text
                        text = self.model.invoke([{'role': 'system', 'content': system_message},
                                                  {'role': 'user', 'content': prompt}])
                    else:
                        text = self.model.invoke(prompt)
                else:
                    assert ':text' not in self.model_name
                    text = self.model.invoke(prompt)

                return self.text_cleaner.clean_text(text)
            except Exception as e:
                bt.logging.info("Couldn't get response from Ollama, probably it's restarting now: {}".format(e))
                time.sleep(1)

    def __repr__(self) -> str:
        return f"{self.model_name}"


if __name__ == '__main__':
    bt.logging.info("started")
    model = OllamaModel('llama2')
    bt.logging.info("finished")
    print(model.model)
