import bittensor as bt
import requests

from langchain_community.llms import Ollama
from detection.validator.text_postprocessing import TextCleaner


class OllamaModel:
    def __init__(self, model_name='llama2'):
        """
        available models you can find on https://github.com/ollama/ollama
        before running model <model_name> install ollama and run 'ollama pull <model_name>'
        """
        bt.logging.info(f'Initializing OllamaModel {model_name}')
        self.model_name = model_name
        self.model = Ollama(model=model_name, timeout=100, num_thread=2)
        self.text_cleaner = TextCleaner()

    def __call__(self, prompt: str) -> str | None:
        try:
            text = self.model.invoke(prompt)
        except requests.exceptions.Timeout as e:
            bt.logging.error("Got timeout exception during calling ollama with model {} and prompt: {}".format(self.model_name, prompt))
            return None
        except Exception as e:
            bt.logging.error("Got unknown exception during calling ollama with model {} and prompt: {}".format(self.model_name, prompt))
            bt.logging.exception(e)
            return None

        return self.text_cleaner.clean_text(text)

    def __repr__(self) -> str:
        return f"{self.model_name}"
