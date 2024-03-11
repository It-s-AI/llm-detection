import bittensor as bt
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
        self.model = Ollama(model=model_name, timeout=100, num_thread=2, num_predict=1500)
        self.text_cleaner = TextCleaner()

    def __call__(self, prompt: str) -> str | None:
        try:
            text = self.model.invoke(prompt)
        except Exception as e:
            bt.logging.info("Couldn't get response from Ollama, probably it's restarting now: {}".format(e))
            return None

        return self.text_cleaner.clean_text(text)

    def __repr__(self) -> str:
        return f"{self.model_name}"
