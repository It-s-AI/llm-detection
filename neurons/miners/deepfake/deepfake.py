import torch
import os
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from miners.deepfake.utils import preprocess, detect
import bittensor as bt


class DeepfakeTextDetect:
    def __init__(self, config): # use 'cuda:0' if GPU is available
        print("CONFIG ", config)
        model_dir = "yaful/DeepfakeTextDetection"  # model in the online demo
        self.config = config
        self.device = self.config.neuron.device

        bt.logging.info(f"Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        bt.logging.info(f"DeepfakeTextDetect Model Loaded on {self.device}!")


    def __call__(self, text):
        bt.logging.info(f"Text to process: {text}")
        text = preprocess(text)
        bt.logging.info(f"Procced: {text}")
        result = detect(text, self.tokenizer, self.model, self.device)
        bt.logging.info(f"Result: {result}")

        return [float(result == 'machine-generated')]


if __name__ == '__main__':
    text = 'I hope this message finds you well. My colleague, Sergey, and I came across your offer on Twitter regarding the "incubate/bootstrap" opportunity, and we are excited to share our idea that aligns perfectly with this initiative.'
    # model = DeepfakeTextDetect()
    # print(model(text))