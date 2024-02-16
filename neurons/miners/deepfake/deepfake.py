import torch
import os
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from utils import preprocess, detect


class DeepfakeTextDetect:
    def __init__(self, device='cpu'): # use 'cuda:0' if GPU is available
        model_dir = "yaful/DeepfakeTextDetection"  # model in the online demo
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    def __call__(self, text):
        text = preprocess(text)
        result = detect(text, self.tokenizer, self.model, self.device)
        return float(result == 'machine-generated')


if __name__ == '__main__':
    text = 'I hope this message finds you well. My colleague, Sergey, and I came across your offer on Twitter regarding the "incubate/bootstrap" opportunity, and we are excited to share our idea that aligns perfectly with this initiative.'
    model = DeepfakeTextDetect()
    print(model(text))
