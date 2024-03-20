import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pickle
import numpy as np


class PPLModel:
    def __init__(self, device="cuda", model_id="microsoft/phi-2"):
        self.device = device
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        self.max_length = 512 #self.model.config.n_positions
        self.stride = 512
        self.logreg = LogisticRegression(class_weight='balanced')

    def __call__(self, text):
        ppl = self.getPPL(text)
        if ppl is None:
            print('None ppl')
            return 0

        features = [(100 - ppl) / 100]
        return self.logreg.predict_proba([features])[0][1]

    def fit(self, X, y):
        features = []
        mask = []
        for text in tqdm(X):
            ppl = self.getPPL(text)
            ppl = (100 - ppl) / 100 if ppl is not None else None
            features.append(ppl)
            mask.append(ppl is not None)

        features = np.array(features)
        mask = np.array(mask)
        print("Number of not-none ppl: {}".format(mask.sum()))

        features = features[mask]
        y = y[mask]
        self.logreg.fit(features.reshape(-1, 1), y)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.logreg, f)

    def load_pretrained(self, path):
        with open(path, 'rb') as f:
            self.logreg = pickle.load(f)

    def getPPL(self, text):
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        likelihoods = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                loss = self.model(input_ids, labels=target_ids).loss
                neg_log_likelihood = loss * trg_len
                likelihoods.append(neg_log_likelihood)

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        if torch.isnan(torch.Tensor(nlls)).any() or len(nlls) == 0:
            return None

        ppl = int(torch.exp(torch.stack(nlls).sum() / end_loc))
        return ppl


if __name__ == '__main__':
    model = PPLModel(device='cpu')
    model.load_pretrained('neurons/miners/ppl_model.pk')
    text = 'Hello world, i am here'
    res = model(text)
    print(res)
