from deberta_classifier import DebertaClassifier
from gpt_zero import PPLModel


class EnsembleModel:
    def __init__(self, ppl_model: PPLModel, deberta_model: DebertaClassifier):
        self.ppl_model = ppl_model
        self.deberta_model = deberta_model

    def __call__(self, text):
        if self.ppl_model(text) < 0.01:
            return 0
        else:
            return self.deberta_model(text)

    def predict_batch(self, texts):
        ppl_pred = self.ppl_model.predict_batch(texts)
        deberta_pred = self.deberta_model.predict_batch(texts)
        res = []
        for i in range(len(texts)):
            if ppl_pred[i] < 0.01:
                res.append(0)
            else:
                res.append(deberta_pred[i])
        return res

