import nltk
import numpy as np

from detection.attacks.delete import DeleteAttack
from detection.attacks.spelling import SpellingAttack
from detection.attacks.synonym import SynonymAttack

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class DataAugmentator:
    def __init__(self,):
        self.attacks = [{'attacker': SpellingAttack(), 'p': 0.5},
                        {'attacker': SynonymAttack(), 'p': 0.2},
                        {'attacker': DeleteAttack(), 'p': 0.2}]

    def __call__(self, text, labels):
        text = text.strip().copy()

        applied_augs = []
        for augmentation_step in self.attacks:
            if np.random.random() > augmentation_step['p']:
                continue
            text = augmentation_step['attacker'].attack(text)
            applied_augs.append(type(augmentation_step['attacker']).__name__)

        n_auged = len(text.split())
        new_cnt_first_human = int(n_auged * (1 - sum(labels) / len(labels)))
        labels_auged = [0] * new_cnt_first_human + [1] * (n_auged - new_cnt_first_human)
        return text, applied_augs, labels_auged
