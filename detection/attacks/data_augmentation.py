import nltk
import numpy as np

from detection.attacks.delete import DeleteAttack
from detection.attacks.spelling import SpellingAttack
from detection.attacks.synonym import SynonymAttack
from detection.attacks.zero_width_space import ZeroWidthSpaceAttack

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


class DataAugmentator:
    def __init__(self, device):
        self.attacks = [{'attacker': SynonymAttack(device=device), 'p': 0.05, 'pass_labels': True},
                        {'attacker': ZeroWidthSpaceAttack(), 'p': 0.05},
                        {'attacker': SpellingAttack(), 'p': 0.4},
                        {'attacker': DeleteAttack(), 'p': 0.1},
                        ]

        # {'attacker': ParaphraseAttack(), 'p': 0.2, 'apply_label': 1}, - needs too much GPU

    def __call__(self, text, labels):
        text = text.strip()

        applied_augs = []
        for augmentation_step in self.attacks:
            if np.random.random() > augmentation_step['p']:
                continue

            if augmentation_step.get('pass_labels'):
                text = augmentation_step['attacker'].attack(text, labels)
            else:
                text = augmentation_step['attacker'].attack(text)
            applied_augs.append(type(augmentation_step['attacker']).__name__)

        n_auged = len(text.split())

        if not sum(labels):
            labels_auged = [0] * n_auged
        else:
            first_zeros = 0
            for i in range(len(labels)):
                if labels[i] == 0:
                    first_zeros += 1
                else:
                    break
            last_zeros = 0
            for i in range(len(labels) - 1, -1, -1):
                if labels[i] == 0:
                    last_zeros += 1
                else:
                    break
            new_first_zeros = int(n_auged * first_zeros / len(labels))
            new_last_zeros = int(n_auged * last_zeros / len(labels))
            new_middle_ones = n_auged - new_first_zeros - new_last_zeros
            labels_auged = [0] * new_first_zeros + [1] * new_middle_ones + [0] * new_last_zeros

        return text, applied_augs, labels_auged
