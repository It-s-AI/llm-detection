import random

import nltk
import numpy as np

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


class SpellingAttack:
    def __init__(self, max_cycles=5):

        self.char_changes = [
            {'name': 'typo_char_swap', 'p': 0.1},
            {'name': 'typo_missing_char', 'p': 0.1},
            {'name': 'typo_extra_char', 'p': 0.1},
            # {'name': 'typo_nearby_char', 'p': 0.1},
            # {'name': 'typo_similar_char', 'p': 0.1},
            {'name': 'typo_skipped_space', 'p': 0.1},
            {'name': 'typo_random_space', 'p': 0.1},
            # {'name': 'typo_repeated_char', 'p': 0.1},
            {'name': 'typo_unichar', 'p': 0.1},
            {'name': 'decapitalize_char', 'p': 0.1},
            {'name': 'capitalize_char', 'p': 0.1},
        ]

        self.max_cycles = max_cycles

    def decapitalize_char(self, text):
        capital_indices = [i for i, char in enumerate(text) if char.isupper()]
        if len(capital_indices) == 0:
            return text

        random_index = np.random.choice(capital_indices)

        modified_text = text[:random_index] + text[random_index].lower() + text[random_index + 1:]
        return modified_text

    def capitalize_char(self, text):
        lower_indices = [i for i, char in enumerate(text) if char.islower()]
        if len(lower_indices) == 0:
            return text

        random_index = np.random.choice(lower_indices)
        modified_text = text[:random_index] + text[random_index].upper() + text[random_index + 1:]
        return modified_text

    def attack(self, text):
        augs = []
        n_repeated = random.randint(1, self.max_cycles)
        for i in range(n_repeated):
            augs += self.char_changes
        np.random.shuffle(augs)

        for augmentation_step in augs:
            if np.random.random() > augmentation_step['p']:
                continue

            if augmentation_step['name'] == 'decapitalize_char':
                text = self.decapitalize_char(text)
            elif augmentation_step['name'] == 'capitalize_char':
                text = self.capitalize_char(text)
            elif 'typo_' in augmentation_step['name']:
                error_type_name = augmentation_step['name'][5:]
                try:
                    text = eval(f'typo.StrErrer(text).{error_type_name}().result')
                except:
                    pass
            else:
                raise Exception("Unexpected augmentation name: {}".format(augmentation_step['name']))

        return text
