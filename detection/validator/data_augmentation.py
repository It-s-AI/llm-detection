import random

import numpy as np
from nltk import pos_tag

import random
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk import pos_tag
from nltk.tokenize import sent_tokenize

from spellchecker import SpellChecker
# import jamspell
from symspellpy import Verbosity, SymSpell
import re
import typo

FIXED_ORDER_DATA_AUGMENTATION_STEPS = [
    # {'name': 'SubsampleSentences', 'p': 1},
    # {'name' : 'BuggySpellCheck', 'p': 0.2},
]

CHAR_CHANGES = [
    {'name': 'typo_char_swap', 'p': 0.1},
    {'name': 'typo_missing_char', 'p': 0.1},
    {'name': 'typo_extra_char', 'p': 0.1},
    {'name': 'typo_nearby_char', 'p': 0.1},
    {'name': 'typo_similar_char', 'p': 0.1},
    {'name': 'typo_skipped_space', 'p': 0.1},
    {'name': 'typo_random_space', 'p': 0.1},
    {'name': 'typo_repeated_char', 'p': 0.1},
    {'name': 'typo_unichar', 'p': 0.1},
]

RAND_ORDER_DATA_AUGMENTATION_STEPS = [
    {'name': 'DecapitalizeRandomLetter', 'p': 0.1},
    {'name': 'DecapitalizeRandomLetter', 'p': 0.1},
    {'name': 'CapitalizeRandomLetter', 'p': 0.1},
    {'name': 'RemoveRandomAdjective', 'p': 0.2},
]


class DataAugmentator:
    def __init__(self,
                 fixed_order_data_augmentation_steps=FIXED_ORDER_DATA_AUGMENTATION_STEPS,
                 rand_order_data_augmentation_steps=RAND_ORDER_DATA_AUGMENTATION_STEPS,
                 repeated_augmentation_steps=CHAR_CHANGES,
                 repeated_probs=[0.4, 0.3, 0.3]):

        self.fixed_order_data_augmentation_steps = fixed_order_data_augmentation_steps
        self.rand_order_data_augmentation_steps = rand_order_data_augmentation_steps
        self.repeated_augmentation_steps = repeated_augmentation_steps
        self.repeated_probs = repeated_probs

        self.slow_spell_checker = SpellChecker()
        self.sym_spell = SymSpell()

    def __GetCorrectedWord(self, word):
        # candidates = self.fast_spell_checker.GetCandidates([word], 0)
        try:
            candidates = self.sym_spell.lookup(word, Verbosity.ALL, max_edit_distance=3, )
        except ValueError:
            return None

        return np.random.choice(list(candidates)[:5])

    def __BuggyCorrectTypos(self, text):
        words = re.findall(r"\b[\w|']+\b", text)

        misspelled = self.slow_spell_checker.unknown(words)
        corrected_typo_count = min(10, len(misspelled))
        misspelled = np.random.choice(list(misspelled), corrected_typo_count, replace=10)

        corrected_text = text
        for word in misspelled:
            correction = self.slow_spell_checker.correction(word)
            if correction:
                corrected_text = corrected_text.replace(word, correction)

        return corrected_text

    def __DecapitalizeRandomLetter(self, text):
        capital_indices = [i for i, char in enumerate(text) if char.isupper()]
        if len(capital_indices) == 0:
            return text

        random_index = np.random.choice(capital_indices)

        modified_text = text[:random_index] + text[random_index].lower() + text[random_index + 1:]
        return modified_text

    def __CapitalizeRandomLetter(self, text):
        lower_indices = [i for i, char in enumerate(text) if char.islower()]
        if len(lower_indices) == 0:
            return text

        random_index = np.random.choice(lower_indices)
        modified_text = text[:random_index] + text[random_index].upper() + text[random_index + 1:]
        return modified_text

    def __RemoveRandomAdjective(self, text):
        tokens = text.split()
        tagged_tokens = pos_tag(tokens)

        # Identify all adjectives (JJ, JJR, JJS)
        adjectives = [word for word, tag in tagged_tokens if tag in ('JJ', 'JJR', 'JJS')]

        if not adjectives:
            return ' '.join(tokens)

        adjective_to_remove = random.choice(adjectives)
        idx = [i for i, el in enumerate(tokens) if el == adjective_to_remove]
        tokens.remove(adjective_to_remove)
        return ' '.join(tokens), idx

    def __SubsampleSentences(self, text, min_sentence=4, max_sentence=10):
        sentences = sent_tokenize(text)
        if len(sentences) <= min_sentence:
            # min_sentence = max_sentence = len(sentences)
            return ' '.join(sentences)

        cnt = random.randint(min_sentence, min(max_sentence, len(sentences)))
        ind = random.randint(0, len(sentences) - cnt)
        res = sentences[ind:ind + cnt]

        if random.random() > 0.5:
            sent_ind = random.randint(0, len(res[0]) - 1)
            res[0] = res[0][sent_ind:]

        if random.random() > 0.5:
            sent_ind = random.randint(0, len(res[-1]) - 1)
            res[-1] = res[-1][:sent_ind]

        return ' '.join(res)

    def __call__(self, text, labels):
        text = text.strip()

        random_augs = self.rand_order_data_augmentation_steps.copy()
        n_repeated = np.random.choice(np.arange(len(self.repeated_probs)), 1, p=self.repeated_probs)[0]
        for i in range(n_repeated):
            random_augs += self.repeated_augmentation_steps
        np.random.shuffle(random_augs)

        applied_augs = []

        for augmentation_step in (self.fixed_order_data_augmentation_steps + random_augs):
            augmentation_prob = augmentation_step['p']
            skip_step = np.random.uniform(0, 1) > augmentation_prob
            if skip_step:
                continue
            applied_augs.append(augmentation_step['name'])

            if augmentation_step['name'] == 'BuggySpellCheck':
                text = self.__BuggyCorrectTypos(text)
            elif augmentation_step['name'] == 'DecapitalizeRandomLetter':
                text = self.__DecapitalizeRandomLetter(text)
            elif augmentation_step['name'] == 'CapitalizeRandomLetter':
                text = self.__CapitalizeRandomLetter(text)
            elif augmentation_step['name'] == "RemoveRandomAdjective":
                text, idx = self.__RemoveRandomAdjective(text)
            # elif augmentation_step['name'] == 'SubsampleSentences':
            #     text = self.__SubsampleSentences(text)
            elif 'typo_' in augmentation_step['name']:
                error_type_name = augmentation_step['name'][5:]
                try:
                    text = eval(f'typo.StrErrer(text).{error_type_name}().result')
                except:
                    # Sometimes randomly fails for characters that don't have a keyboard neighbor.
                    pass
            else:
                raise Exception("Unexpected augmentation name: {}".format(augmentation_step['name']))

        n_auged = len(text.split())
        new_cnt_first_human = int(n_auged * (1 - sum(labels) / len(labels)))
        labels_auged = [0] * new_cnt_first_human + [1] * (n_auged - new_cnt_first_human)
        return text, applied_augs, labels_auged
