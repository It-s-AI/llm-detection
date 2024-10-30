import random

from nltk import pos_tag


class DeleteAttack:
    def __init__(self, max_remove_words=5):
        self.max_remove_words = max_remove_words

    def remove_random_adjective(self, text):
        tokens = text.split()
        tagged_tokens = pos_tag(tokens)

        adjectives = [word for word, tag in tagged_tokens if tag in ('JJ', 'JJR', 'JJS')]

        if not adjectives:
            return ' '.join(tokens)

        adjective_to_remove = random.choice(adjectives)
        tokens.remove(adjective_to_remove)
        return ' '.join(tokens)

    def attack(self, text):
        n = random.randint(1, self.max_remove_words)
        for i in range(n):
            text = self.remove_random_adjective(text)

        return text
