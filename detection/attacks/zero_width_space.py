import random


class ZeroWidthSpaceAttack:
    def __init__(self, max_p=0.2):
        self.max_p = max_p

    def attack(self, text):
        cur_p = self.max_p * random.random()

        res = ""
        for word in text.split():
            res += word
            if random.random() > cur_p:
                res += ' '

        return res
