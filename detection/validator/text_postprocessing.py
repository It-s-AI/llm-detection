import re

import numpy as np


class TextCleaner:
    def __init__(self):
        pass

    def _remove_emoji(self, text: str) -> str:
        # remove emojies
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)

        text = emoji_pattern.sub(r'', text)
        return text

    def _remove_subtext(self, text: str) -> str:
        # remove words like *smiling*, *adjusts glasses*, etc
        last = None
        mask = np.ones(len(text))
        for i, c in enumerate(text):
            if c == '*':
                if last is None or (i - last) > 50:
                    last = i
                else:
                    mask[last:i + 1] = 0
                    last = None
        return ''.join([c for i, c in enumerate(text) if mask[i]])

    def clean_text(self, text: str) -> str:
        text = text.strip()
        text = self._remove_emoji(text)
        text = self._remove_subtext(text)
        return text