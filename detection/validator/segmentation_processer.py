import numpy as np
import random


class SegmentationProcesser:
    def __init__(self, ):
        pass

    def merge_prompt_text(self, prompt, text):
        now = {}
        el = {'prompt': prompt, 'text': text}
        if not prompt:
            raise Exception("There is should be a prompt during merging")

        if np.random.random() < 0.67:
            now['text'] = el['prompt'] + el['text']
            now['cnt_first_human'] = len(el['prompt'].split())
        else:
            now['cnt_first_human'] = 0
            now['text'] = el['text']

        return now['text'], now['cnt_first_human']

    def subsample_words(self, text, cnt_first_human, min_cnt=50, max_cnt=250):
        words = text.split()
        labels = [0] * cnt_first_human + [1] * (len(words) - cnt_first_human)
        if len(words) <= min_cnt:
            return ' '.join(words), labels

        cnt = random.randint(min_cnt, min(max_cnt, len(words)))
        if cnt_first_human > 0 and cnt_first_human < len(words):
            ind = random.randint(max(cnt_first_human - cnt, 0), min(len(words) - cnt, cnt_first_human + cnt))
        else:
            ind = random.randint(0, len(words) - cnt)

        res = words[ind:ind + cnt]
        labels = labels[ind:ind + cnt]

        if random.random() > 0.5 and len(res):
            sent_ind = random.randint(0, len(res[0]) - 1)
            res[0] = res[0][sent_ind:]

        if random.random() > 0.5:
            sent_ind = random.randint(0, len(res[-1]) - 1)
            res[-1] = res[-1][:sent_ind]

        return ' '.join(res), labels
