import numpy as np
import random


HUMAN_THEN_AI_PERCENT = 40
AI_PERCENT = 25

class SegmentationProcesser:
    def __init__(self, ):
        pass

    def merge_prompt_text(self, prompt, text):
        now = {}
        el = {'prompt': prompt, 'text': text}
        if not prompt:
            raise Exception("There is should be a prompt during merging")

        if np.random.random() < HUMAN_THEN_AI_PERCENT / (HUMAN_THEN_AI_PERCENT + AI_PERCENT):
            now['text'] = el['prompt'] + el['text']
            now['cnt_first_human'] = len(el['prompt'].split())
        else:
            now['cnt_first_human'] = 0
            now['text'] = el['text']

        return now['text'], now['cnt_first_human']

    def subsample_words(self, text, labels, min_cnt=35, max_cnt=350):
        words = text.split()
        if len(words) <= min_cnt:
            return ' '.join(words), labels

        cnt = random.randint(min_cnt, min(max_cnt, len(words)))

        has_01 = False
        has_10 = False

        for i in range(len(labels) - 1):
            if labels[i] == 0 and labels[i + 1] == 1:
                has_01 = True
            if labels[i] == 1 and labels[i + 1] == 0:
                has_10 = True

        if has_01 and has_10:
            # if random.random() < 0.5:
            # currently we always take ai the first and then human
            ind = None
            for i in range(len(labels) - 1):
                if labels[i] == 0 and labels[i + 1] == 1:
                    ind = i + 1
                    break
            return self.subsample_words(' '.join(words[ind:]), labels[ind:])
            # else:
            #     ind = None
            #     for i in range(len(labels) - 1):
            #         if labels[i] == 1 and labels[i + 1] == 0:
            #             ind = i + 1
            #             break
            #     return self.subsample_words(' '.join(words[:ind]), labels[:ind])

        split_index = None
        for i in range(len(labels) - 1):
            if labels[i] != labels[i + 1]:
                split_index = i
                break
        
        if split_index is not None:  # for two class case
            ind = random.randint(max(split_index - cnt, 0), min(len(words) - cnt, split_index))
        else:  # for one class case
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
