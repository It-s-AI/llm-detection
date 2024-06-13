# The MIT License (MIT)
 # Copyright © 2024 It's AI 
 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
from typing import List
import bittensor as bt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, average_precision_score

from detection.protocol import TextSynapse

import time


def reward(y_pred: np.array, y_true: np.array) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    preds = np.round(y_pred)

    # accuracy = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    f1 = f1_score(y_true, preds)
    ap_score = average_precision_score(y_true, y_pred)

    res = {'fp_score': 1 - fp / len(y_pred),
            'f1_score': f1,
            'ap_score': ap_score}
    reward = sum([v for v in res.values()]) / len(res)
    return reward, res


def count_penalty(
    y_pred: np.array,
    check_predictions: np.array,
    check_ids: np.array,
    version_predictions_array: List
    ) -> float:
    bad = np.any((y_pred < 0) | (y_pred > 1))

    if (check_predictions.round(2) != y_pred[check_ids].round(2)).any():
        bad = 1

    if version_predictions_array:
        bad = 1

    return 0 if bad else 1

    
def get_rewards(
    self,
    labels: np.array,
    responses: List[TextSynapse],
    check_responses: List[TextSynapse],
    version_responses: List[TextSynapse],
    check_ids: np.array
) -> torch.FloatTensor:
    """
    Returns a tensor of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - torch.FloatTensor: A tensor of rewards for the given query and responses.
    """
    # Get all the reward results by iteratively calling your reward() function.
    predictions_list = [synapse.predictions for synapse in responses]
    check_predictions_list = [synapse.predictions for synapse in check_responses]
    version_predictions_list = [synapse.predictions for synapse in version_responses]

    rewards = []
    metrics = []
    for uid in range(len(predictions_list)):
        try:
            if not predictions_list[uid] or len(predictions_list[uid]) != len(labels) or \
                    not check_predictions_list[uid] or len(check_predictions_list[uid]) != len(check_ids):
                # if not version_predictions_list[uid] and check_predictions_list[uid]:
                #     bt.logging.info(f"VERSION BLOCKED for #{uid} miner: got {version_predictions_list[uid]}")
                rewards.append(0)
                metrics.append({'fp_score': 0, 'f1_score': 0, 'ap_score': 0, 'penalty': 1})
                continue

            predictions_array = np.array(predictions_list[uid])
            check_predictions_array = np.array(check_predictions_list[uid])

            miner_reward, metric = reward(predictions_array, labels)
            penalty = count_penalty(
                predictions_array, check_predictions_array, check_ids, version_predictions_list[uid])

            miner_reward *= penalty
            rewards.append(miner_reward)
            metric['penalty'] = penalty
            metrics.append(metric)
        except Exception as e:
            bt.logging.error("Couldn't count miner reward for {}, his predictions = {} and his labels = {}".format(uid, predictions_list[uid], labels))
            bt.logging.exception(e)
            rewards.append(0)
            metrics.append({'fp_score': 0, 'f1_score': 0, 'ap_score': 0, 'penalty': 1})

    return torch.FloatTensor(rewards), metrics
