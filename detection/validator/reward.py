# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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


def reward(y_pred: np.array, y_true: np.array) -> float:
    """
    Reward the miner response to the dummy request. This method returns a reward
    value for the miner, which is used to update the miner's score.

    Returns:
    - float: The reward value for the miner.
    """
    preds = y_pred.astype(int)

    # accuracy = accuracy_score(y_true, preds)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    f1 = f1_score(y_true, preds)
    ap_score = average_precision_score(y_true, y_pred)

    res = {'fp_score': 1 - fp / len(y_pred),
            'f1_score': f1,
            'ap_score': ap_score}
    reward = sum([v for v in res.values()]) / len(res)
    return reward


def count_penalty(y_pred: np.array) -> float:
    bad = np.any((y_pred < 0) | (y_pred > 1))
    return 0 if bad else 1

    
def get_rewards(
    self,
    labels: torch.FloatTensor,
    responses: List[float],
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

    rewards = []
    for uid in range(len(predictions_list)):
        # if there is no answer reward should be 0
        if not predictions_list[uid]:
            rewards.append(0)
            continue

        predictions_array = np.array(predictions_list[uid])
        miner_reward = reward(predictions_array, labels)

        miner_reward *= count_penalty(predictions_array)
        rewards.append(miner_reward)

    return torch.FloatTensor(rewards)
