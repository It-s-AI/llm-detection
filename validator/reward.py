# The MIT License (MIT)
# Copyright © 2024 It's AI
import traceback
import torch
from typing import List
import bittensor as bt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, average_precision_score
from utils.protocol import TextRequest

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
        labels: list[list[list[bool]]],
        responses: List[TextRequest],
        check_responses: List[TextRequest],
        version_responses: List[TextRequest],
        check_ids: np.array,
        out_of_domain_ids: np.array,
        update_out_of_domain: bool = False
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
    predictions_list = [np.concatenate(synapse.predictions) if len(synapse.predictions) else [] for synapse in responses]
    check_predictions_list = [np.concatenate(synapse.predictions) if len(synapse.predictions) else [] for synapse in check_responses]
    version_predictions_list = [synapse.predictions for synapse in version_responses]

    flatten_check_ids = []
    for uid_labels in labels:
        cur_mask = np.concatenate([len(text_labels) * [i in check_ids] for i, text_labels in enumerate(uid_labels)])
        cur_ids = np.where(cur_mask)[0]
        flatten_check_ids.append(cur_ids)

    flatten_out_of_domain_masks = []
    for uid_labels in labels:
        cur_mask = np.concatenate([len(text_labels) * [i in out_of_domain_ids] for i, text_labels in enumerate(uid_labels)])
        flatten_out_of_domain_masks.append(cur_mask)

    labels = [np.concatenate(el) for el in labels]

    rewards = []
    metrics = []
    for uid in range(len(predictions_list)):
        try:
            if len(predictions_list[uid]) != len(labels[uid]) or len(check_predictions_list[uid]) != len(flatten_check_ids[uid]):
                rewards.append(0)
                metrics.append({'fp_score': 0, 'f1_score': 0, 'ap_score': 0, 'penalty': 1})
                continue

            predictions_array = np.array(predictions_list[uid])
            check_predictions_array = np.array(check_predictions_list[uid])

            bt.logging.info(f"{predictions_array.shape}, {labels[uid].shape}, {check_predictions_array.shape}, {flatten_check_ids[uid].shape}")

            mask = flatten_out_of_domain_masks[uid]
            out_of_domain_miner_reward, out_of_domain_metric = reward(predictions_array[mask], labels[uid][mask])

            miner_reward, metric = reward(predictions_array[~mask], labels[uid][~mask])
            penalty = count_penalty(predictions_array, check_predictions_array, flatten_check_ids[uid], version_predictions_list[uid])

            if update_out_of_domain:
                self.out_of_domain_f1_scores[uid] = self.out_of_domain_f1_scores[uid] * (1 - self.out_of_domain_alpha) + \
                                                    out_of_domain_metric['f1_score'] * self.out_of_domain_alpha

            if self.out_of_domain_f1_scores[uid] < self.subnet_config.neuron.out_of_domain_min_f1_score:
                penalty = 0

            miner_reward *= penalty
            rewards.append(miner_reward)
            metric['penalty'] = penalty
            metric['out_of_domain_f1_score'] = out_of_domain_metric['f1_score']
            metric['weighted_out_of_domain_f1_score'] = self.out_of_domain_f1_scores[uid]
            metrics.append(metric)
        except Exception as e:
            bt.logging.error("Couldn't count miner reward for {}, his predictions = {} and his labels = {}".format(uid, predictions_list[uid], labels[uid]))
            bt.logging.info(traceback.format_exc())
            rewards.append(0)
            metrics.append({'fp_score': 0, 'f1_score': 0, 'ap_score': 0, 'penalty': 1})

    return torch.FloatTensor(rewards), metrics
