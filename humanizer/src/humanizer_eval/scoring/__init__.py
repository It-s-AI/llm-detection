"""Scorer and crown statistics (phase 5): stratified AUROC, soft-min, paired bootstrap, crown, payouts.

Pure numpy, no network: inputs are detector scores that the detector gateway already collected.
"""

from humanizer_eval.scoring.bootstrap import PairedBootstrapResult, paired_bootstrap
from humanizer_eval.scoring.crown import CrownDecision, KingEntry, KingState, crown_decision
from humanizer_eval.scoring.metric import (
    GATE_FAIL_SCORE,
    HumanReference,
    StratifiedAUROC,
    SubmissionRun,
    SubmissionScore,
    apply_gate,
    auroc,
    evasion,
    softmin,
    stratified_auroc,
    stratified_auroc_detail,
    submission_score,
)

__all__ = [
    "GATE_FAIL_SCORE",
    "CrownDecision",
    "HumanReference",
    "KingEntry",
    "KingState",
    "PairedBootstrapResult",
    "StratifiedAUROC",
    "SubmissionRun",
    "SubmissionScore",
    "apply_gate",
    "auroc",
    "crown_decision",
    "evasion",
    "paired_bootstrap",
    "softmin",
    "stratified_auroc",
    "stratified_auroc_detail",
    "submission_score",
]
