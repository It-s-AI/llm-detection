"""Full evaluation cycle of one submission (orchestrator).

quote + hold (billing) -> run the bundle (Lium pod) -> G0 normalization -> gates G1-G3 (levels) ->
crown detectors on the outputs that can still count -> score S and the crown test against the king ->
metered statement -> settle the hold -> records (private until the period is disclosed, D-58).
"""

from humanizer_eval.evaluation.context import KingRun, PeriodContext, load_period
from humanizer_eval.evaluation.metering import MeteringLLM
from humanizer_eval.evaluation.pipeline import (
    EvalConfig,
    EvalResult,
    EvaluationPaused,
    InsufficientBalance,
    Submission,
    SubmissionStore,
    evaluate_submission,
)
from humanizer_eval.evaluation.runners import LiumRunner, LocalHarnessRunner, ReplayRunner, RunOutput

__all__ = [
    "EvalConfig",
    "EvalResult",
    "EvaluationPaused",
    "InsufficientBalance",
    "KingRun",
    "LiumRunner",
    "LocalHarnessRunner",
    "MeteringLLM",
    "PeriodContext",
    "ReplayRunner",
    "RunOutput",
    "Submission",
    "SubmissionStore",
    "evaluate_submission",
    "load_period",
]
