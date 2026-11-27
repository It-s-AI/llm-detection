"""AI-text detector gateway (phase 1)."""

from humanizer_eval.detectors.base import Detector, DetectorError, DetectorScore, text_hash
from humanizer_eval.detectors.fake import FakeDetector
from humanizer_eval.detectors.registry import CROWN_DETECTORS, build_detector

__all__ = [
    "CROWN_DETECTORS",
    "Detector",
    "DetectorError",
    "DetectorScore",
    "FakeDetector",
    "build_detector",
    "text_hash",
]
