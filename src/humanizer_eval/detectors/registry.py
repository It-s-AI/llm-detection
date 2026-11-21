"""Build configured detectors from settings."""

from __future__ import annotations

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import Settings, get_settings
from humanizer_eval.detectors.cache import DetectorCache
from humanizer_eval.detectors.vendors import GPTZero, ItsAI, Pangram, VendorDetector, ZeroGPT
from humanizer_eval.paths import ledger_path, var_dir

# Detectors that take part in crown scoring (D-30). It's AI is a simulation proxy only (D-33).
CROWN_DETECTORS = ("gptzero", "zerogpt", "pangram")


def build_detector(
    name: str,
    *,
    settings: Settings | None = None,
    ledger: SpendLedger | None = None,
    use_cache: bool = True,
    transport=None,
) -> VendorDetector:
    s = settings or get_settings()
    led = ledger or SpendLedger(ledger_path(), s.caps())
    cache = DetectorCache(var_dir() / "detector_cache.sqlite") if use_cache else None
    kw = {"ledger": led, "cache": cache, "transport": transport}
    if name == "gptzero":
        return GPTZero.build(s.gptzero_api_key, **kw)
    if name == "zerogpt":
        return ZeroGPT.build(s.zerogpt_api_key, **kw)
    if name == "pangram":
        return Pangram.build(s.pangram_api_key, **kw)
    if name == "its_ai":
        return ItsAI.build(s.its_ai_special_url, s.its_ai_api_key, **kw)
    raise ValueError(f"unknown detector {name!r}")
