"""Runtime settings loaded from environment variables (and an optional .env file).

Secrets are only ever read from the environment / .env and are never logged:
`Settings.__repr__` masks every field whose name looks like a secret.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "URL")


def _env_files() -> tuple[str, ...]:
    """.env lookup order: $HUMEVAL_ENV_FILE, ./.env, ../.env (the owner workspace)."""
    candidates = [os.environ.get("HUMEVAL_ENV_FILE"), ".env", "../.env"]
    return tuple(p for p in candidates if p and Path(p).is_file())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_env_files(), extra="ignore", case_sensitive=False)

    # detectors
    gptzero_api_key: str = ""
    originality_api_key: str = ""
    winston_api_key: str = ""
    zerogpt_api_key: str = ""
    pangram_api_key: str = ""
    its_ai_special_url: str = ""
    its_ai_api_key: str = ""
    # external reference humanizer (pilots only)
    phrasly_api_key: str = ""

    # llm gateway / data / execution
    openrouter_api_key: str = ""
    hf_api_key: str = ""
    hf_mirror_token: str = ""  # write token of our HF org: public copies of king bundles (D-70)
    alarmer_key: str = ""  # Telegram alerts via @alarmer_bot (ops/alerts.py)
    lium_api_key: str = ""
    tao_rpc: str = ""  # Bittensor EVM JSON-RPC for payments (default: billing/evm.py RouteConfig)
    base_rpc: str = ""  # Base JSON-RPC (default: billing/evm.py RouteConfig)
    lium_enabled: bool = False

    # hard spending caps, USD
    budget_pangram_usd: float = 30.0
    budget_gptzero_usd: float = 20.0
    budget_originality_usd: float = 20.0
    budget_winston_usd: float = 20.0
    budget_zerogpt_usd: float = 10.0
    budget_openrouter_usd: float = 25.0
    budget_lium_usd: float = 50.0
    budget_phrasly_usd: float = 15.0
    #: the soft limit (scripts/watchdog.py): since this UTC time (ISO 8601) our spend on every paid service
    #: should stay under what miners paid in plus this gap; past it the owner gets an alert, nothing stops
    soft_limit_since: str = ""
    soft_limit_gap_usd: float = 300.0
    #: and this much more for every period begun since then (our own spend: H and the king's run)
    soft_limit_weekly_usd: float = 300.0
    #: automatic payout of the earnings (scripts/tao_treasury.py --payout, a timer): USDC on Base above what
    #: miners are owed (balances, open holds, refunds not bought yet) and a reserve goes to this address.
    #: Empty: off.
    payout_address: str = ""
    payout_reserve_usdc: float = 50.0  # kept for refund fees and slippage
    payout_min_usdc: float = 20.0  # smaller amounts wait (fees per transfer on the receiving side)
    lium_balance_floor_hours: float = 48.0
    #: the Lium account tops itself up: a rent then needs a positive balance only, not the pod's whole cost
    lium_auto_topup: bool = False

    var_dir: Path = Field(default=Path("var"), validation_alias="HUMEVAL_VAR_DIR")

    def caps(self) -> dict[str, float]:
        return {
            "pangram": self.budget_pangram_usd,
            "gptzero": self.budget_gptzero_usd,
            "originality": self.budget_originality_usd,
            "winston": self.budget_winston_usd,
            "zerogpt": self.budget_zerogpt_usd,
            "openrouter": self.budget_openrouter_usd,
            "lium": self.budget_lium_usd,
            "phrasly": self.budget_phrasly_usd,
            # It's AI is our own service: no cost, but still metered for visibility.
            "its_ai": float("inf"),
        }

    def __repr_args__(self):  # type: ignore[override]
        for name, value in super().__repr_args__():
            if name and any(m in name.upper() for m in _SECRET_MARKERS) and value:
                yield name, "***"
            else:
                yield name, value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
