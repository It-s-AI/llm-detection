"""Runtime settings loaded from environment variables (and an optional .env file).

Secrets are only ever read from the environment / .env and are never logged:
`Settings.__repr__` masks every field whose name looks like a secret.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
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
    zerogpt_api_key: str = ""
    pangram_api_key: str = ""
    its_ai_special_url: str = ""
    its_ai_api_key: str = ""
    # external reference humanizer (pilots only)
    phrasly_api_key: str = ""

    # miner billing (D-80); test keys until launch
    stripe_secret_key: str = ""
    stripe_publishable_key: str = Field(
        default="", validation_alias=AliasChoices("stripe_publishable_key", "stripe_pushishable_key")
    )
    stripe_webhook_secret: str = ""

    # llm gateway / data / execution
    openrouter_api_key: str = ""
    hf_api_key: str = ""
    hf_mirror_token: str = ""  # write token of our HF org: public copies of king bundles (D-70)
    alarmer_key: str = ""  # Telegram alerts via @alarmer_bot (ops/alerts.py)
    lium_api_key: str = ""
    lium_enabled: bool = False

    # hard spending caps, USD
    budget_pangram_usd: float = 30.0
    budget_gptzero_usd: float = 20.0
    budget_zerogpt_usd: float = 10.0
    budget_openrouter_usd: float = 25.0
    budget_lium_usd: float = 50.0
    budget_phrasly_usd: float = 15.0
    lium_balance_floor_hours: float = 48.0

    var_dir: Path = Field(default=Path("var"), validation_alias="HUMEVAL_VAR_DIR")

    def caps(self) -> dict[str, float]:
        return {
            "pangram": self.budget_pangram_usd,
            "gptzero": self.budget_gptzero_usd,
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
