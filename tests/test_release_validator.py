import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("release_validator", Path("scripts/release_validator.py"))
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)

SRC = """MECHID = 1
# Filled in at launch (the public key of the owner's leaderboard signing key and its URLs).
DEFAULT_PUBLIC_KEY = ""
DEFAULT_URLS: Tuple[str, ...] = ()
"""


def test_patch_fills_key_and_urls():
    out = rv.patch(SRC, "ab" * 32, ["https://a/v1/leaderboard", "https://b/lb.json"])
    assert f'DEFAULT_PUBLIC_KEY = "{"ab" * 32}"' in out
    assert 'DEFAULT_URLS: Tuple[str, ...] = ("https://a/v1/leaderboard", "https://b/lb.json",)' in out
    one = rv.patch(SRC, "ab" * 32, ["https://a"])
    assert 'DEFAULT_URLS: Tuple[str, ...] = ("https://a",)' in one  # a one-element tuple, not a string
    assert rv.patch(out, "cd" * 32, ["https://c"]).count("DEFAULT_PUBLIC_KEY") == 1


def test_patch_refuses_unknown_source():
    with pytest.raises(SystemExit):
        rv.patch("nothing here", "ab" * 32, ["https://a"])
