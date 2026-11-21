"""Pool test configuration.

``datasets`` streaming leaves fsspec/aiohttp threads behind that can abort the interpreter during
finalization (``PyGILState_Release``). After a run that executed live (network) tests, exit right after
pytest has reported, with pytest's own exit status, instead of going through finalization.
"""

import os
import sys

_live_ran = False


def pytest_runtest_setup(item):
    global _live_ran
    if item.get_closest_marker("live"):
        _live_ran = True


def pytest_unconfigure(config):
    if _live_ran:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(int(getattr(config, "_pool_exitstatus", 0)))


def pytest_sessionfinish(session, exitstatus):
    session.config._pool_exitstatus = int(exitstatus)
