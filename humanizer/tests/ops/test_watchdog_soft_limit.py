import importlib.util
from pathlib import Path

from humanizer_eval.billing.accounts import Accounts
from humanizer_eval.budget import SpendLedger

spec = importlib.util.spec_from_file_location("watchdog", Path("scripts/watchdog.py"))
wd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wd)


def test_soft_limit_counts_spend_and_miners_money_since_the_launch(tmp_path):
    inf = float("inf")
    led = SpendLedger(tmp_path / "ledger.sqlite", {s: inf for s in (*wd.SERVICES, "its_ai")})
    acc = Accounts(tmp_path / "billing.sqlite")
    led.reserve("pangram", 50.0).commit(40.0)  # before the launch: dev spend does not count
    acc.record_payment("old", "hk", 10_000, 0)
    import time

    since = time.time() + 0.001
    time.sleep(0.01)
    led.reserve("pangram", 30.0).commit(25.0)
    led.reserve("gptzero", 5.0)  # open: counts at its estimate
    led.reserve("zerogpt", 9.0).release()  # released: nothing was spent
    led.reserve("its_ai", 7.0).commit(7.0)  # our own service: not a cost
    acc.record_payment("new", "hk", 2_000, 0)
    spent, allowed = wd.soft_limit(tmp_path / "ledger.sqlite", tmp_path / "billing.sqlite", since, 10.0)
    assert spent == 30.0 and allowed == 30.0
