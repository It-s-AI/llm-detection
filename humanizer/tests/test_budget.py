import threading

import pytest

from humanizer_eval.budget import BudgetExceededError, SpendLedger


def make(tmp_path, cap=1.0):
    return SpendLedger(tmp_path / "l.sqlite", {"svc": cap})


def test_reserve_commit_and_cap(tmp_path):
    led = make(tmp_path, cap=1.0)
    r = led.reserve("svc", 0.6)
    assert led.outstanding("svc") == pytest.approx(0.6)
    r.commit(0.5)
    assert led.spent("svc") == pytest.approx(0.5)
    assert led.outstanding("svc") == 0
    with pytest.raises(BudgetExceededError):
        led.reserve("svc", 0.6)
    led.reserve("svc", 0.5).commit(0.5)
    assert led.remaining("svc") == pytest.approx(0.0)


def test_release_frees_budget(tmp_path):
    led = make(tmp_path, cap=1.0)
    led.reserve("svc", 0.9).release()
    led.reserve("svc", 0.9).commit(0.9)
    assert led.spent("svc") == pytest.approx(0.9)


def test_unknown_service_refused(tmp_path):
    led = make(tmp_path)
    with pytest.raises(BudgetExceededError):
        led.reserve("other", 0.01)


def test_open_reservation_counts_against_cap(tmp_path):
    led = make(tmp_path, cap=1.0)
    led.reserve("svc", 0.8)  # never closed (simulated crash)
    with pytest.raises(BudgetExceededError):
        led.reserve("svc", 0.3)
    # a new ledger instance on the same file sees the open reservation too
    led2 = SpendLedger(tmp_path / "l.sqlite", {"svc": 1.0})
    with pytest.raises(BudgetExceededError):
        led2.reserve("svc", 0.3)


def test_spend_context_commits_at_estimate_on_error(tmp_path):
    led = make(tmp_path, cap=1.0)
    with pytest.raises(RuntimeError), led.spend("svc", 0.4):
        raise RuntimeError("boom")
    assert led.spent("svc") == pytest.approx(0.4)


def test_concurrent_reservations_never_exceed_cap(tmp_path):
    led = make(tmp_path, cap=1.0)
    ok = []

    def worker():
        try:
            SpendLedger(tmp_path / "l.sqlite", {"svc": 1.0}).reserve("svc", 0.1).commit(0.1)
            ok.append(1)
        except BudgetExceededError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ok) == 10
    assert led.spent("svc") == pytest.approx(1.0)


def test_pool_limit_follows_the_miners_money(tmp_path):
    """The dynamic limit: all paid services together may spend what miners paid in plus the gap; each
    payment raises it, a refund lowers it, a loop of paid calls is stopped at the gap."""
    from humanizer_eval.billing import Accounts
    from humanizer_eval.budget import PoolLimit

    acc = Accounts(tmp_path / "billing.sqlite")
    inf = float("inf")
    caps = dict.fromkeys(("pangram", "gptzero", "zerogpt", "openrouter", "lium"), inf)
    led = SpendLedger(
        tmp_path / "ledger.sqlite", caps, pool=PoolLimit(tmp_path / "billing.sqlite", 0.0, 50.0)
    )
    led.reserve("pangram", 30).commit(30)
    led.reserve("openrouter", 20).commit(20)  # exactly the gap
    with pytest.raises(BudgetExceededError, match="spend limit"):
        led.reserve("lium", 1)  # a bug looping on paid calls stops here
    acc.record_payment("p1", "5Fminer", 15000, 0)  # a miner pays $150: $150 more room
    led.reserve("pangram", 140).commit(140)
    with pytest.raises(BudgetExceededError):
        led.reserve("gptzero", 11)
    r = led.reserve("gptzero", 10)  # a released reservation gives its room back
    r.release()
    led.reserve("gptzero", 10).commit(10)
    with pytest.raises(BudgetExceededError):
        led.reserve("zerogpt", 0.01)


def test_pool_limit_adds_a_weekly_allowance_per_period_begun(tmp_path):
    """Our own weekly spend (building a period) has its own room: gap + weekly × periods begun; it does not
    eat the gap, and with no miner money a loop is still stopped."""
    from humanizer_eval.budget import PoolLimit, pool_allowance

    day = 86400.0
    assert pool_allowance(1000.0, 300, 150, now=999.0) == 300  # before the launch
    assert pool_allowance(0.0, 300, 150, now=0.0) == 450  # the first period
    assert pool_allowance(0.0, 300, 150, now=6.9 * day) == 450
    assert pool_allowance(0.0, 300, 150, now=7 * day) == 600  # the second begins
    assert pool_allowance(0.0, 300, 150, now=15 * day) == 750
    inf = float("inf")
    caps = dict.fromkeys(("pangram", "gptzero", "zerogpt", "openrouter", "lium"), inf)
    since = __import__("time").time() - 8 * day  # in the second period
    led = SpendLedger(tmp_path / "ledger.sqlite", caps,
                      pool=PoolLimit(tmp_path / "billing.sqlite", since, 50.0, weekly_usd=100.0))  # fmt: skip
    led.reserve("pangram", 250).commit(250)  # 50 + 2 × 100
    with pytest.raises(BudgetExceededError, match="weekly allowance"):
        led.reserve("gptzero", 0.01)
