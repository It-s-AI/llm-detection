from humanizer_eval.llm.ratelimit import ModelRateLimiter, RateLimitedLLM


class Clock:
    def __init__(self, t):
        self.t = t

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_per_model_calendar_minute_and_models_do_not_wait_for_each_other():
    c = Clock(1000 * 60 + 10.0)  # 10 s into a minute
    lim = ModelRateLimiter(3, exempt=("free/",), now=c.now, sleep=c.sleep)
    assert [lim.wait("a") for _ in range(3)] == [0, 0, 0]
    assert lim.wait("b") == 0  # another model: its own budget
    w = lim.wait("a")  # the 4th call to "a" waits for the next minute (+ skew)
    assert 50 < w < 52 and c.t % 60 > 1.0
    assert all(lim.wait("free/x") == 0 for _ in range(10))  # exempt models are never limited


def test_off_and_wrapper_delegates():
    class LLM:
        ledger = "L"

        def chat(self, model, messages, **kw):
            return (model, messages, kw)

    w = RateLimitedLLM(LLM(), ModelRateLimiter(0))
    assert w.chat("m", [1], purpose="p") == ("m", [1], {"purpose": "p"}) and w.ledger == "L"


def test_a_call_in_the_skew_seconds_counts_in_both_minutes():
    """Up to ``skew`` seconds past :00 OpenRouter may still be in the old minute or already in the new one:
    such a call needs room in both, else the server's calendar minute could see up to 2x rpm."""
    from humanizer_eval.llm.ratelimit import ModelRateLimiter

    t = [59.0]
    slept = []
    lim = ModelRateLimiter(
        2, skew=1.0, now=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s))
    )
    lim.wait("m")  # minute 0: 1
    t[0] = 60.5
    lim.wait("m")  # in the skew: minute 0 -> 2, minute 1 -> 1
    assert not slept
    lim.wait("m")  # minute 0 is full: waits until 61.05, then only minute 1 counts (-> 2)
    assert slept and t[0] >= 61.0
    lim.wait("m")  # minute 1 full now: waits for minute 2
    assert t[0] >= 121.0
