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
