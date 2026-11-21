import httpx

from humanizer_eval.ops.alerts import Alerter


def _client(seen, status=200):
    def handler(req):
        seen.append(dict(req.url.params))
        return httpx.Response(status, text="OK")

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_alert_goes_out_with_key_and_prefix(tmp_path):
    seen = []
    a = Alerter("k-1", http=_client(seen), state_path=tmp_path / "s.json")
    assert a.send("pod stuck") is True
    assert seen == [{"key": "k-1", "message": "[SN32 humanizer] pod stuck"}]


def test_dedupe_holds_repeats_until_the_interval_passes(tmp_path):
    seen, t = [], [1000.0]
    a = Alerter("k", http=_client(seen), state_path=tmp_path / "s.json", min_interval_s=600, now=lambda: t[0])
    assert a.send("low balance", dedupe="lium") is True
    assert a.send("low balance", dedupe="lium") is False
    b = Alerter("k", http=_client(seen), state_path=tmp_path / "s.json", min_interval_s=600, now=lambda: t[0])
    assert b.send("low balance", dedupe="lium") is False  # another process sees the state file
    t[0] += 601
    assert b.send("low balance", dedupe="lium") is True and len(seen) == 2


def test_failures_and_a_missing_key_never_raise(tmp_path):
    assert Alerter("", state_path=tmp_path / "s.json").send("x") is False
    seen = []
    assert Alerter("k", http=_client(seen, status=500)).send("x") is False

    def boom(req):
        raise httpx.ConnectError("down")

    assert Alerter("k", http=httpx.Client(transport=httpx.MockTransport(boom))).send("x") is False
