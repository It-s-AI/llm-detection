import logging

import httpx

from humanizer_eval.ops.alerts import Alerter


def test_the_alarm_key_never_reaches_the_log(caplog, tmp_path):
    """httpx logs the request URL at INFO; the key in its query is masked."""
    key = "k" * 40
    http = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200)))
    caplog.set_level(logging.INFO)
    assert Alerter(key, http=http, state_path=tmp_path / "s.json").send("hello")
    text = caplog.text
    assert "HTTP Request" in text and key not in text and "key=***" in text
