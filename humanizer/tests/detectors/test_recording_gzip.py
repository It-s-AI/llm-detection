import gzip
import json

import httpx

from humanizer_eval.detectors.fake import RecordingTransport


def test_recording_transport_handles_gzip(tmp_path):
    payload = json.dumps({"ok": True}).encode()

    def handler(_r):
        return httpx.Response(200, content=gzip.compress(payload), headers={"content-encoding": "gzip"})

    rec = RecordingTransport(tmp_path / "r.jsonl", inner=httpx.MockTransport(handler))
    with httpx.Client(transport=rec) as c:
        assert c.post("http://x/y", json={"a": 1}).json() == {"ok": True}
    assert json.loads((tmp_path / "r.jsonl").read_text())["body"] == payload.decode()
