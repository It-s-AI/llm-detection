"""Free stand-ins for real detectors.

* `FakeDetector` — no HTTP; deterministic, programmable scores. For downstream
  modules (scoring, pipelines) and simulations.
* `fake_vendor_transport(vendor)` — an httpx.MockTransport that speaks each
  vendor's wire format (including 429s, 5xx and malformed bodies) so adapters can
  be tested without paying.
* `RecordingTransport` / `ReplayTransport` — record real responses once (secrets
  stripped), replay them forever in tests.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from pathlib import Path

import httpx

from humanizer_eval.detectors.base import DetectorScore, text_hash

SECRET_HEADERS = {"x-api-key", "apikey", "authorization", "x-oai-api-key"}


def _unit(text: str, salt: str) -> float:
    h = hashlib.sha256((salt + text).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


class FakeDetector:
    """Deterministic fake: score = rule(text) if a rule matches, else hash-based in [lo, hi]."""

    def __init__(
        self,
        name: str = "fake",
        rules: list[tuple[Callable[[str], bool], float]] | None = None,
        lo: float = 0.0,
        hi: float = 1.0,
        version: str = "fake-1",
    ):
        self.name = name
        self.rules = rules or []
        self.lo, self.hi = lo, hi
        self.version = version
        self.calls = 0

    def score(self, texts: list[str]) -> list[DetectorScore]:
        out = []
        for t in texts:
            self.calls += 1
            s = next((v for pred, v in self.rules if pred(t)), None)
            if s is None:
                s = self.lo + (self.hi - self.lo) * _unit(t, self.name)
            out.append(DetectorScore(self.name, text_hash(t), s, {"primary": s}, self.version))
        return out


# --- vendor wire-format fakes --------------------------------------------------


def _ai_prob(text: str) -> float:
    return round(_unit(text, "vendor"), 4)


def fake_vendor_transport(
    vendor: str, *, fail_first: int = 0, fail_status: int = 429, malformed: bool = False
) -> httpx.MockTransport:
    """MockTransport emulating `vendor`'s API. The first `fail_first` requests fail
    with `fail_status` (to exercise retries)."""
    state = {"n": 0, "tasks": {}}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] <= fail_first:
            return httpx.Response(fail_status, json={"error": "simulated"}, headers={"retry-after": "0"})
        if malformed:
            return httpx.Response(200, content=b"<html>oops</html>")
        body = json.loads(request.content or b"{}")
        if vendor == "gptzero":
            p = _ai_prob(body["document"])
            return httpx.Response(
                200,
                json={
                    "version": "fake-gptzero-1",
                    "documents": [
                        {
                            "class_probabilities": {
                                "human": round(1 - p, 4),
                                "ai": round(p * 0.8, 4),
                                "mixed": round(p * 0.2, 4),
                            },
                            "completely_generated_prob": p,
                            "average_generated_prob": p,
                            "predicted_class": "ai",
                        }
                    ],
                },
            )
        if vendor == "zerogpt":
            p = _ai_prob(body["input_text"])
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "code": 200,
                    "message": "ok",
                    "data": {
                        "fakePercentage": round(p * 100, 2),
                        "isHuman": 0,
                        "textWords": len(body["input_text"].split()),
                    },
                },
            )
        if vendor == "pangram":
            if request.method == "POST" and request.url.path == "/task":
                tid = uuid.uuid4().hex
                state["tasks"][tid] = {"text": body["text"], "polls": 0}
                return httpx.Response(200, json={"task_id": tid})
            if request.method == "GET" and request.url.path.startswith("/task/"):
                tid = request.url.path.split("/")[-1]
                task = state["tasks"][tid]
                task["polls"] += 1
                if task["polls"] < 2:
                    return httpx.Response(200, json={"task_id": tid, "stage": "STAGE_PREPROCESSING"})
                p = _ai_prob(task["text"])
                return httpx.Response(
                    200,
                    json={
                        "stage": "STAGE_SUCCESS",
                        "version": "4.1",
                        "headline": "fake",
                        "fraction_ai": p,
                        "fraction_ai_assisted": 0.0,
                        "fraction_human": 1 - p,
                        "windows": [
                            {
                                "label": "AI-Generated",
                                "token_length": 100,
                                "humanizer_score": p / 2,
                                "class_probabilities": [1 - p, 0.0, p],
                                "confidence": "High",
                            }
                        ],
                    },
                )
        if vendor == "its_ai":
            return httpx.Response(
                200, json={"preds": [[_ai_prob(t)] * len(t.split()) for t in body["texts"]]}
            )
        return httpx.Response(404, json={"error": "unknown fake route"})

    return httpx.MockTransport(handler)


# --- record / replay -------------------------------------------------------------


def _fingerprint(request: httpx.Request) -> str:
    body = request.content or b""
    try:
        obj = json.loads(body)
        obj.pop("api_key", None)  # It's AI sends the key in the body
        body = json.dumps(obj, sort_keys=True).encode()
    except ValueError:
        pass
    return hashlib.sha256(request.method.encode() + request.url.path.encode() + body).hexdigest()


class RecordingTransport(httpx.BaseTransport):
    """Wraps a real transport and appends (request fingerprint, response) to a JSONL file.
    Secrets (auth headers, api_key body fields) are never written."""

    def __init__(self, path: Path | str, inner: httpx.BaseTransport | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        resp = self.inner.handle_request(request)
        resp.read()
        rec = {
            "fp": _fingerprint(request),
            "method": request.method,
            "path": request.url.path,
            "status": resp.status_code,
            "body": resp.text,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        # The body is already decoded: drop encoding/length headers so the client
        # does not try to decompress it a second time.
        drop = SECRET_HEADERS | {"content-encoding", "content-length", "transfer-encoding"}
        headers = {k: v for k, v in resp.headers.items() if k.lower() not in drop}
        return httpx.Response(resp.status_code, content=resp.content, headers=headers)


class ReplayTransport(httpx.BaseTransport):
    """Serves recorded responses by request fingerprint (polling GETs are replayed in order)."""

    def __init__(self, path: Path | str):
        self.by_fp: dict[str, list[dict]] = {}
        for line in Path(path).read_text().splitlines():
            rec = json.loads(line)
            self.by_fp.setdefault(rec["fp"], []).append(rec)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        recs = self.by_fp.get(_fingerprint(request))
        if not recs:
            return httpx.Response(599, json={"error": "no recording for request"})
        rec = recs.pop(0) if len(recs) > 1 else recs[0]
        return httpx.Response(rec["status"], content=rec["body"].encode())
