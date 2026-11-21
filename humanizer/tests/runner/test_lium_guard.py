import json
import re
from pathlib import Path

import httpx
import pytest

from humanizer_eval.budget import SpendLedger
from humanizer_eval.runner.lium import POD_PREFIX, GuardError, LiumClient, Registry

OTHER = {"id": "09ad96e9-775e-4c22-92da-d28ec3a82a4b", "pod_name": "sn107-research", "status": "RUNNING"}
EXEC = {"id": "03ed3a8e-a9f0-404f-983a-65797b567d41", "price_per_gpu": 0.5, "machine_name": "GPU"}
NEW_ID = "11111111-2222-3333-4444-555555555555"


class FakeLium:
    def __init__(self, balance=131.6, burn=2.2, schedule_fails=False):
        self.balance, self.burn = balance, burn
        self.pods = [dict(OTHER)]
        self.calls = []
        self.schedule_fails = schedule_fails

    def handler(self, request: httpx.Request) -> httpx.Response:
        m, p = request.method, request.url.path.removeprefix("/api")
        self.calls.append((m, p))
        if (m, p) == ("GET", "/users/me"):
            return httpx.Response(200, json={"balance": self.balance, "burn_per_hour": self.burn})
        if (m, p) == ("GET", "/pods"):
            return httpx.Response(200, json=self.pods)
        if m == "POST" and p.endswith("/rent"):
            body = json.loads(request.content)
            self.pods.append({"id": NEW_ID, "pod_name": body["pod_name"], "status": "PENDING"})
            return httpx.Response(200, json={"id": NEW_ID})
        if m == "POST" and p.endswith("/schedule-removal"):
            return httpx.Response(500 if self.schedule_fails else 200, json={})
        if m == "DELETE" and p.startswith("/pods/"):
            pid = p.split("/")[2]
            self.pods = [x for x in self.pods if x["id"] != pid]
            return httpx.Response(200, json={})
        if m == "GET" and p.endswith("/statement"):
            return httpx.Response(200, json={"total": 0.12})
        return httpx.Response(404, json={})


def client(tmp_path, fake, enabled=True, cap=50.0, **kw):
    led = SpendLedger(tmp_path / "l.sqlite", {"lium": cap})
    c = LiumClient(
        "k",
        led,
        Registry(tmp_path / "reg.json"),
        enabled=enabled,
        transport=httpx.MockTransport(fake.handler),
        **kw,
    )
    c.http.sleep = lambda _s: None
    return c, led


def test_disabled_by_default_refuses(tmp_path):
    fake = FakeLium()
    c, _ = client(tmp_path, fake, enabled=False)
    with pytest.raises(GuardError, match="disabled"):
        c.rent(EXEC, purpose="t", ssh_public_key="ssh-ed25519 AAA")
    assert not any(m == "POST" for m, _ in fake.calls)


def test_rent_schedules_removal_and_uses_prefix(tmp_path):
    fake = FakeLium()
    c, led = client(tmp_path, fake)
    pod = c.rent(EXEC, purpose="Smoke Test", ssh_public_key="ssh-ed25519 AAA", max_hours=1)
    assert pod.name.startswith(POD_PREFIX)
    assert ("POST", f"/pods/{NEW_ID}/schedule-removal") in fake.calls
    assert led.outstanding("lium") == pytest.approx(0.5)  # projected reserved
    c.delete(pod.pod_id)
    assert led.spent("lium") == pytest.approx(0.12) and led.outstanding("lium") == 0


def test_never_touches_other_projects_pods(tmp_path):
    fake = FakeLium()
    c, _ = client(tmp_path, fake)
    with pytest.raises(GuardError, match="not created by this project"):
        c.delete(OTHER["id"])
    assert ("DELETE", f"/pods/{OTHER['id']}") not in fake.calls
    assert any(p["id"] == OTHER["id"] for p in fake.pods)


def test_registry_entry_with_foreign_name_refused(tmp_path):
    fake = FakeLium()
    c, _ = client(tmp_path, fake)
    pod = c.rent(EXEC, purpose="t", ssh_public_key="k")
    # someone renamed the pod (or registry got corrupted to point at a foreign pod)
    for p in fake.pods:
        if p["id"] == pod.pod_id:
            p["pod_name"] = "sn124-main"
    with pytest.raises(GuardError, match="lacks prefix"):
        c.delete(pod.pod_id)


def test_balance_floor_protects_other_pods(tmp_path):
    fake = FakeLium(balance=106.0, burn=2.2)  # floor = 2.2 * 48 = 105.6
    c, _ = client(tmp_path, fake)
    with pytest.raises(GuardError, match="balance floor"):
        c.rent(EXEC, purpose="t", ssh_public_key="k", max_hours=1)


def test_spend_cap(tmp_path):
    c, _ = client(tmp_path, FakeLium(), cap=0.4)
    with pytest.raises(GuardError, match="spend cap"):
        c.rent(EXEC, purpose="t", ssh_public_key="k", max_hours=1)


def test_price_and_hours_limits(tmp_path):
    c, _ = client(tmp_path, FakeLium(), max_price_per_hour=0.3)
    with pytest.raises(GuardError, match="exceeds max"):
        c.rent(EXEC, purpose="t", ssh_public_key="k")
    c2, _ = client(tmp_path / "b", FakeLium())
    with pytest.raises(GuardError, match="max_hours"):
        c2.rent(EXEC, purpose="t", ssh_public_key="k", max_hours=5)


def test_max_concurrent_own_pods(tmp_path):
    c, _ = client(tmp_path, FakeLium())
    c.rent(EXEC, purpose="a", ssh_public_key="k")
    with pytest.raises(GuardError, match="already 1 own pod"):
        c.rent(EXEC, purpose="b", ssh_public_key="k")


def test_failed_schedule_removal_deletes_pod(tmp_path):
    fake = FakeLium(schedule_fails=True)
    c, _ = client(tmp_path, fake)
    with pytest.raises(RuntimeError):
        c.rent(EXEC, purpose="t", ssh_public_key="k")
    assert ("DELETE", f"/pods/{NEW_ID}") in fake.calls
    assert all(p["id"] != NEW_ID for p in fake.pods)


def test_endpoint_allowlist_blocks_payments(tmp_path):
    c, _ = client(tmp_path, FakeLium())
    for method, path in [
        ("POST", "/payments/topup"),
        ("PUT", "/users/me/auto-top-up"),
        ("POST", "/pods/x/reboot"),
        ("DELETE", "/users/me"),
        ("POST", "/users/me/transfer-credits"),
    ]:
        with pytest.raises(GuardError, match="not allowed"):
            c._request(method, path)


def test_source_has_no_payment_endpoints():
    src = Path(__import__("humanizer_eval.runner.lium", fromlist=["x"]).__file__).read_text()
    # no string literal in the client may reference a money-moving endpoint
    assert not re.search(r"[\"']/(payments|stripe|tao/pay|users/me/(auto-top-up|transfer-credits))", src)
