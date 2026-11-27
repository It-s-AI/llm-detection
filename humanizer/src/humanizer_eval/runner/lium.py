"""Guarded Lium (SN51) client for one-off evaluation pods (D-62, D-65).

The Lium account is SHARED with other projects that have running pods. This client
therefore enforces, in code, before any request is sent:

* an endpoint allow-list (no payments / top-ups / account mutation is even expressible);
* destructive calls (delete, schedule-removal) ONLY on pods that this client created
  (recorded in the local registry) AND whose live name carries our prefix;
* a kill switch (`enabled`), max concurrent own pods, max price per hour;
* a hard spend cap via the spend ledger (projected cost reserved at rent time);
* a balance floor: renting must leave the other pods at least `floor_hours` of runway;
* server-side auto-termination (`termination_hours`) plus an explicit scheduled
  removal; if scheduling fails the pod is deleted immediately.
"""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from humanizer_eval.budget import SpendLedger
from humanizer_eval.detectors.http import HttpClient, RetryPolicy

API = "https://lium.io/api"
POD_PREFIX = "humeval-"

# (method, path regex) pairs this client may ever call.
_ALLOWED: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GET", re.compile(r"^/users/me$")),
    ("GET", re.compile(r"^/pods$")),
    ("GET", re.compile(r"^/pods/[0-9a-f-]{36}$")),
    ("GET", re.compile(r"^/pods/[0-9a-f-]{36}/statement$")),
    ("GET", re.compile(r"^/executors$")),
    ("GET", re.compile(r"^/templates$")),
    ("POST", re.compile(r"^/executors/[0-9a-f-]{36}/rent$")),
    ("POST", re.compile(r"^/pods/[0-9a-f-]{36}/schedule-removal$")),
    ("DELETE", re.compile(r"^/pods/[0-9a-f-]{36}$")),
)


class GuardError(RuntimeError):
    """A safety rule refused the operation before any request was sent."""


@dataclass
class OwnPod:
    pod_id: str
    name: str
    executor_id: str
    price_per_hour: float
    max_hours: int
    created_at: float
    reservation_id: str
    projected_usd: float
    removed: bool = False
    actual_usd: float | None = None


class Registry:
    """Local record of pods created by THIS project (the only pods we may touch)."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, OwnPod]:
        if not self.path.exists():
            return {}
        return {k: OwnPod(**v) for k, v in json.loads(self.path.read_text()).items()}

    def save(self, pods: dict[str, OwnPod]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: asdict(v) for k, v in pods.items()}, indent=1))
        tmp.replace(self.path)


class LiumClient:
    service = "lium"

    def __init__(
        self,
        api_key: str,
        ledger: SpendLedger,
        registry: Registry,
        *,
        enabled: bool,
        floor_hours: float = 48.0,
        max_price_per_hour: float = 4.0,
        max_concurrent: int = 1,
        max_hours: int = 2,
        transport: httpx.BaseTransport | None = None,
    ):
        self.ledger = ledger
        self.registry = registry
        self.enabled = enabled
        self.floor_hours = floor_hours
        self.max_price_per_hour = max_price_per_hour
        self.max_concurrent = max_concurrent
        self._rent_lock = threading.Lock()  # parallel evaluations: the limit check and the rent are one step
        self.max_hours = max_hours
        self.http = HttpClient(
            base_url=API,
            headers={"X-API-Key": api_key},
            rate_per_sec=2,
            burst=4,
            transport=transport,
            retry=RetryPolicy(max_attempts=3),
        )

    # --- transport with allow-list -------------------------------------------
    def _request(self, method: str, path: str, **kw) -> Any:
        if not any(m == method and rx.match(path) for m, rx in _ALLOWED):
            raise GuardError(f"endpoint not allowed: {method} {path}")
        resp = self.http.request(method, path, **kw)
        if resp.status_code >= 400:
            raise RuntimeError(f"lium {method} {path}: HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json() if resp.content else None

    # --- read-only -----------------------------------------------------------
    def me(self) -> dict[str, Any]:
        return self._request("GET", "/users/me")

    def pods(self) -> list[dict[str, Any]]:
        return self._request("GET", "/pods")

    def executors(self, **filters) -> list[dict[str, Any]]:
        out = self._request("GET", "/executors", params={"size": 500, **filters})
        return out if isinstance(out, list) else out.get("items", [])

    def statement(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}/statement")

    # --- guards ---------------------------------------------------------------
    def _own_active(self) -> dict[str, OwnPod]:
        return {k: v for k, v in self.registry.load().items() if not v.removed}

    def _assert_own(self, pod_id: str) -> OwnPod:
        own = self.registry.load().get(pod_id)
        if own is None:
            raise GuardError(f"pod {pod_id} was not created by this project; refusing to touch it")
        live = next((p for p in self.pods() if p.get("id") == pod_id), None)
        if live is not None and not str(live.get("pod_name", "")).startswith(POD_PREFIX):
            raise GuardError(f"pod {pod_id} name {live.get('pod_name')!r} lacks prefix {POD_PREFIX!r}")
        return own

    def check_can_rent(self, price_per_hour: float, max_hours: int) -> dict[str, float]:
        if not self.enabled:
            raise GuardError("Lium renting disabled (LIUM_ENABLED=false)")
        if price_per_hour > self.max_price_per_hour:
            raise GuardError(f"price ${price_per_hour}/h exceeds max ${self.max_price_per_hour}/h")
        if max_hours > self.max_hours or max_hours < 1:
            raise GuardError(f"max_hours {max_hours} outside 1..{self.max_hours}")
        if len(self._own_active()) >= self.max_concurrent:
            raise GuardError(
                f"already {len(self._own_active())} own pod(s) active (max {self.max_concurrent})"
            )
        projected = price_per_hour * max_hours
        me = self.me()
        balance = float(me.get("balance") or 0.0)
        burn = float(me.get("burn_per_hour") or 0.0)
        own_burn = sum(p.price_per_hour for p in self._own_active().values())
        other_burn = max(0.0, burn - own_burn)
        floor = other_burn * self.floor_hours
        if balance - projected < floor:
            raise GuardError(
                f"balance floor: balance ${balance:.2f} - projected ${projected:.2f} < floor ${floor:.2f}"
                f" ({other_burn:.2f}/h other pods x {self.floor_hours}h)"
            )
        if self.ledger.remaining(self.service) < projected:
            raise GuardError(
                f"spend cap: remaining ${self.ledger.remaining(self.service):.2f} < ${projected:.2f}"
            )
        return {"balance": balance, "other_burn": other_burn, "floor": floor, "projected": projected}

    # --- mutations ------------------------------------------------------------
    def rent(
        self,
        executor: dict[str, Any],
        *,
        purpose: str,
        ssh_public_key: str,
        max_hours: int = 1,
        gpu_count: int = 1,
        template_id: str | None = None,
    ) -> OwnPod:
        with self._rent_lock:
            return self._rent(executor, purpose=purpose, ssh_public_key=ssh_public_key, max_hours=max_hours,
                              gpu_count=gpu_count, template_id=template_id)  # fmt: skip

    def _rent(
        self,
        executor: dict[str, Any],
        *,
        purpose: str,
        ssh_public_key: str,
        max_hours: int,
        gpu_count: int,
        template_id: str | None,
    ) -> OwnPod:
        price = float(executor["price_per_gpu"]) * gpu_count
        self.check_can_rent(price, max_hours)
        name = f"{POD_PREFIX}{re.sub(r'[^a-z0-9-]', '-', purpose.lower())[:20]}-{secrets.token_hex(3)}"
        res = self.ledger.reserve(self.service, price * max_hours, ref=f"lium:{name}")
        body = {
            "pod_name": name,
            "user_public_key": [ssh_public_key],
            "gpu_count": gpu_count,
            "termination_hours": max_hours,
            "expected_price_per_gpu_hour": float(executor["price_per_gpu"]),
        }
        if template_id:
            body["template_id"] = template_id
        try:
            out = self._request(
                "POST", f"/executors/{executor['id']}/rent", json=body, headers={"Idempotency-Key": name}
            )
        except Exception:
            res.release({"note": "rent failed"})
            raise
        pod_id = (out or {}).get("id") or (out or {}).get("pod_id")
        if not pod_id:  # find it by our unique name
            pod_id = next((p["id"] for p in self.pods() if p.get("pod_name") == name), None)
        if not pod_id:
            res.commit(price * max_hours, {"note": "rent returned no id; assume billed"})
            raise RuntimeError(f"rented {name} but could not find its id; check Lium dashboard")
        pod = OwnPod(pod_id, name, executor["id"], price, max_hours, time.time(), res.id, price * max_hours)
        pods = self.registry.load()
        pods[pod_id] = pod
        self.registry.save(pods)
        try:
            when = (datetime.now(UTC) + timedelta(hours=max_hours)).strftime("%Y-%m-%dT%H:%M:%S")
            self._request("POST", f"/pods/{pod_id}/schedule-removal", json={"removal_scheduled_at": when})
        except Exception:
            self.delete(pod_id)
            raise
        return pod

    def delete(self, pod_id: str) -> OwnPod:
        own = self._assert_own(pod_id)
        if not own.removed:
            try:
                self._request("DELETE", f"/pods/{pod_id}")
            except RuntimeError as e:
                if "404" not in str(e):
                    raise
            own.removed = True
        try:
            own.actual_usd = float(self.statement(pod_id).get("total"))
        except Exception:
            own.actual_usd = None
        pods = self.registry.load()
        pods[pod_id] = own
        self.registry.save(pods)
        self._settle(own)
        return own

    def _settle(self, own: OwnPod) -> None:
        """Commit the rent reservation with the actual billed total (or projection)."""
        from humanizer_eval.budget import Reservation

        res = Reservation(self.ledger, own.reservation_id, self.service, own.projected_usd)
        res.commit(
            own.actual_usd if own.actual_usd is not None else own.projected_usd,
            {"pod": own.name, "actual_known": own.actual_usd is not None},
        )
