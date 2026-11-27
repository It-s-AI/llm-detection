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

import contextlib
import json
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from humanizer_eval.budget import SpendLedger
from humanizer_eval.detectors.http import HttpClient, RetryPolicy

API = "https://lium.io/api"
POD_PREFIX = "humeval-"
PENDING = "pending:"  # registry key of a rent whose answer was lost (no pod id known yet)

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


class RentRefused(RuntimeError):
    """Lium refused the rent outright (a clear 4xx): no pod was created, nothing was billed."""


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
    delete_failed: bool = False  # its run is over but the delete failed: reconcile() deletes it


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
        max_price_per_hour: float = 4.5,
        max_concurrent: int = 1,
        max_hours: int = 2,
        transport: httpx.BaseTransport | None = None,
        auto_topup: bool = False,
    ):
        self.ledger = ledger
        self.registry = registry
        self.enabled = enabled
        self.floor_hours = floor_hours
        self.max_price_per_hour = max_price_per_hour
        self.max_concurrent = max_concurrent
        self._rent_lock = threading.RLock()  # the check and the rent are one step; delete() re-enters it
        self.max_hours = max_hours
        self.auto_topup = auto_topup
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

    def sweep_orphans(self) -> list[str]:
        """Delete every own pod still active in the registry, and every live pod named with our prefix the
        registry lacks (a rent whose answer was lost). Only for the start of the one process that rents (the
        intake server): after a restart no evaluation owns them any more."""
        gone = []
        with contextlib.suppress(Exception):
            self._adopt_pending(self.pods())
        with contextlib.suppress(Exception):
            # the reservation of each rent, by its ref 'lium:<pod name>': a pod the registry lacks is settled
            # with it; one with neither a registry record nor a live pod (killed before anything was rented)
            # is released, or it would count against the cap for good
            open_res = {ref[len("lium:") :]: (rid, est) for rid, ref, est in self.ledger.open_reservations(
                self.service) if ref.startswith("lium:")}  # fmt: skip
            listed = self.pods()
            live_names = {p.get("pod_name") or "" for p in listed}
            with self._rent_lock:
                pods = self.registry.load()
                known_ids = set(pods)
                known_names = {v.name for v in pods.values() if not v.removed}
                for p in listed:
                    pid, nm = p.get("id"), p.get("pod_name") or ""
                    if pid and nm.startswith(POD_PREFIX) and pid not in known_ids:
                        rid, est = open_res.get(nm, ("", 0.0))
                        pods[pid] = OwnPod(
                            pid, nm, str(p.get("executor_id") or ""), 0.0, 0, time.time(), rid, est
                        )
                self.registry.save(pods)
            from humanizer_eval.budget import Reservation

            for nm, (rid, est) in open_res.items():
                if nm not in live_names and nm not in known_names:
                    Reservation(self.ledger, rid, self.service, est).release(
                        {"note": "no pod: rent never made"}
                    )
        for pod_id in list(self._own_active()):
            if pod_id.startswith(PENDING):  # no pod known: settled by _adopt_pending
                continue
            try:
                self.delete(pod_id)
                gone.append(pod_id)
            except Exception:  # noqa: S112 - Lium removes it at its scheduled time anyway
                continue
        return gone

    def mark_delete_failed(self, pod_id: str) -> None:
        with self._rent_lock:
            pods = self.registry.load()
            if pod_id in pods:
                pods[pod_id].delete_failed = True
                self.registry.save(pods)

    def reconcile(self) -> list[str]:
        """Own pods our registry still counts as active but Lium no longer lists (a delete that failed, then
        the scheduled removal): marked removed and their reservation settled. Returns their ids."""
        active = self._own_active()
        if not active:
            return []
        listed = self.pods()
        self._adopt_pending(listed)
        for k, v in list(self._own_active().items()):  # a finished run's pod whose delete failed
            if v.delete_failed:
                with contextlib.suppress(Exception):
                    self.delete(k)
        active = self._own_active()
        live = {p.get("id") for p in listed}
        gone = [k for k in active if k not in live and not k.startswith(PENDING)]
        if gone:
            pods = self.registry.load()
            for k in gone:
                own = pods[k]
                own.removed = True
                with contextlib.suppress(Exception):
                    own.actual_usd = float(self.statement(k).get("total"))
                pods[k] = own
                self._settle(own)
            self.registry.save(pods)
        return gone

    def _set_pending(self, name: str, pod: OwnPod | None) -> None:
        with self._rent_lock:
            pods = self.registry.load()
            if pod is None:
                pods.pop(PENDING + name, None)
            else:
                pods[PENDING + name] = pod
            self.registry.save(pods)

    def _adopt_pending(self, listed: list[dict[str, Any]]) -> None:
        """Rents whose answer was lost: a pod that showed up under the name is adopted (its reservation
        kept) and deleted, it belonged to a failed attempt; one that never showed up within 15 minutes
        releases its reservation."""
        from humanizer_eval.budget import Reservation

        by_name = {p.get("pod_name"): p for p in listed}
        adopted = []
        with self._rent_lock:
            pods = self.registry.load()
            for k, v in list(pods.items()):
                if not k.startswith(PENDING) or v.removed:
                    continue
                live = by_name.get(v.name)
                if live is not None and live.get("id"):
                    pods[live["id"]] = replace(v, pod_id=live["id"])
                    v.removed = True
                    adopted.append(live["id"])
                elif time.time() - v.created_at > 900:
                    v.removed = True
                    Reservation(self.ledger, v.reservation_id, self.service, v.projected_usd).release(
                        {"note": "lost rent: no pod appeared"}
                    )
            self.registry.save(pods)
        for pid in adopted:
            with contextlib.suppress(Exception):
                self.delete(pid)

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
        self.reconcile()
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
        # with auto top-up the account refills as it is spent: a positive balance is enough
        needed = 0.0 if self.auto_topup else projected
        if balance <= 0 or balance - needed < floor:
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
        try:
            self.check_can_rent(price, max_hours)
        except GuardError:
            raise
        except Exception as e:  # the pre-rent check (balance, pods) failed: nothing was rented
            raise RentRefused(f"pre-rent check failed: {type(e).__name__}: {e}") from e
        name = f"{POD_PREFIX}{re.sub(r'[^a-z0-9-]', '-', purpose.lower())[:20]}-{secrets.token_hex(3)}"
        res = self.ledger.reserve(self.service, price * max_hours, ref=f"lium:{name}")
        # recorded before the request: a kill anywhere below leaves a record that _adopt_pending resolves
        # (the pod found by its name is deleted with this reservation; none within 15 min: released)
        self._set_pending(name, OwnPod(PENDING + name, name, executor["id"], price, max_hours, time.time(),
                                       res.id, price * max_hours))  # fmt: skip
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
        except Exception as e:
            # a clear refusal (the node is taken, rents whole only, ...) created nothing; any other failure
            # (timeout, 5xx, 409 on a retry) may have created the pod: look for it by its unique name
            refused = isinstance(e, GuardError) or bool(re.search(r"HTTP 4(00|01|02|03|04|22)\b", str(e)))
            if refused:
                res.release({"note": "rent refused"})
                self._set_pending(name, None)
                if isinstance(e, GuardError):
                    raise
                raise RentRefused(str(e)) from e
            found, listed = None, False
            for _ in range(2):  # the pod may show up a moment after the failed answer
                try:
                    found = next((p["id"] for p in self.pods() if p.get("pod_name") == name), None)
                    listed = True
                except Exception:  # noqa: S112 - the list itself failed: tried once more
                    continue
                if found is not None:
                    break
                time.sleep(5)
            if found is None and listed:  # not listed (yet): stays a pending rent, its reservation open
                raise
            if found is None:  # unknown: count it as billed (the cap keeps counting it); the start sweeps
                res.commit(price * max_hours, {"note": "rent answer lost; assume billed"})
                raise
            pod = OwnPod(
                found, name, executor["id"], price, max_hours, time.time(), res.id, price * max_hours
            )
            with self._rent_lock:
                pods = self.registry.load()
                pods[found] = pod
                pods.pop(PENDING + name, None)
                self.registry.save(pods)
            with contextlib.suppress(Exception):
                self.delete(found)  # this attempt failed: never leave it running
            raise
        pod_id = (out or {}).get("id") or (out or {}).get("pod_id")
        if not pod_id:  # find it by our unique name
            pod_id = next((p["id"] for p in self.pods() if p.get("pod_name") == name), None)
        if not pod_id:
            res.commit(price * max_hours, {"note": "rent returned no id; assume billed"})
            raise RuntimeError(f"rented {name} but could not find its id; check Lium dashboard")
        pod = OwnPod(pod_id, name, executor["id"], price, max_hours, time.time(), res.id, price * max_hours)
        with self._rent_lock:
            pods = self.registry.load()
            pods[pod_id] = pod
            pods.pop(PENDING + name, None)
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
                    with contextlib.suppress(Exception):
                        self.mark_delete_failed(pod_id)  # reconcile() (before every rent) deletes it again
                    raise
            own.removed = True
        try:
            own.actual_usd = float(self.statement(pod_id).get("total"))
        except Exception:
            own.actual_usd = None
        with self._rent_lock:  # the registry is read-modified-written: never interleaved with a rent's save
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
