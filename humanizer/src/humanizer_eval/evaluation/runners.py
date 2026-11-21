"""Ways to get a submission's outputs on the period inputs.

* ``LiumRunner`` -- production: a fresh Lium pod per submission, hardened inner container (D-60..D-62).
* ``ReplayRunner`` -- outputs saved earlier (pilot submissions, re-scoring after a pause); no pod.
* ``LocalHarnessRunner`` -- tests and development only: runs ``harness/run.py`` on this machine with NO
  isolation, so only for bundles we wrote ourselves.

Status of a run: ``ok`` (outputs, possibly partial), ``miner_error`` (the bundle could not run; the miner
pays for the pod time used) or ``infra_error`` (our side failed; the hold is released).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from humanizer_eval.runner.bundle import BundleError, inspect_bundle

HARNESS_RUN = Path(__file__).resolve().parents[3] / "harness" / "run.py"


@dataclass
class Submission:
    submission_id: str
    hotkey: str
    bundle: Path | None = None  # local .tar (inspected here, uploaded to the pod)
    bundle_url: str | None = None  # https URL the pod downloads ...
    bundle_sha256: str | None = None  # ... and checks against this hash
    name: str = ""
    reg_block: int | None = None  # registration block of the hotkey (kept on the king entry, D-66)

    def __post_init__(self) -> None:
        if not self.submission_id or any(c in self.submission_id for c in "/\\. "):
            raise ValueError("submission_id must be a plain token")
        if self.bundle_url and not self.bundle_sha256:
            raise ValueError("bundle_url needs bundle_sha256")


@dataclass
class RunOutput:
    status: str  # "ok" | "miner_error" | "infra_error"
    outputs: dict[str, str]
    pod_seconds: float = 0.0
    reason: str = ""  # short, our wording only (never text produced by the miner's code)
    meta: dict[str, Any] = field(default_factory=dict)  # harness meta: private, may hold miner text
    cost_usd: float | None = None  # what the pod cost the owner


class Runner(Protocol):
    def run(self, sub: Submission, items: list[dict[str, Any]], seed: int, workdir: Path) -> RunOutput: ...


class ReplayRunner:
    """Outputs produced earlier: a mapping id -> text or a list aligned with the items."""

    def __init__(self, outputs: Mapping[str, str] | Sequence[str], pod_seconds: float = 0.0):
        self.outputs = outputs
        self.pod_seconds = pod_seconds

    def run(self, sub: Submission, items: list[dict[str, Any]], seed: int, workdir: Path) -> RunOutput:
        if isinstance(self.outputs, Mapping):
            outs = {it["id"]: self.outputs[it["id"]] for it in items if it["id"] in self.outputs}
        else:
            if len(self.outputs) != len(items):
                raise ValueError("replayed outputs are not aligned with the period items")
            outs = {it["id"]: o for it, o in zip(items, self.outputs, strict=True) if isinstance(o, str)}
        return RunOutput("ok", outs, self.pod_seconds, meta={"replay": True})


class LocalHarnessRunner:
    """Development only: ``harness/run.py`` on this host, the bundle given as an unpacked directory."""

    def __init__(self, bundle_dir: Path, time_limit_s: int = 120):
        self.bundle_dir = Path(bundle_dir)
        self.time_limit_s = time_limit_s

    def run(self, sub: Submission, items: list[dict[str, Any]], seed: int, workdir: Path) -> RunOutput:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            inp, out = Path(tmp) / "in", Path(tmp) / "out"
            inp.mkdir()
            (inp / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))
            env = {
                **os.environ,
                "HUMEVAL_SUB": str(self.bundle_dir),
                "HUMEVAL_IN": str(inp),
                "HUMEVAL_OUT": str(out),
                "HUMEVAL_SEED": str(seed),
                "HUMEVAL_TIME_LIMIT": str(self.time_limit_s),
                "HUMEVAL_SITE": str(Path(tmp) / "site"),
            }
            subprocess.run(
                [sys.executable, str(HARNESS_RUN)],
                env=env,
                check=True,
                timeout=self.time_limit_s + 60,
                capture_output=True,
            )
            meta = json.loads((out / "meta.json").read_text())
            outs = {}
            for line in (out / "outputs.jsonl").read_text().splitlines():
                row = json.loads(line)
                outs[row["id"]] = row["text"]
        failed = bool(meta.get("errors"))
        status = "miner_error" if failed and not any(outs.values()) else "ok"
        return RunOutput(
            status, outs, round(time.time() - t0, 2), "entrypoint failed" if failed else "", meta
        )


def _usable(e: dict[str, Any], max_price: float, min_reliability: float) -> bool:
    return (
        e.get("tier") == "secure"
        and float(e.get("price_per_gpu", 1e9)) <= max_price
        and (e.get("reliability_score") or 0) >= min_reliability
        and e.get("available_gpu_count", 0) >= 1
        and not e.get("frequent_rent_failures")
        and e.get("disk_health_ok", True) is not False
    )


def rank_executors(
    executors: list[dict[str, Any]],
    max_price: float,
    gpus: tuple[str, ...] = ("H100",),
    min_reliability: float = 90,
) -> list[dict[str, Any]]:
    """Usable executors whose name has one of ``gpus``: single-GPU nodes first (a multi-GPU node may rent
    whole only), then by price."""
    ok = [
        e
        for e in executors
        if any(g in e.get("machine_name", "") for g in gpus) and _usable(e, max_price, min_reliability)
    ]
    return sorted(ok, key=lambda e: (e.get("gpu_count", 1) != 1, float(e["price_per_gpu"])))


class LiumRunner:
    """One fresh pod per submission; the pod never holds keys or secrets (D-62).

    A refused rent (node taken, node rents whole only) is retried on the next ranked executor, up to
    ``rent_attempts``; any failure after the pod exists is final for this attempt.
    """

    def __init__(
        self,
        client: Any,  # runner.lium.LiumClient
        ssh_key: Path,
        *,
        max_price: float = 2.6,
        time_limit_s: int = 3600,
        max_hours: int = 2,
        rank: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
        rent_attempts: int = 3,
        gpus: bool = True,
    ):
        self.client = client
        self.ssh_key = Path(ssh_key)
        self.max_price = max_price
        self.time_limit_s = time_limit_s
        self.max_hours = max_hours
        self.rank = rank or (lambda ex: rank_executors(ex, self.max_price))
        self.rent_attempts = rent_attempts
        self.gpus = gpus

    def run(self, sub: Submission, items: list[dict[str, Any]], seed: int, workdir: Path) -> RunOutput:
        from humanizer_eval.runner.evaluate import run_pod_job

        if sub.bundle is not None:
            try:
                inspect_bundle(sub.bundle)
            except (BundleError, OSError) as e:
                return RunOutput("miner_error", {}, 0.0, f"invalid bundle: {e}")
        try:
            ranked = self.rank(self.client.executors())
        except Exception as e:
            return RunOutput("infra_error", {}, 0.0, f"executor list failed: {type(e).__name__}")
        if not ranked:
            return RunOutput("infra_error", {}, 0.0, "no suitable executor")
        tried: list[dict[str, Any]] = []
        for ex in ranked[: self.rent_attempts]:
            r = run_pod_job(
                items,
                client=self.client,
                executor=ex,
                ssh_key=self.ssh_key,
                workdir=workdir,
                purpose=f"eval-{sub.submission_id}"[:20],
                bundle=sub.bundle,
                bundle_url=sub.bundle_url,
                bundle_sha256=sub.bundle_sha256,
                seed=seed,
                time_limit_s=self.time_limit_s,
                max_hours=self.max_hours,
                gpus=self.gpus,
            )
            tried.append({"executor": ex.get("id"), "status": r.status, "reason": r.reason[:300]})
            if not (r.status == "infra_error" and r.reason.startswith("rent failed")):
                break
        meta = {
            **r.meta,
            "executor": ex.get("machine_name"),
            "price_per_hour": ex.get("price_per_gpu"),
            "pod": r.pod,
            "attempts": tried,
        }
        # the miner pays the pod time of the run that happened, not our refused rents
        return RunOutput(r.status, r.outputs, r.seconds, r.reason, meta, r.cost_usd)
