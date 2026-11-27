"""Run one submission bundle on a fresh Lium pod and collect its outputs (D-60..D-62).

Pod-side flow (outer container, root, has internet):
  build the harness image -> extract the bundle -> run the miner code in an INNER
  container: --network none, no capabilities, no-new-privileges, read-only rootfs,
  unprivileged user, pids/memory limits, GPU passed through -> copy outputs back.
The pod is deleted in `finally`, whatever happens.
"""

from __future__ import annotations

import contextlib
import json
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from humanizer_eval.runner.bundle import MAX_BUNDLE_BYTES, inspect_bundle
from humanizer_eval.runner.lium import LiumClient

HARNESS_DIR = Path(__file__).resolve().parents[3] / "harness"
IMAGE = "humeval-harness:v1"
TEMPLATE_DIND = "cbc0ffdc-aa35-441a-abaa-170a87c51a64"  # Lium "Pytorch (Cuda + DinD)"
RESULT_CAP_BYTES = 64 * 1024**2  # outputs.jsonl fetched from the pod: 500 x 20k chars fits easily


def inner_docker_cmd(seed: int, time_limit_s: int, *, gpus: bool = True, memory: str = "48g") -> list[str]:
    """The hardened `docker run` for miner code. Kept as data so tests can assert every flag."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,exec,size=16g",
        "--pids-limit",
        "4096",
        "--memory",
        memory,
        "--user",
        "65534:65534",
        "-e",
        "HOME=/tmp",
        "-e",
        f"HUMEVAL_SEED={int(seed)}",
        "-e",
        f"HUMEVAL_TIME_LIMIT={int(time_limit_s)}",
        "-v",
        "/work/sub:/sub:ro",
        "-v",
        "/work/in:/in:ro",
        "-v",
        "/work/out:/out:rw",
    ]
    if gpus:
        cmd += ["--gpus", "all"]
    return [*cmd, IMAGE]


def pod_script(seed: int, time_limit_s: int, gpus: bool = True) -> str:
    inner = " ".join(shlex.quote(c) for c in inner_docker_cmd(seed, time_limit_s, gpus=gpus))
    return f"""set -e
(docker info >/dev/null 2>&1) || (nohup dockerd >/tmp/dockerd.log 2>&1 & sleep 8)
docker build -q -t {IMAGE} /work/harness >/work/build.log 2>&1
rm -rf /work/sub /work/out && mkdir -p /work/sub /work/out
tar -xf /work/bundle.tar -C /work/sub --no-same-owner --no-same-permissions \
  || {{ echo '{{"error": "extract"}}' > /work/bundle_error.json; exit 0; }}
chmod -R a+rX /work/sub /work/in && chown 65534:65534 /work/out
set +e
timeout {int(time_limit_s) + 300} {inner}
code=$?
info="{{\\"exit\\": $code, \\"kernel\\": \\"$(uname -r)\\", \\"docker\\": \\"$(docker version --format '{{{{.Server.Version}}}}')\\"}}"
echo "$info" > /work/pod.json
rm -f /work/out/pod.json && echo "$info" > /work/out/pod.json
# /work/out is writable by the miner's code: copy only regular files, size-capped, for the orchestrator
rm -rf /work/result && mkdir -p /work/result
for f in outputs.jsonl:{RESULT_CAP_BYTES} meta.json:1048576; do
  n=${{f%%:*}}; cap=${{f##*:}}
  if [ -f /work/out/$n ] && [ ! -L /work/out/$n ]; then head -c $cap /work/out/$n > /work/result/$n; fi
done
tail -c 1048576 /work/run.log > /work/result/run.log 2>/dev/null || true
"""


@dataclass
class PodSession:
    host: str
    port: str
    key: Path
    opts: list[str] = field(
        default_factory=lambda: [
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "LogLevel=ERROR",
        ]
    )

    def run(self, script: str, timeout: int) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["ssh", "-i", str(self.key), "-p", self.port, *self.opts, self.host, "bash -s"],
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def put(self, local: Path, remote: str, timeout: int = 1800) -> None:
        subprocess.run(
            [
                "scp",
                "-r",
                "-i",
                str(self.key),
                "-P",
                self.port,
                *self.opts,
                str(local),
                f"{self.host}:{remote}",
            ],
            check=True,
            capture_output=True,
            timeout=timeout,
        )

    def get(self, remote: str, local: Path, timeout: int = 600) -> None:
        subprocess.run(
            [
                "scp",
                "-r",
                "-i",
                str(self.key),
                "-P",
                self.port,
                *self.opts,
                f"{self.host}:{remote}",
                str(local),
            ],
            check=True,
            capture_output=True,
            timeout=timeout,
        )


def _ssh_target(pod: dict[str, Any]) -> tuple[str, str]:
    parts = shlex.split(pod.get("ssh_connect_cmd") or "")
    host = next((p for p in parts if "@" in p), None)
    port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
    if not host:
        raise RuntimeError("cannot parse ssh_connect_cmd")
    return host, port


BUNDLE_PY = Path(__file__).resolve().parent / "bundle.py"  # stdlib only: runs on the pod as a script
POLL_SECONDS = 30
SETUP_SECONDS = 1800  # dockerd, image build, bundle download, upload of the inputs


def fetch_script(url: str, sha256: str, max_bytes: int = MAX_BUNDLE_BYTES) -> str:
    """Pod side: download a remote bundle, check its sha256 and headers before anything is extracted.

    A miner-side problem (unreachable URL, wrong hash, invalid bundle) writes /work/bundle_error.json and
    ends the job without running anything; the orchestrator reports it to the miner as a failed bundle.
    """
    if not (
        url.startswith("https://") and len(sha256) == 64 and all(c in "0123456789abcdef" for c in sha256)
    ):
        raise ValueError("bundle_url must be https and bundle_sha256 a lowercase hex sha256")
    return f"""set -u
fail() {{ printf '{{"error": "%s"}}\\n' "$1" > /work/bundle_error.json; exit 0; }}
curl -fsSL --proto =https --proto-redir =https --retry 3 --connect-timeout 30 --max-filesize {int(max_bytes)} \\
  -o /work/bundle.tar \\
  {shlex.quote(url)} > /work/fetch.log 2>&1 || fail download
echo "{sha256}  /work/bundle.tar" | sha256sum -c --status - || fail sha256
python3 /work/inspect_bundle.py /work/bundle.tar > /work/bundle_info.json 2>/work/inspect.log || fail invalid
"""


@dataclass
class PodJobResult:
    """What came back from the pod. ``status``: ``ok`` (harness ran; outputs may still be partial),
    ``miner_error`` (the bundle could not run: bad download/bundle, killed container) or ``infra_error``
    (our side: no pod, setup failed, lost the pod)."""

    status: str
    reason: str
    outputs: dict[str, str]
    meta: dict[str, Any]
    pod: dict[str, Any]
    cost_usd: float | None
    seconds: float
    log_dir: Path


def _read_json(p: Path) -> dict[str, Any]:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def run_pod_job(
    items: list[dict[str, Any]],
    *,
    client: LiumClient,
    executor: dict[str, Any],
    ssh_key: Path,
    workdir: Path,
    purpose: str,
    bundle: Path | None = None,
    bundle_url: str | None = None,
    bundle_sha256: str | None = None,
    seed: int = 0,
    time_limit_s: int = 3600,
    max_hours: int = 2,
    gpus: bool = True,
    poll_seconds: int = POLL_SECONDS,
) -> PodJobResult:
    """Rent a fresh pod, run one bundle in the hardened harness, fetch outputs, delete the pod.

    The job runs under ``nohup`` on the pod and is polled, so a dropped ssh connection does not kill a
    one-hour run. Exactly one of ``bundle`` (uploaded with scp) and ``bundle_url`` (downloaded by the pod,
    checked against ``bundle_sha256``) is given. The pod is deleted in ``finally``.
    """
    if (bundle is None) == (bundle_url is None):
        raise ValueError("give exactly one of bundle, bundle_url")
    fetch = fetch_script(bundle_url, bundle_sha256 or "") if bundle_url else ""
    if bundle is not None:
        inspect_bundle(bundle)  # never extract on the orchestrator
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "in").mkdir(exist_ok=True)
    (workdir / "in" / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))
    for stale in ("result", "pod.json", "bundle_error.json", "build.log"):
        subprocess.run(["rm", "-rf", str(workdir / stale)], check=True)
    pub = Path(str(ssh_key) + ".pub").read_text().strip()
    t0 = time.time()
    try:
        pod = client.rent(executor, purpose=purpose, ssh_public_key=pub, max_hours=max_hours,
                          template_id=TEMPLATE_DIND)  # fmt: skip
    except Exception as e:
        return PodJobResult("infra_error", f"rent failed: {type(e).__name__}: {e}", {}, {}, {}, None,
                            round(time.time() - t0, 1), workdir)  # fmt: skip
    status, reason, outputs, meta, podinfo = "infra_error", "", {}, {}, {}
    try:
        live = None
        deadline = time.time() + 900
        while time.time() < deadline:
            live = next((p for p in client.pods() if p["id"] == pod.pod_id), None)
            if live and live.get("status") == "RUNNING" and live.get("ssh_connect_cmd"):
                break
            time.sleep(10)
        else:
            raise RuntimeError("pod not ready in 15 min")
        sess = PodSession(*_ssh_target(live), ssh_key)
        time.sleep(10)
        sess.run("mkdir -p /work", timeout=120)
        sess.put(HARNESS_DIR, "/work/harness")
        sess.put(workdir / "in", "/work/in")
        if bundle is not None:
            sess.put(bundle, "/work/bundle.tar", timeout=7200)
        else:
            sess.put(BUNDLE_PY, "/work/inspect_bundle.py")
        script = fetch + pod_script(seed, time_limit_s, gpus)
        launch = (f"cat > /work/run.sh <<'EOF_JOB'\n{script}\nEOF_JOB\n"
                  "nohup bash -c 'bash /work/run.sh > /work/run.log 2>&1; echo $? > /work/DONE' "
                  ">/dev/null 2>&1 &")  # fmt: skip
        sess.run(launch, timeout=120)
        done, ssh_failures = None, 0
        job_deadline = time.time() + time_limit_s + SETUP_SECONDS
        while time.time() < job_deadline:
            time.sleep(poll_seconds)
            try:
                r = sess.run("cat /work/DONE 2>/dev/null || true", timeout=120)
                ssh_failures = 0
            except subprocess.TimeoutExpired:
                ssh_failures += 1
                if ssh_failures >= 10:
                    raise RuntimeError("lost the pod (ssh failed 10 times in a row)") from None
                continue
            if r.stdout.strip().isdigit():
                done = int(r.stdout.strip())
                break
        for remote in ("/work/result", "/work/pod.json", "/work/bundle_error.json", "/work/build.log"):
            # absent files are part of the classification below
            with contextlib.suppress(subprocess.CalledProcessError, subprocess.TimeoutExpired):
                sess.get(remote, workdir)
        out_dir = workdir / "result"
        berr = _read_json(workdir / "bundle_error.json")
        meta = _read_json(out_dir / "meta.json")
        podinfo = _read_json(workdir / "pod.json")
        if (out_dir / "outputs.jsonl").exists():
            for line in (out_dir / "outputs.jsonl").read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if (
                    isinstance(row, dict)
                    and isinstance(row.get("id"), str)
                    and isinstance(row.get("text"), str)
                ):
                    outputs[row["id"]] = row["text"][:20000]
        if done is None:
            status, reason = "infra_error", "job did not finish in time"
        elif berr:
            status, reason = "miner_error", f"bundle: {berr.get('error', 'invalid')}"
        elif meta:
            status = "miner_error" if meta.get("errors") and not outputs else "ok"
            reason = "entrypoint failed" if meta.get("errors") else ""
        elif podinfo:
            status, reason = "miner_error", f"container ended without results (exit {podinfo.get('exit')})"
        else:
            status, reason = "infra_error", f"setup failed (job exit {done})"
    except Exception as e:
        status, reason = "infra_error", f"{type(e).__name__}: {e}"
    finally:
        own = client.delete(pod.pod_id)
    return PodJobResult(status, reason, outputs, meta, podinfo, own.actual_usd, round(time.time() - t0, 1),
                        workdir)  # fmt: skip


@dataclass
class RunResult:
    outputs: dict[str, str]
    meta: dict[str, Any]
    pod: dict[str, Any]
    cost_usd: float | None
    seconds: float


def run_on_lium(
    bundle: Path,
    items: list[dict[str, Any]],
    *,
    client: LiumClient,
    executor: dict[str, Any],
    ssh_key: Path,
    workdir: Path,
    seed: int = 0,
    time_limit_s: int = 3600,
    max_hours: int = 1,
    gpus: bool = True,
) -> RunResult:
    """Local bundle on a fresh pod (kept for the e2e scripts); raises on anything but a clean run."""
    info = inspect_bundle(bundle)
    r = run_pod_job(items, client=client, executor=executor, ssh_key=ssh_key, workdir=workdir,
                    purpose=f"eval-{info.manifest.name}", bundle=bundle, seed=seed, time_limit_s=time_limit_s,
                    max_hours=max_hours, gpus=gpus)  # fmt: skip
    if r.status != "ok":
        raise RuntimeError(f"{r.status}: {r.reason}")
    return RunResult(r.outputs, r.meta, r.pod, r.cost_usd, r.seconds)
