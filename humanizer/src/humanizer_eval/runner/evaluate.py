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
import logging
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from humanizer_eval.runner.bundle import MAX_BUNDLE_BYTES, inspect_bundle
from humanizer_eval.runner.lium import LiumClient

log = logging.getLogger(__name__)

HARNESS_DIR = Path(__file__).resolve().parents[3] / "harness"
IMAGE = "humeval-harness:v1"
#: the harness's base image (harness/Dockerfile FROM). A pod never pulls it from a registry: on 2026-11-26
#: Lium nodes got Docker Hub (and mirror.gcr.io) at ~90 KB/s, so a build never finished. We host the
#: `docker save` of it on our HF repo (pinned commit, checked by sha256); pods fetch HF at ~70 MB/s.
BASE_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
BASE_IMAGE_URL = (
    "https://huggingface.co/sergak0/sn32-kings/resolve/3da9552a07fad8ded4f8f851f1a1abb1eb27dea6/"
    "harness/pytorch-2.7.1-cuda12.8-cudnn9-runtime.tar"
)
BASE_IMAGE_SHA256 = "2832d13baa70c167e7cb54c4994d71e0f0ca08e6e6d24bf01b2662b1018ac872"
TEMPLATE_DIND = "cbc0ffdc-aa35-441a-abaa-170a87c51a64"  # Lium "Pytorch (Cuda + DinD)"
RESULT_CAP_BYTES = 64 * 1024**2  # outputs.jsonl fetched from the pod: 500 x 20k chars fits easily
JOB_CONTAINER = "humeval-job"
#: the harness stops the miner's code at the time limit (SIGALRM, inside its own process, so the miner can
#: switch it off); the pod KILLS the container this long after the limit, from outside, whatever it does
KILL_GRACE_S = 120
#: a run longer than its limit by more than this (container start included) is scored with the outputs it
#: had written at limit + this, not with what it wrote after switching the alarm off
OVER_LIMIT_SLACK_S = 30


def inner_docker_cmd(seed: int, time_limit_s: int, *, gpus: bool = True, memory: str = "48g") -> list[str]:
    """The hardened `docker run` for miner code. Kept as data so tests can assert every flag."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        JOB_CONTAINER,
        "--log-driver",
        "none",  # stdout is capped by the pod script; docker's own log files would fill the disk
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


def pod_script(seed: int, time_limit_s: int, gpus: bool = True, grace_s: int = KILL_GRACE_S) -> str:
    inner = " ".join(shlex.quote(c) for c in inner_docker_cmd(seed, time_limit_s, gpus=gpus))
    kill_at = int(time_limit_s) + int(grace_s)
    gpu = "--gpus all " if gpus else ""
    return f"""set -e
# the stage the job is in (read at the deadline: past it in a stage of the miner's bundle, the run is his)
echo ours_setup > /work/STAGE
# room kept for our own report files whatever the miner's code writes (it can fill /work/out)
fallocate -l 160M /work/.reserve 2>/dev/null || dd if=/dev/zero of=/work/.reserve bs=1M count=160 2>/dev/null
infra() {{ rm -f /work/.reserve; printf '{{"error": "%s"}}\\n' "$1" > /work/infra_error.json; exit 0; }}
(docker info >/dev/null 2>&1) || (nohup dockerd >/tmp/dockerd.log 2>&1 & sleep 8)
if ! docker image inspect {BASE_IMAGE} >/dev/null 2>&1; then
  curl -fsSL --retry 3 --retry-delay 5 --max-time 1800 -o /work/base.tar "{BASE_IMAGE_URL}" || infra base_download
  echo "{BASE_IMAGE_SHA256}  /work/base.tar" | sha256sum -c --status || infra base_sha256
  docker load -i /work/base.tar >/work/load.log 2>&1 || infra base_load
  rm -f /work/base.tar
fi
# the legacy builder uses the loaded base as it is (never contacts a registry)
DOCKER_BUILDKIT=0 docker build -q -t {IMAGE} /work/harness >/work/build.log 2>&1 || infra build
# our pod must start a container and see its GPU before any miner code runs
docker run --rm --network none {gpu}--entrypoint {"nvidia-smi" if gpus else "true"} {IMAGE} {"-L" if gpus else ""} \\
  >/work/probe.log 2>&1 || infra gpu
rm -rf /work/sub /work/out && mkdir -p /work/sub /work/out /work/result
size=$(stat -c %s /work/bundle.tar)
if [ "$(df --output=avail -B1 /work | tail -1)" -lt $(( size * 3 + 1073741824 )) ]; then infra disk; fi
# the miner's archive: bounded in time, a stall in it is the miner's (never a missing DONE, never ours)
echo miner_extract > /work/STAGE
timeout {EXTRACT_S} tar -xf /work/bundle.tar -C /work/sub --no-same-owner --no-same-permissions || {{
  rm -f /work/.reserve; echo '{{"error": "extract"}}' > /work/bundle_error.json; exit 0; }}
timeout {EXTRACT_S} chmod -R a+rX /work/sub /work/in || {{
  rm -f /work/.reserve; echo '{{"error": "extract"}}' > /work/bundle_error.json; exit 0; }}
chown 65534:65534 /work/out
# inodes for the report files now: the miner's code could use them all up
: > /work/pod.json; : > /work/result/outputs.jsonl; : > /work/result/meta.json; : > /work/result/run.log
: > /work/snap_outputs.jsonl
set +e
echo miner_run > /work/STAGE
t0=$(date +%s)
# what the miner's code had written at its time limit: a run that switched the alarm off gains nothing
# (never blocks: no symlink, non-blocking open, so a FIFO swapped in by the miner yields nothing)
( sleep {int(time_limit_s) + OVER_LIMIT_SLACK_S}
  timeout 60 dd if=/work/out/outputs.jsonl of=/work/snap_outputs.jsonl iflag=nofollow,nonblock bs=1M \\
    count={RESULT_CAP_BYTES // 2**20} 2>/dev/null ) &
snap=$!
# SIGTERM to the docker CLI does not stop the container (its PID 1 ignores it): kill it by name;
# its stdout is capped (and drained, so the container never blocks on a full pipe)
timeout -k 15 {kill_at} {inner} 2>&1 | {{ head -c 52428800 > /work/inner.log; cat > /dev/null; }}
code=${{PIPESTATUS[0]}}
secs=$(( $(date +%s) - t0 ))
killed=false
if [ $secs -ge {kill_at} ]; then killed=true; docker kill {JOB_CONTAINER} >/dev/null 2>&1; fi
docker rm -f {JOB_CONTAINER} >/dev/null 2>&1 || true
rm -f /work/.reserve
over=false
if [ $secs -gt {int(time_limit_s) + OVER_LIMIT_SLACK_S} ]; then over=true; else kill $snap 2>/dev/null; fi
docker_failed=false
case "$code" in 125|126|127) docker_failed=true ;; esac
info="{{\\"exit\\": $code, \\"seconds\\": $secs, \\"killed_at_limit\\": $killed, \\"over_limit\\": $over, \\"docker_failed\\": $docker_failed, \\"kernel\\": \\"$(uname -r)\\"}}"
echo "$info" > /work/pod.json
# /work/out is writable by the miner's code: copy only regular files, size-capped, for the orchestrator
for f in outputs.jsonl:{RESULT_CAP_BYTES} meta.json:1048576; do
  n=${{f%%:*}}; cap=${{f##*:}}
  if [ -f /work/out/$n ] && [ ! -L /work/out/$n ]; then head -c $cap /work/out/$n > /work/result/$n; fi
done
if [ "$over" = true ]; then
  for i in $(seq 1 90); do kill -0 $snap 2>/dev/null || break; sleep 1; done; kill $snap 2>/dev/null
  cp /work/snap_outputs.jsonl /work/result/outputs.jsonl; fi
tail -c 1048576 /work/inner.log > /work/result/run.log 2>/dev/null || true
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
#: the longest the miner's archive may take to check or unpack on the pod (50 GB at ~100 MB/s is ~9 min)
EXTRACT_S = 1500
SETUP_SECONDS = 3600  # download (up to 50 GB), checks, image build, extraction; within the 3 h pod lifetime


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
echo ours_download > /work/STAGE
fail() {{ printf '{{"error": "%s"}}\\n' "$1" > /work/bundle_error.json; exit 0; }}
# a 4xx (the file is gone, no access) is the miner's; no answer, DNS or a 5xx is the network's or the host's
infra() {{ printf '{{"error": "%s", "http": "%s"}}\\n' "$1" "$2" > /work/infra_error.json; exit 0; }}
code=$(curl -sSL --proto =https --proto-redir =https --retry 3 --connect-timeout 30 --max-filesize {int(max_bytes)} \\
  -o /work/bundle.tar -w '%{{http_code}}' \\
  {shlex.quote(url)} 2> /work/fetch.log)
rc=$?
case "$code" in
  2??) if [ "$rc" -eq 63 ]; then fail too_big; fi  # over --max-filesize: the miner's bundle is too big
       [ "$rc" -eq 0 ] || infra download "$code/curl$rc" ;;  # cut off mid-download: ours, not a bad hash
  408|425|429) infra download "$code" ;;
  4??) fail download ;;
  *) infra download "$code" ;;
esac
echo "{sha256}  /work/bundle.tar" | sha256sum -c --status - || fail sha256
echo miner_inspect > /work/STAGE
timeout {EXTRACT_S} python3 /work/inspect_bundle.py /work/bundle.tar > /work/bundle_info.json 2>/work/inspect.log \
  || fail invalid
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


def _flat_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """The harness meta the miner's code may have rewritten, kept flat: a few known keys of plain types,
    strings capped. Nothing nested is kept, so storing it can never fail (a deep nesting raised
    RecursionError when the run was saved, which looked like our failure)."""
    out: dict[str, Any] = {}
    errs = meta.get("errors")
    if isinstance(errs, list):
        out["errors"] = [str(x)[:500] for x in errs[:20] if isinstance(x, (str, int, float))] or (
            ["error"] if errs else []
        )
    for k in ("traceback", "stage", "version"):
        if isinstance(meta.get(k), str):
            out[k] = meta[k][:4000]
    for k, v in meta.items():
        if isinstance(k, str) and len(k) <= 40 and isinstance(v, (int, float, bool)) and len(out) < 40:
            out.setdefault(k, v)
    return out


def _read_json(p: Path) -> dict[str, Any]:
    """A JSON object from a file the miner's code may have written: anything malformed reads as empty (the
    run is then classified from what else came back), never an exception that would look like ours."""
    try:
        d = json.loads(p.read_text(errors="replace"))
        return d if isinstance(d, dict) else {}
    except Exception:
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
    for stale in ("result", "pod.json", "bundle_error.json", "infra_error.json", "build.log"):
        p = workdir / stale  # left by an earlier run in this workdir (a king re-runs in the same one)
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
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
        from humanizer_eval.runner.lium import GuardError, RentRefused

        # 'rent refused': nothing was created (a clear refusal, or our own guard); 'rent failed': a pod
        # may have been created and billed (a timeout, a 5xx), so it counts like any paid attempt
        tag = "rent refused" if isinstance(e, (RentRefused, GuardError)) else "rent failed"
        return PodJobResult("infra_error", f"{tag}: {type(e).__name__}: {e}", {}, {}, {}, None,
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
                  # DONE exists before the job: writing it after needs no new inode (the miner's code
                  # could use them all up through /out, and a missing DONE would look like our failure)
                  ": > /work/DONE; : > /work/run.log; : > /work/STAGE\n"
                  "nohup bash -c 'bash /work/run.sh > /work/run.log 2>&1; echo $? > /work/DONE' "
                  ">/dev/null 2>&1 &")  # fmt: skip
        sess.run(launch, timeout=120)
        done, ssh_failures = None, 0
        # our setup, the miner's bounded stages (inspect, unpack: EXTRACT_S each; the run: its limit plus the
        # kill grace) and slack: a miner's stage can never run past it, so missing it is always ours
        job_deadline = (time.time() + time_limit_s + KILL_GRACE_S + OVER_LIMIT_SLACK_S + 2 * EXTRACT_S
                        + SETUP_SECONDS)  # fmt: skip
        while time.time() < job_deadline:
            time.sleep(poll_seconds)
            try:
                r = sess.run("cat /work/DONE 2>/dev/null || true", timeout=120)
                if r.returncode == 255:  # ssh itself failed (refused, reset): the pod may be gone
                    raise subprocess.TimeoutExpired("ssh", 120)
                ssh_failures = 0
            except subprocess.TimeoutExpired:
                ssh_failures += 1
                if ssh_failures >= 10:
                    raise RuntimeError("lost the pod (ssh failed 10 times in a row)") from None
                continue
            if r.stdout.strip().isdigit():
                done = int(r.stdout.strip())
                break
        stage = ""
        if done is None:  # past the deadline: in whose stage?
            with contextlib.suppress(Exception):
                stage = sess.run("cat /work/STAGE 2>/dev/null || true", timeout=60).stdout.strip()
        for remote in ("/work/result", "/work/pod.json", "/work/bundle_error.json", "/work/infra_error.json",
                       "/work/build.log"):  # fmt: skip
            # absent files are part of the classification below; a failed copy is tried again (ours)
            for attempt in range(3):
                try:
                    sess.get(remote, workdir)
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    if attempt < 2:
                        time.sleep(5)
        out_dir = workdir / "result"
        berr = _read_json(workdir / "bundle_error.json")
        ierr = _read_json(workdir / "infra_error.json")
        meta = _flat_meta(_read_json(out_dir / "meta.json"))
        podinfo = _read_json(workdir / "pod.json")
        # every file here was written by the miner's code: whatever it holds is the miner's, never a crash
        raw = (out_dir / "outputs.jsonl").read_bytes() if (out_dir / "outputs.jsonl").exists() else b""
        for line in raw.decode("utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except (ValueError, RecursionError):  # anything the miner wrote: skipped, never a crash of ours
                continue
            if isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("text"), str):
                # lone surrogates (valid JSON escapes, invalid text) would crash the judges' requests
                text = row["text"][:20000].encode("utf-8", errors="replace").decode("utf-8")
                outputs[row["id"]] = text
        if done is None:
            status, reason = "infra_error", f"job did not finish in time (stage {stage or '?'})"
        elif ierr:  # the download, the disk, docker or the GPU failed on our side: nothing charged, retried
            status, reason = "infra_error", f"pod: {ierr.get('error')} (http {ierr.get('http', '-')})"
        elif berr:
            status, reason = "miner_error", f"bundle: {berr.get('error', 'invalid')}"
        elif not out_dir.exists():  # it ran (no bundle error): /work/result not copied back is our transfer
            status, reason = "infra_error", "result copy failed"
        elif podinfo and podinfo.get("killed_at_limit"):  # the bundle ran past its limit: scored as it is
            status, reason = "miner_error", f"time limit exceeded: killed after {podinfo.get('seconds')} s"
        elif meta or outputs:  # outputs came back (a malformed meta.json reads as empty): scored as they are
            status = "miner_error" if meta.get("errors") and not outputs else "ok"
            reason = "entrypoint failed" if meta.get("errors") else ""
        elif podinfo:
            status, reason = "miner_error", f"container ended without results (exit {podinfo.get('exit')})"
        else:
            status, reason = "infra_error", f"setup failed (job exit {done})"
    except Exception as e:
        status, reason = "infra_error", f"{type(e).__name__}: {e}"
    finally:
        # a failed delete must not throw away a finished run: Lium removes the pod at its scheduled time
        # anyway, and the next rent reconciles our registry with the pods that still exist
        cost = None
        for attempt in range(3):  # a pod left running costs money and a rent slot: tried again
            try:
                cost = client.delete(pod.pod_id).actual_usd
                break
            except Exception as e:
                if attempt == 2:
                    log.error("pod %s not deleted (%s: %s); the next rent deletes it", pod.pod_id,
                              type(e).__name__, e)  # fmt: skip
                    with contextlib.suppress(Exception):
                        client.mark_delete_failed(pod.pod_id)
                else:
                    time.sleep(5)
    return PodJobResult(status, reason, outputs, meta, podinfo, cost, round(time.time() - t0, 1),
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
