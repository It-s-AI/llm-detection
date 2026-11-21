"""Rent ONE cheap Lium pod for <= 1 hour, probe the environment, delete it.

Purpose: learn what the evaluation harness can rely on inside a Lium pod (kernel,
GPU, Docker-in-Docker, `--network none`, GPU inside an inner container, gVisor).
Uses the guarded LiumClient: own pods only, spend cap, balance floor, scheduled removal.

Usage: .venv/bin/python scripts/lium_probe.py --yes [--executor <uuid>]
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.runner.lium import LiumClient, Registry

TEMPLATE = "cbc0ffdc-aa35-441a-abaa-170a87c51a64"  # Lium official "Pytorch (Cuda + DinD)", CUDA 12.8

PROBE = r"""
set +e
echo "== kernel"; uname -r
echo "== id"; id
echo "== gpu"; nvidia-smi -L 2>&1 | head -2
echo "== docker"; (docker info >/dev/null 2>&1 || (nohup dockerd >/tmp/dockerd.log 2>&1 & sleep 8)); docker version --format '{{.Server.Version}}' 2>&1 | head -1
echo "== pod outbound"; curl -s -m 6 -o /dev/null -w "%{http_code}" https://example.com; echo
echo "== caps"; grep -E 'CapEff|NoNewPrivs|Seccomp' /proc/self/status
echo "== iptables"; iptables -L OUTPUT -n 2>&1 | head -2
echo "== inner --network none"; timeout 120 docker run --rm --network none alpine sh -c "ip -o link | cut -d: -f2; wget -T 3 -q -O- http://1.1.1.1 >/dev/null 2>&1 && echo NET_OK || echo NO_NET" 2>&1 | tail -4
echo "== inner hardened"; timeout 120 docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges --read-only --user 65534:65534 --pids-limit 256 alpine sh -c "id; touch /x 2>&1 | head -1" 2>&1 | tail -3
echo "== inner gpu"; timeout 300 docker run --rm --network none --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi -L 2>&1 | tail -2
echo "== gvisor"; (curl -fsSL -m 60 https://storage.googleapis.com/gvisor/releases/release/latest/x86_64/runsc -o /usr/local/bin/runsc && chmod +x /usr/local/bin/runsc && runsc --version | head -1) 2>&1 | tail -1
echo "== gvisor run"; (mkdir -p /etc/docker && echo '{"runtimes":{"runsc":{"path":"/usr/local/bin/runsc"},"runsc-gpu":{"path":"/usr/local/bin/runsc","runtimeArgs":["--nvproxy=true"]}}}' > /etc/docker/daemon.json && pkill dockerd; sleep 3; nohup dockerd >/tmp/dockerd2.log 2>&1 & sleep 10; timeout 120 docker run --rm --runtime=runsc --network none alpine sh -c "dmesg 2>/dev/null | head -1; echo GVISOR_OK") 2>&1 | tail -2
echo "== gvisor gpu"; timeout 300 docker run --rm --runtime=runsc-gpu --network none --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi -L 2>&1 | tail -2
echo "== done"
"""


def _ssh_target(pod: dict) -> tuple[str, str]:
    cmd = pod.get("ssh_connect_cmd") or ""
    parts = shlex.split(cmd)
    host = next((p for p in parts if "@" in p), None)
    port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
    if not host:
        raise RuntimeError(f"cannot parse ssh_connect_cmd: {cmd!r}")
    return host, port


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually rent (costs money)")
    ap.add_argument("--executor", default="bcf6eb87-3278-4e35-b1ab-28af16c8feea")
    ap.add_argument("--max-price", type=float, default=0.30)
    ap.add_argument("--script", default=None, help="custom probe script to run instead of PROBE")
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    lium_dir = var_dir() / "lium"
    lium_dir.mkdir(parents=True, exist_ok=True)
    client = LiumClient(
        s.lium_api_key,
        led,
        Registry(lium_dir / "registry.json"),
        enabled=args.yes,
        floor_hours=s.lium_balance_floor_hours,
        max_price_per_hour=args.max_price,
        max_hours=1,
    )
    execs = {e["id"]: e for e in client.executors()}
    ex = execs.get(args.executor)
    if ex is None:
        cands = sorted(
            (
                e
                for e in execs.values()
                if e.get("tier") == "secure"
                and e["price_per_gpu"] <= args.max_price
                and (e.get("reliability_score") or 0) >= 90
                and e.get("available_gpu_count", 0) >= 1
            ),
            key=lambda e: e["price_per_gpu"],
        )
        if not cands:
            print("no suitable executor")
            return 1
        ex = cands[0]
    print("executor:", ex["id"], ex["machine_name"], ex["price_per_gpu"], "$/h")
    print("guard check:", client.check_can_rent(float(ex["price_per_gpu"]), 1))
    # ssh refuses keys with open permissions; /workspace is a network FS that forces 0666,
    # so the ephemeral key lives in ~/.ssh on the local disk.
    key = Path.home() / ".ssh" / "humeval_lium_ed25519"
    key.parent.mkdir(mode=0o700, exist_ok=True)
    if not key.exists():
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key), "-q"], check=True)
    key.chmod(0o600)
    pub = (key.with_suffix(".pub")).read_text().strip()
    pod = client.rent(ex, purpose="probe", ssh_public_key=pub, max_hours=1, template_id=TEMPLATE)
    print("rented:", pod.name, pod.pod_id)
    result = {
        "executor": {k: ex.get(k) for k in ("id", "machine_name", "price_per_gpu", "tier")},
        "pod": pod.name,
    }
    try:
        live = None
        deadline = time.time() + 900
        while time.time() < deadline:
            live = next((p for p in client.pods() if p["id"] == pod.pod_id), None)
            if live and live.get("status") == "RUNNING" and live.get("ssh_connect_cmd"):
                break
            time.sleep(10)
        else:
            raise RuntimeError(f"pod not ready in time: {live and live.get('status')}")
        host, port = _ssh_target(live)
        time.sleep(15)
        ssh = [
            "ssh",
            "-i",
            str(key),
            "-p",
            port,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=20",
            host,
            "bash -s",
        ]
        t0 = time.time()
        proc = subprocess.run(
            ssh,
            input=Path(args.script).read_text() if args.script else PROBE,
            capture_output=True,
            text=True,
            timeout=1500,
        )
        result["probe_seconds"] = round(time.time() - t0)
        result["stdout"] = proc.stdout
        result["stderr_tail"] = proc.stderr[-1500:]
        print(proc.stdout)
    finally:
        own = client.delete(pod.pod_id)
        result["actual_usd"] = own.actual_usd
        print("deleted:", own.name, "billed:", own.actual_usd)
        (lium_dir / "probe_result.json").write_text(json.dumps(result, indent=1))
    print("ledger lium:", led.summary().get("lium"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
