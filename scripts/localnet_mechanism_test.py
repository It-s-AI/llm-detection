"""Mechanism-1 weights on a subtensor localnet, on ONE cheap Lium pod (this machine has no docker).

Rents a 3090/4090-class pod (guarded client: own pods only, spend cap, scheduled removal), starts
ghcr.io/opentensor/subtensor-localnet in docker there, installs bittensor 10.5.0 and runs
scripts/localnet/mech1_test.py with the validator module from llm-detection. Fetches
/work/result.json and the logs into docs/reports/data/11-localnet/, deletes the pod.

Usage: python scripts/localnet_mechanism_test.py --yes [--validator-module ../llm-detection/...py]
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import time
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.evaluation.runners import rank_executors
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.runner.evaluate import TEMPLATE_DIND, PodSession, _ssh_target
from humanizer_eval.runner.lium import LiumClient, Registry

IMAGE = "ghcr.io/opentensor/subtensor-localnet:devnet-ready"
CHEAP = ("3090", "4090", "A4000", "A5000")

SETUP = rf"""
set -x
cd /work
(docker info >/dev/null 2>&1 || (nohup dockerd >/work/dockerd.log 2>&1 & sleep 12))
docker pull {IMAGE} > /work/pull.log 2>&1
docker run -d --name localnet -p 9944:9944 -p 9945:9945 {IMAGE}
(python3 -m venv /work/venv || (pip install -q virtualenv && python3 -m virtualenv /work/venv)) > /work/venv.log 2>&1
/work/venv/bin/pip install -q "bittensor==10.5.0" pynacl > /work/pip.log 2>&1
python3 --version
for i in $(seq 1 180); do
  curl -s -m 3 -H 'Content-Type: application/json' \
    -d '{{"id":1,"jsonrpc":"2.0","method":"system_health"}}' http://127.0.0.1:9944 | grep -q peers && break
  sleep 2
done
docker logs localnet 2>&1 | tail -5
timeout 3000 /work/venv/bin/python /work/mech1_test.py
echo "exit=$?"
docker logs localnet > /work/localnet.log 2>&1
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--max-price", type=float, default=0.4)
    ap.add_argument(
        "--validator-module",
        type=Path,
        default=Path("/workspace/sn32_owner/llm-detection/detection/validator/humanizer_mechanism.py"),
    )
    ap.add_argument("--out", type=Path, default=Path("docs/reports/data/11-localnet"))
    ap.add_argument("--test", type=Path, default=Path("scripts/localnet/mech1_test.py"),
                    help="pod-side test script (runs as /work/mech1_test.py)")  # fmt: skip
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    client = LiumClient(s.lium_api_key, led, Registry(var_dir() / "lium" / "registry.json"), enabled=args.yes,
                        floor_hours=s.lium_balance_floor_hours, max_price_per_hour=args.max_price, max_hours=2)  # fmt: skip
    key = Path.home() / ".ssh" / "humeval_lium_ed25519"
    pub = Path(str(key) + ".pub").read_text().strip()
    ranked = rank_executors(client.executors(), args.max_price, CHEAP)
    print(
        f"{len(ranked)} candidate executors; first: {[(e['machine_name'], e['price_per_gpu']) for e in ranked[:3]]}"
    )
    pod = None
    for ex in ranked[:3]:
        try:
            pod = client.rent(ex, purpose="localnet-mech1", ssh_public_key=pub, max_hours=2,
                              template_id=TEMPLATE_DIND)  # fmt: skip
            print("rented", pod.name, ex["machine_name"], ex["price_per_gpu"])
            break
        except Exception as e:
            print("rent failed:", type(e).__name__, str(e)[:300])
    if pod is None:
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        deadline = time.time() + 900
        live = None
        while time.time() < deadline:
            live = next((p for p in client.pods() if p["id"] == pod.pod_id), None)
            if live and live.get("status") == "RUNNING" and live.get("ssh_connect_cmd"):
                break
            time.sleep(10)
        else:
            raise RuntimeError("pod not ready in 15 min")
        sess = PodSession(*_ssh_target(live), key)
        time.sleep(10)
        sess.run("mkdir -p /work", timeout=120)
        sess.put(args.test, "/work/mech1_test.py")
        sess.put(args.validator_module, "/work/humanizer_mechanism.py")
        launch = (f"cat > /work/setup.sh <<'EOF_JOB'\n{SETUP}\nEOF_JOB\n"
                  "nohup bash -c 'bash /work/setup.sh > /work/run.log 2>&1; echo $? > /work/DONE' >/dev/null 2>&1 &")  # fmt: skip
        sess.run(launch, timeout=120)
        t_end = time.time() + 105 * 60  # the pod is removed at 2 h anyway
        last = ""
        while time.time() < t_end:
            time.sleep(30)
            try:
                r = sess.run("cat /work/DONE 2>/dev/null; tail -n 1 /work/run.log 2>/dev/null", timeout=120)
            except subprocess.TimeoutExpired:
                continue
            lines = r.stdout.strip().splitlines()
            if lines and lines[-1] != last:
                last = lines[-1]
                print("pod:", last[:200], flush=True)
            if lines and lines[0].strip().isdigit():
                break
        for remote in ("/work/result.json", "/work/run.log", "/work/localnet.log", "/work/pip.log"):
            with contextlib.suppress(subprocess.CalledProcessError, subprocess.TimeoutExpired):
                sess.get(remote, args.out)
    finally:
        own = client.delete(pod.pod_id)
        print("deleted", own.name, "billed", own.actual_usd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
