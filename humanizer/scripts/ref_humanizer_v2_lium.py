"""Reference humanizer v2 on one Lium H100 pod: LoRA training -> bundle (weights + wheels) -> evaluation of
that bundle in the hardened harness container on a pilot's inputs -> outputs and adapter downloaded.

Everything heavy stays on the pod (the merged 8 GB model is not downloaded unless --download-bundle).
The pod is deleted in `finally`; the Lium client guards (own pods only, spend cap, balance floor, one pod,
scheduled removal) apply as for any rental.

Usage: python scripts/ref_humanizer_v2_lium.py --yes [--period pilot-1] [--max-price 2.6]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, encrypt_at_rest, load_or_create_key
from humanizer_eval.runner.evaluate import (
    HARNESS_DIR,
    IMAGE,
    TEMPLATE_DIND,
    PodSession,
    _ssh_target,
    pod_script,
)
from humanizer_eval.runner.lium import LiumClient, Registry

REF = Path("examples/ref_humanizer")
TRAIN_DEPS = "transformers==4.57.1 peft==0.17.1"


def remote_script(seed: int, time_limit_s: int, train_args: str) -> str:
    """Train, assemble /work/bundle.tar, then run the standard harness flow; /work/DONE gets the exit code."""
    build = f"""set -e
cd /work
pip install -q {TRAIN_DEPS} > /work/pip.log 2>&1
python ref/train/train_lora.py --pairs /work/pairs.jsonl --out /work/bundle {train_args} > /work/train.log 2>&1
cp -r ref/src ref/submission.json ref/requirements.txt /work/bundle/
(docker info >/dev/null 2>&1) || (nohup dockerd >/tmp/dockerd.log 2>&1 & sleep 8)
docker build -q -t {IMAGE} /work/harness > /work/build.log 2>&1
mkdir -p /work/bundle/wheels
docker run --rm --entrypoint pip -v /work/bundle/wheels:/w -v /work/bundle/requirements.txt:/r.txt:ro \
  {IMAGE} download -q -r /r.txt -d /w > /work/wheels.log 2>&1
tar -cf /work/bundle.tar -C /work/bundle --exclude=./adapter --exclude=./train_log.json .
du -sb /work/bundle.tar | cut -f1 > /work/bundle_bytes
"""
    return build + pod_script(seed, time_limit_s, gpus=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="really rent (otherwise the guards refuse)")
    ap.add_argument("--period", default="pilot-1")
    ap.add_argument("--max-price", type=float, default=2.6)
    ap.add_argument("--max-hours", type=int, default=2)
    ap.add_argument("--train-args", default="", help="extra args for train_lora.py, e.g. '--epochs 1'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time-limit", type=int, default=3600, help="harness time limit, seconds (D-61)")
    ap.add_argument("--download-bundle", action="store_true", help="also fetch bundle.tar (~8 GB)")
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    lium_dir = var_dir() / "lium"
    work = var_dir() / "ref_v2" / "run"
    work.mkdir(parents=True, exist_ok=True)
    client = LiumClient(
        s.lium_api_key,
        led,
        Registry(lium_dir / "registry.json"),
        enabled=args.yes,
        floor_hours=s.lium_balance_floor_hours,
        max_price_per_hour=args.max_price,
        max_hours=args.max_hours,
    )
    pairs = var_dir() / "ref_v2" / "pairs.jsonl"
    rows = [json.loads(x) for x in pairs.read_text().splitlines() if x.strip()]
    train = [r for r in rows if r["genre"] != "application"]  # genre left the mix (2026-11-10)
    (work / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in train))
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    data = json.loads(
        decrypt_at_rest(key, (var_dir() / "periods" / args.period / "dataset.enc").read_bytes())
    )
    items = [
        {"id": it["id"], "text": it["text"], "lang": "en", "genre": it["slot"]["genre"]}
        for it in data["items"]
    ]
    (work / "in").mkdir(exist_ok=True)
    (work / "in" / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items))

    cands = sorted(
        (
            e
            for e in client.executors()
            if "H100" in e.get("machine_name", "")
            and e.get("tier") == "secure"
            and e["price_per_gpu"] <= args.max_price
            and (e.get("reliability_score") or 0) >= 90
            and e.get("available_gpu_count", 0) >= 1
        ),
        key=lambda e: e["price_per_gpu"],
    )
    if not cands:
        print("no suitable H100 executor")
        return 1
    ex = cands[0]
    print("executor:", ex["machine_name"], ex["price_per_gpu"], "$/h", flush=True)
    print("guard:", client.check_can_rent(float(ex["price_per_gpu"]), args.max_hours), flush=True)
    ssh_key = Path.home() / ".ssh" / "humeval_lium_ed25519"
    pub = Path(str(ssh_key) + ".pub").read_text().strip()
    t0 = time.time()
    pod = client.rent(ex, purpose="ref-v2-train", ssh_public_key=pub, max_hours=args.max_hours,
                      template_id=TEMPLATE_DIND)  # fmt: skip
    summary: dict = {"executor": ex["machine_name"], "price_per_hour": ex["price_per_gpu"], "pod": pod.name}
    try:
        live = None
        deadline = time.time() + 900
        while time.time() < deadline:
            live = next((p for p in client.pods() if p["id"] == pod.pod_id), None)
            if live and live.get("status") == "RUNNING" and live.get("ssh_connect_cmd"):
                break
            time.sleep(10)
        else:
            raise RuntimeError("pod not ready in time")
        sess = PodSession(*_ssh_target(live), ssh_key)
        time.sleep(10)
        sess.run("mkdir -p /work/ref", timeout=120)
        for local, remote in ((HARNESS_DIR, "/work/harness"), (REF / "src", "/work/ref/src"),
                              (REF / "train", "/work/ref/train"), (REF / "submission.json", "/work/ref/"),
                              (REF / "requirements.txt", "/work/ref/"), (work / "pairs.jsonl", "/work/"),
                              (work / "in", "/work/in")):  # fmt: skip
            sess.put(local, remote)
        script = remote_script(args.seed, args.time_limit, args.train_args)
        launch = (f"cat > /work/run.sh <<'EOF'\n{script}\nEOF\n"
                  "nohup bash -c 'bash /work/run.sh > /work/run.log 2>&1; echo $? > /work/DONE' >/dev/null 2>&1 &")  # fmt: skip
        sess.run(launch, timeout=120)
        last = ""
        while time.time() - t0 < args.max_hours * 3600 - 600:
            time.sleep(60)
            st = sess.run("cat /work/DONE 2>/dev/null; tail -n 1 /work/train.log 2>/dev/null", timeout=120)
            lines = st.stdout.strip().splitlines()
            if lines and lines[0].strip().isdigit() and not lines[0].startswith("step"):
                summary["exit"] = int(lines[0])
                break
            if lines and lines[-1] != last:
                last = lines[-1]
                print(f"[{time.time() - t0:.0f}s] {last[:160]}", flush=True)
        else:
            summary["exit"] = "timeout"
        for remote in ("/work/out", "/work/bundle/train_log.json", "/work/train.log", "/work/run.log",
                       "/work/wheels.log", "/work/bundle_bytes", "/work/bundle/adapter"):  # fmt: skip
            try:
                sess.get(remote, work)
            except Exception as e:  # keep going: a missing file must not skip the pod deletion
                summary.setdefault("missing", []).append(f"{remote}: {type(e).__name__}")
        if args.download_bundle:
            t1 = time.time()
            sess.get("/work/bundle.tar", work, timeout=7200)
            summary["bundle_download_s"] = round(time.time() - t1, 1)
    finally:
        own = client.delete(pod.pod_id)
        summary["cost_usd"] = own.actual_usd
        summary["seconds"] = round(time.time() - t0, 1)
    outs = {}
    of = work / "out" / "outputs.jsonl"
    if of.exists():
        for line in of.read_text().splitlines():
            row = json.loads(line)
            outs[row["id"]] = row["text"]
        # pilot outputs are kept encrypted at rest like the other pilot submissions
        chosen = [outs.get(it["id"], "") for it in items]
        pdir = var_dir() / "periods" / args.period
        (pdir / "ref_v2.enc").write_bytes(encrypt_at_rest(key, json.dumps(chosen).encode()))
        of.unlink()
    summary["n_outputs"] = sum(1 for v in outs.values() if v)
    meta = work / "out" / "meta.json"
    summary["meta"] = json.loads(meta.read_text()) if meta.exists() else None
    print(json.dumps(summary, indent=1, default=str))
    (work / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return 0 if summary.get("exit") == 0 and summary["n_outputs"] == len(items) else 2


if __name__ == "__main__":
    raise SystemExit(main())
