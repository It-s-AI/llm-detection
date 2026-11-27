"""In-container runner: load the miner entrypoint and humanize all items.

Runs as an unprivileged user with no network. Paths can be overridden for tests:
HUMEVAL_SUB (bundle, read-only), HUMEVAL_IN (items.jsonl), HUMEVAL_OUT (outputs dir).
"""

import importlib
import json
import os
import platform
import random
import signal
import socket
import subprocess
import sys
import time
import traceback

SUB = os.environ.get("HUMEVAL_SUB", "/sub")
IN = os.environ.get("HUMEVAL_IN", "/in")
OUT = os.environ.get("HUMEVAL_OUT", "/out")
SEED = int(os.environ.get("HUMEVAL_SEED", "0"))
TIME_LIMIT = int(os.environ.get("HUMEVAL_TIME_LIMIT", "3600"))
MAX_CHARS = 20_000


def network_reachable() -> bool:
    try:
        socket.create_connection(("1.1.1.1", 53), timeout=2).close()
        return True
    except OSError:
        return False


def install_wheels(meta):
    req = os.path.join(SUB, "requirements.txt")
    wheels = os.path.join(SUB, "wheels")
    if not os.path.isdir(wheels):
        return
    target = os.environ.get("HUMEVAL_SITE", "/tmp/site")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        wheels,
        "--target",
        target,
        "-q",
    ]
    cmd += ["-r", req] if os.path.isfile(req) else [os.path.join(wheels, w) for w in os.listdir(wheels)]
    t0 = time.time()
    subprocess.run(cmd, check=True, timeout=1800)
    meta["install_seconds"] = round(time.time() - t0, 1)
    sys.path.insert(0, target)


def seed_everything(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % 2**32)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def main():
    meta = {"python": platform.python_version(), "network_reachable": network_reachable(), "errors": []}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(IN, "items.jsonl")) as f:
        items = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(SUB, "submission.json")) as f:
        manifest = json.load(f)
    outputs = {}

    def on_timeout(signum, frame):
        raise TimeoutError(f"time limit {TIME_LIMIT}s exceeded")

    signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(TIME_LIMIT)
    t0 = time.time()
    try:
        install_wheels(meta)
        sys.path.insert(0, os.path.join(SUB, "src"))
        mod_name, _, fn_name = manifest.get("entrypoint", "humanize:humanize").partition(":")
        fn = getattr(importlib.import_module(mod_name), fn_name)
        seed_everything(SEED)
        bs = int(manifest.get("batch_size", 16))
        for start in range(0, len(items), bs):
            batch = items[start : start + bs]
            res = fn([{k: it.get(k) for k in ("text", "lang", "genre")} for it in batch])
            if not isinstance(res, list) or len(res) != len(batch):
                raise ValueError("entrypoint must return a list of strings, one per item")
            for it, text in zip(batch, res, strict=True):
                outputs[it["id"]] = (text if isinstance(text, str) else "")[:MAX_CHARS]
    except BaseException as e:  # record and still write whatever was produced
        meta["errors"].append(f"{type(e).__name__}: {e}")
        meta["traceback"] = traceback.format_exc()[-4000:]
    finally:
        signal.alarm(0)
    meta["seconds"] = round(time.time() - t0, 2)
    meta["n_items"] = len(items)
    meta["n_outputs"] = len(outputs)
    with open(os.path.join(OUT, "outputs.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps({"id": it["id"], "text": outputs.get(it["id"], "")}) + "\n")
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    main()
