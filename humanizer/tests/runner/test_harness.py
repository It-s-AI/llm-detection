import json
import subprocess
import sys
from pathlib import Path

import pytest

from humanizer_eval.runner.evaluate import inner_docker_cmd, pod_script

ROOT = Path(__file__).parents[2]


def run_harness(tmp_path, src_code, items, manifest=None):
    sub, inp, out = tmp_path / "sub", tmp_path / "in", tmp_path / "out"
    (sub / "src").mkdir(parents=True)
    inp.mkdir()
    (sub / "submission.json").write_text(json.dumps(manifest or {"name": "t", "batch_size": 2}))
    (sub / "src" / "humanize.py").write_text(src_code)
    (inp / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    env = {
        "HUMEVAL_SUB": str(sub),
        "HUMEVAL_IN": str(inp),
        "HUMEVAL_OUT": str(out),
        "HUMEVAL_TIME_LIMIT": "20",
        "PATH": "/usr/bin:/bin",
    }
    subprocess.run([sys.executable, str(ROOT / "harness" / "run.py")], env=env, check=True, timeout=60)
    outs = [json.loads(line) for line in (out / "outputs.jsonl").read_text().splitlines()]
    return outs, json.loads((out / "meta.json").read_text())


ITEMS = [{"id": f"i{k}", "text": f"text {k}", "lang": "en", "genre": "essay"} for k in range(5)]


def test_identity_humanizer_roundtrip(tmp_path):
    outs, meta = run_harness(
        tmp_path, "def humanize(items):\n    return [i['text'].upper() for i in items]\n", ITEMS
    )
    assert [o["text"] for o in outs] == [f"TEXT {k}" for k in range(5)]
    assert meta["errors"] == [] and meta["n_outputs"] == 5


def test_crashing_entrypoint_is_recorded_and_outputs_empty(tmp_path):
    outs, meta = run_harness(tmp_path, "def humanize(items):\n    raise RuntimeError('boom')\n", ITEMS)
    assert all(o["text"] == "" for o in outs) and "boom" in meta["errors"][0]


def test_wrong_return_shape_is_an_error(tmp_path):
    outs, meta = run_harness(tmp_path, "def humanize(items):\n    return ['x']\n", ITEMS)
    assert meta["errors"] and all(o["text"] == "" for o in outs)


def test_time_limit_enforced(tmp_path):
    code = "import time\ndef humanize(items):\n    time.sleep(100)\n    return [i['text'] for i in items]\n"
    _outs, meta = run_harness(tmp_path, code, ITEMS[:1])
    assert "TimeoutError" in meta["errors"][0]


def test_inner_container_is_hardened():
    cmd = inner_docker_cmd(seed=7, time_limit_s=60)
    joined = " ".join(cmd)
    for flag in [
        "--network none",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--read-only",
        "--user 65534:65534",
        "--pids-limit",
        "/work/sub:/sub:ro",
        "/work/in:/in:ro",
    ]:
        assert flag in joined, flag
    assert "--privileged" not in joined and "docker.sock" not in joined and "--network host" not in joined
    assert "HUMEVAL_SEED=7" in joined


def test_pod_script_runs_inner_container_with_timeout():
    s = pod_script(seed=1, time_limit_s=60)
    assert "timeout -k 15 180 docker run --rm --name humeval-job --log-driver none --network none" in s
    assert "--entrypoint nvidia-smi" in s and "infra gpu" in s  # our GPU is checked before miner code


FAKE_DOCKER = """#!/bin/bash
# a bundle that switched its alarm off and ignores SIGTERM: only a kill by name stops it
case "$1" in
  run) case "$*" in *--entrypoint*) exit 0 ;; esac  # our probe container
       trap '' TERM; echo $$ > "$STATE/pid"; while true; do sleep 0.2; done ;;
  kill) kill -9 "$(cat "$STATE/pid")"; echo killed > "$STATE/killed" ;;
  version) echo 27.0 ;;
  *) exit 0 ;;
esac
"""


def test_a_bundle_that_ignores_the_time_limit_is_killed_from_outside(tmp_path):
    import os
    import tarfile

    work, bin_, state = tmp_path / "work", tmp_path / "bin", tmp_path / "state"
    for d in (work / "in", work / "harness", bin_, state):
        d.mkdir(parents=True)
    with tarfile.open(work / "bundle.tar", "w"):
        pass
    (bin_ / "docker").write_text(FAKE_DOCKER)
    (bin_ / "docker").chmod(0o755)
    s = pod_script(seed=1, time_limit_s=1, gpus=False, grace_s=1).replace("/work", str(work))
    s = s.replace("chown 65534:65534", "true")
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "STATE": str(state)}
    r = subprocess.run(["bash", "-c", s], env=env, timeout=60, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    info = json.loads((work / "pod.json").read_text())
    assert info["killed_at_limit"] is True and info["seconds"] < 30
    assert (state / "killed").exists()


@pytest.mark.parametrize(("http", "who"), [("404", "bundle_error"), ("403", "bundle_error"),
                                           ("503", "infra_error"), ("000", "infra_error")])  # fmt: skip
def test_a_failed_download_is_the_miners_only_for_a_4xx(tmp_path, http, who):
    """No answer, DNS trouble or a 5xx from Hugging Face is not the miner's fault: it must not be billed
    as a failed bundle (it used up the registration)."""
    import os

    from humanizer_eval.runner.evaluate import fetch_script

    work, bin_ = tmp_path / "work", tmp_path / "bin"
    work.mkdir()
    bin_.mkdir()
    (bin_ / "curl").write_text(f"#!/bin/bash\nprintf {http}\nexit 22\n")
    (bin_ / "curl").chmod(0o755)
    s = fetch_script("https://huggingface.co/a/b/resolve/" + "c" * 40 + "/bundle.tar", "a" * 64)
    env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"}
    subprocess.run(["bash", "-c", s.replace("/work", str(work))], env=env, timeout=30, check=True)
    assert (work / f"{who}.json").exists()
    assert not (work / ("infra_error.json" if who == "bundle_error" else "bundle_error.json")).exists()
