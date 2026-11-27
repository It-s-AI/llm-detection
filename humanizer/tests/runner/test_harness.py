import json
import subprocess
import sys
from pathlib import Path

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
    assert "timeout 360 docker run --rm --network none" in s
