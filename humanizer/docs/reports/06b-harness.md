# 06b. Submission execution (harness) — report

## What was done
- **Bundle format** (`runner/bundle.py`): `submission.json` (name, entrypoint `module:function`, license from the
  allowlist, batch size), `src/`, optionally `wheels/` + `requirements.txt` (all dependencies included — there is no network during
  the run), `weights/`. The orchestrator **does not unpack** the bundle: it reads only the tar headers and the manifest; it rejects
  absolute paths, `..`, symlinks/hardlinks/devices, oversized entries and "archive bombs".
- **Sandbox image** (`harness/Dockerfile`, `harness/run.py`): based on PyTorch 2.7.1 / CUDA 12.8; inside —
  offline installation of wheels, loading the entrypoint, batches, seed, time limit, output truncation, response format
  validation, writing `outputs.jsonl` and `meta.json` (including the fact that there is no network).
- **Run on Lium** (`runner/evaluate.py`): rental via the safeguarded client → upload of harness/bundle/inputs over
  SSH → image build in the pod → bundle unpacking **in the pod** → launch in the inner container:
  `--network none --cap-drop ALL --security-opt no-new-privileges --read-only --user 65534:65534 --pids-limit
  --memory --gpus all`, bundle and inputs read-only → output collection → pod deletion in `finally`.
- Example bundle `examples/identity_humanizer` ("change nothing" baseline).

## Live check (RTX 3090, $0.22/h)
| Check | Result |
|---|---|
| Outputs = inputs (identity) | ✅ 6/6 |
| Network from inside the container | ✅ unreachable (`network_reachable: false`) |
| Exit code, kernel, Docker | 0, 6.8.0-147, 27.3.1 |
| Total time (rental → deletion) | 355 s (mostly downloading the base image) |
| Cost | **$0.022** |

## Tests
32 offline tests for runner: bundles (paths, links, devices, manifest, size), harness locally (identity,
entrypoint crash, wrong response shape, time limit), presence of all isolation flags in the launch command.

## What's next
- Publish the image to a registry (GHCR) so it does not have to be built in every pod (will save ~3–4 minutes per run).
- Encrypt inputs with the pod key (`period/crypto.seal_to`) — relevant for CVM nodes.
- Decision on the two-stage crown (gVisor is unavailable, see report 06).
