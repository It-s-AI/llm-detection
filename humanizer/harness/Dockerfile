# Evaluation harness image (D-60). Miner code runs inside it with --network none,
# no capabilities, read-only rootfs, unprivileged user (see runner/evaluate.py).
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
COPY run.py /opt/humeval/run.py
ENTRYPOINT ["python", "/opt/humeval/run.py"]
