"""End-to-end check of the evaluation harness on Lium with the identity baseline bundle (cents).

Bundle -> fresh pod -> hardened inner container (--network none, GPU) -> outputs -> pod deleted.
Usage: python scripts/lium_e2e_identity.py --yes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.runner.bundle import pack_bundle
from humanizer_eval.runner.evaluate import run_on_lium
from humanizer_eval.runner.lium import LiumClient, Registry

ITEMS = [
    {"id": f"t{i}", "text": f"Sample input number {i}. It has two sentences.", "lang": "en", "genre": "essay"}
    for i in range(6)
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--max-price", type=float, default=0.30)
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    lium_dir = var_dir() / "lium"
    client = LiumClient(
        s.lium_api_key,
        led,
        Registry(lium_dir / "registry.json"),
        enabled=args.yes,
        floor_hours=s.lium_balance_floor_hours,
        max_price_per_hour=args.max_price,
        max_hours=1,
    )
    cands = sorted(
        (
            e
            for e in client.executors()
            if e.get("tier") == "secure"
            and e["price_per_gpu"] <= args.max_price
            and (e.get("reliability_score") or 0) >= 90
            and e.get("available_gpu_count", 0) >= 1
            and any(g in e["machine_name"] for g in ("3090", "4090", "A4000", "A5000"))
        ),
        key=lambda e: e["price_per_gpu"],
    )
    if not cands:
        print("no suitable executor")
        return 1
    ex = cands[0]
    print("executor:", ex["machine_name"], ex["price_per_gpu"], "$/h")
    bundle = pack_bundle(Path("examples/identity_humanizer"), lium_dir / "identity.tar.gz")
    key = Path.home() / ".ssh" / "humeval_lium_ed25519"
    r = run_on_lium(
        bundle,
        ITEMS,
        client=client,
        executor=ex,
        ssh_key=key,
        workdir=lium_dir / "e2e",
        seed=42,
        time_limit_s=900,
        max_hours=1,
        gpus=True,
    )
    ok = all(r.outputs.get(it["id"]) == it["text"] for it in ITEMS)
    summary = {
        "outputs_match_identity": ok,
        "n_outputs": len(r.outputs),
        "meta": r.meta,
        "pod": r.pod,
        "cost_usd": r.cost_usd,
        "seconds": r.seconds,
    }
    print(json.dumps(summary, indent=1, default=str))
    (lium_dir / "e2e_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("ledger lium:", led.summary().get("lium"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
