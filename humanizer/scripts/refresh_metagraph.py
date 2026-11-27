"""Write the metagraph snapshot the intake API checks registrations against (read-only chain query).

    python scripts/refresh_metagraph.py --netuid 32 --network finney --out var/intake/metagraph.json
    # cron: every 5 minutes; the API refuses new submissions when the snapshot is older than an hour

Needs `bittensor` (the validator's environment). The snapshot holds public data only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from humanizer_eval.intake.registry import write_snapshot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--netuid", type=int, default=32)
    ap.add_argument("--network", default="finney")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import bittensor as bt

    st = bt.Subtensor(network=args.network)
    mg = st.metagraph(args.netuid, lite=True)
    hotkeys = list(mg.hotkeys)
    reg = {hk: int(b) for hk, b in zip(hotkeys, mg.block_at_registration, strict=True)}
    commitments = {}
    registered = set(hotkeys)
    for hk, data in st.get_all_commitments(args.netuid).items():
        if hk in registered:  # commitments outlive registrations; only current ones matter
            meta = st.get_commitment_metadata(args.netuid, hk)
            commitments[hk] = (data, int(meta["block"]))
    write_snapshot(args.out, args.netuid, int(mg.block), hotkeys, reg, commitments=commitments)
    print(
        f"netuid {args.netuid} block {int(mg.block)}: {len(hotkeys)} hotkeys, {len(commitments)} commitments"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
