"""Back up the evaluation state that cannot be rebuilt: miners' balances, the queue, the spend ledger,
the king state and the keys. SQLite files are copied with the backup API (safe while the server runs).

    python scripts/backup_state.py --var /srv/humeval/var --out /srv/humeval/backups --keep 72

Each run writes <out>/<UTC timestamp>/ (mode 700) and keeps the newest --keep runs. The keys are in it:
copy the backups off the machine encrypted. Not included: the king bundle archive (large; back it up
separately) and the per-submission records (encrypted; part of a daily snapshot of the var dir).
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import time
from pathlib import Path

SQLITE = ("spend_ledger.sqlite", "intake/queue.sqlite", "intake/billing.sqlite", "billing/pilot.sqlite")
FILES = (
    "intake/king_state.*.json",
    "crown/*.json",
    "keys/*",
    "keys/wallets/*/coldkey*",  # the notary wallet (period/notary.py)
    "intake/periods/*/king_run.*.enc",
    # the period lifecycle: the schedule, a committed but unbuilt secret, the periods themselves
    "periods/schedule.json",
    "periods/CURRENT",
    "periods/NEXT",
    "periods/.*.secret",
    "periods/*/manifest.json",
    "periods/*/secret.hex",
    "periods/*/dataset.enc",
    "periods/*/human_scores.*.enc",
)


def backup(var: Path, out: Path, keep: int) -> Path:
    dest = out / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest.mkdir(parents=True, mode=0o700)
    for rel in SQLITE:
        src = var / rel
        if not src.exists():
            continue
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(src) as s, sqlite3.connect(dest / rel) as d:
            s.backup(d)
    for pattern in FILES:
        for src in var.glob(pattern):
            if src.is_file():
                tgt = dest / src.relative_to(var)
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tgt)
    for p in dest.rglob("*"):
        p.chmod(0o700 if p.is_dir() else 0o600)
    runs = sorted(p for p in out.iterdir() if p.is_dir())
    for old in runs[:-keep]:
        shutil.rmtree(old)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--keep", type=int, default=72)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
    print(backup(args.var, args.out, args.keep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
