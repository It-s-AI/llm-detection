"""After a detector changed its model in the middle of a period (alert "scoring conditions changed"):
make the period's human reference scores and the king's run stale, so that both are scored again by the
new model before any further comparison (evaluation/conditions.py).

    python scripts/rescore_period.py --period p2 --detector zerogpt [--state var/intake] [--yes]
    sudo systemctl restart humeval-intake        # loads the period again: H and the king are rescored

What it does (nothing is deleted except the stale cache rows; the old files are kept with a suffix):
* removes every cached score of that detector (they come from the old model);
* moves the period's human_scores.<detectors>.enc and the king's run aside (.stale-<time>).
After the restart the server scores H again (new baseline, new canaries) and rescores the king from his
saved outputs; the paused evaluations then resume and are compared under the new, identical conditions.
Evaluations finished before the change keep their results: each was compared under one model.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

from humanizer_eval.paths import var_dir

CROWN = ("gptzero", "pangram", "zerogpt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", required=True)
    ap.add_argument("--detector", required=True, action="append", help="the detector(s) whose model changed")
    ap.add_argument("--detectors", default=",".join(CROWN), help="the crown detector set of the server")
    ap.add_argument("--state", type=Path, default=None, help="the intake state dir")
    ap.add_argument("--yes", action="store_true", help="do it (without: show what would be done)")
    args = ap.parse_args()
    tag = "-".join(sorted(d for d in args.detectors.split(",") if d))
    state = args.state or (var_dir() / "intake")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    files = [var_dir() / "periods" / args.period / f"human_scores.{tag}.enc",
             state / "periods" / args.period / f"king_run.{tag}.enc"]  # fmt: skip
    cache = var_dir() / "detector_cache.sqlite"
    with sqlite3.connect(cache) as c:
        n = {d: c.execute("SELECT COUNT(*) FROM detector_cache WHERE detector = ?", (d,)).fetchone()[0]
             for d in args.detector}  # fmt: skip
    print("cached scores to remove:", n)
    for f in files:
        print(("move aside: " if f.exists() else "not present: ") + str(f))
    if not args.yes:
        print("dry run; add --yes")
        return 0
    with sqlite3.connect(cache) as c:
        for d in args.detector:
            c.execute("DELETE FROM detector_cache WHERE detector = ?", (d,))
    for f in files:
        if f.exists():
            f.rename(f.with_name(f"{f.name}.stale-{stamp}"))
    print("done; restart humeval-intake to rescore H and the king under the new model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
