"""Submissions parked because our side kept failing (intake/worker.py _park): the owner decides.

    python scripts/parked.py list
    python scripts/parked.py resume <submission_id>     # run it again (from its saved outputs if any)
    python scripts/parked.py release <submission_id>    # close it: the hold given back, nothing charged
    python scripts/parked.py settle <submission_id>     # close it, charging what was used (pod, metered
                                                        # judges and detectors, at most the hold); no score

Restart nothing: the running server picks a resumed row up on its next step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from humanizer_eval.billing import Accounts
from humanizer_eval.intake.queue import SubmissionQueue
from humanizer_eval.paths import var_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "resume", "release", "settle"))
    ap.add_argument("sid", nargs="?")
    ap.add_argument("--state", type=Path, default=None)
    args = ap.parse_args()
    state = args.state or var_dir() / "intake"
    q = SubmissionQueue(state / "queue.sqlite")
    if args.cmd == "list":
        for r in q.by_state("paused"):
            if '"parked"' in (r.note or ""):
                print(json.dumps({"id": r.id, "hotkey": r.hotkey, "period": r.period_id, "note": r.note}))
        return 0
    row = q.get(args.sid or "")
    if row is None or row.state != "paused" or '"parked"' not in (row.note or ""):
        print("not a parked submission", file=sys.stderr)
        return 1
    if args.cmd == "resume":
        q.update(row.id, state="queued", retry_at=None, pauses=0, attempts=0, note=None)
        print(f"{row.id} queued again")
        return 0
    acc = Accounts(state / "billing.sqlite")
    h = acc.hold(row.id)
    if args.cmd == "settle":
        from humanizer_eval.billing import BillingService, PriceCard
        from humanizer_eval.billing.prices import statement as make_statement
        from humanizer_eval.evaluation.pipeline import SubmissionStore
        from humanizer_eval.period.crypto import load_or_create_key

        if h is None or h.status != "open":
            print("no open hold to settle", file=sys.stderr)
            return 1
        store = SubmissionStore(state / "periods" / str(row.period_id) / "submissions",
                                load_or_create_key(var_dir() / "keys" / "period.key"))  # fmt: skip
        run, m = store.load_run(row.id), store.load_metered(row.id)
        card = BillingService(PriceCard.load("configs/price_card.json"), acc).card_for(row.id)  # the quote's
        st = make_statement(
            card,
            detector_words={d: list(w) for d, w in m["detector_words"].items() if d in card.detectors},
            judge_tokens={k: tuple(v) for k, v in m["judge_tokens"].items() if k in card.raw["judges"]},
            pod_hours=(run.pod_seconds if run else 0.0) / 3600.0,
            meta={"submission_id": row.id, "settled_by_owner": True},
        )
        out = BillingService(PriceCard.load("configs/price_card.json"), acc).finish(row.id, st)
        store.save_statement(row.id, {**st.to_dict(), "settlement": out})
        q.update(
            row.id, state="done", note=json.dumps({"settled_by_owner": True})
        )  # the registration is used
        print(
            f"{row.id} closed; charged {out['charged_cents'] / 100:.2f} USDC of {out['hold_cents'] / 100:.2f}"
        )
        return 0
    if h is not None and h.status == "settled":  # charged for a recorded result: only resume finishes it
        print(f"{row.id} was already charged ({h.charged_cents / 100:.2f} USDC): use resume, not release",
              file=sys.stderr)  # fmt: skip
        return 1
    if h is not None and h.status == "open":
        acc.release(row.id)
    q.update(row.id, state="failed", note=json.dumps({"released_by_owner": True}))
    print(f"{row.id} closed; hold released")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
