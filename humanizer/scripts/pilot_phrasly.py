"""Run the Phrasly humanizer on a pilot period's AI inputs (external reference humanizer).

Outputs are stored encrypted next to the period (``phrasly.enc``: {item index: text}); a re-run only
sends the missing items. Cost: $0.14 per 1,000 words (~$3 for the 78-item pilot-1).
The inputs leave our trusted-provider set (D-90): pilots only.
Usage: python scripts/pilot_phrasly.py --period pilot-1 [--limit N]
"""

from __future__ import annotations

import argparse
import json

from humanizer_eval.baselines.phrasly import Phrasly
from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.crypto import decrypt_at_rest, encrypt_at_rest, load_or_create_key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="pilot-1")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default=None, help="Phrasly Ultra mode (gentle|light|balanced|strong|max)")
    ap.add_argument("--model", default=None, help='"latest" forces the standard model on an Ultra account')
    args = ap.parse_args()
    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    key = load_or_create_key(var_dir() / "keys" / "period.key")
    pdir = var_dir() / "periods" / args.period
    items = json.loads(decrypt_at_rest(key, (pdir / "dataset.enc").read_bytes()))["items"]
    suffix = "".join(f"_{x}" for x in (args.model, args.mode) if x)
    out_path = pdir / f"phrasly{suffix}.enc"
    done: dict[str, str] = (
        json.loads(decrypt_at_rest(key, out_path.read_bytes())) if out_path.exists() else {}
    )
    todo = [i for i in range(len(items)) if str(i) not in done][: args.limit]
    opts = {k: v for k, v in (("model", args.model), ("mode", args.mode)) if v}
    client = Phrasly.build(s.phrasly_api_key, led, options=opts)
    results = client.humanize_all([items[i]["text"] for i in todo], [f"{args.period}:{i}" for i in todo])
    errors = {}
    for i, r in zip(todo, results, strict=True):
        if r.text:
            done[str(i)] = r.text
        else:
            errors[i] = r.error
    out_path.write_bytes(encrypt_at_rest(key, json.dumps(done).encode()))
    words_in = sum(len(items[int(i)]["text"].split()) for i in done)
    words_out = sum(len(t.split()) for t in done.values())
    print(
        json.dumps(
            {
                "done": len(done),
                "of": len(items),
                "errors": errors,
                "length_ratio": round(words_out / max(1, words_in), 3),
                "phrasly": led.summary().get("phrasly"),
            },
            indent=1,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
