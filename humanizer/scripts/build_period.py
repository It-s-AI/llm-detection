"""Build one production period (D-50..D-57): n hidden AI texts generated from the pool's briefs and the
matched human reference texts, saved encrypted in var/periods/<id>/.

    python scripts/build_period.py --new-secret var/periods/.p1.secret          # prints c = H(secret)
    # publish c (docs/periods.md, pushed), then take the hash of a later finney block B:
    python scripts/build_period.py --period-id p1 --n 500 --secret-file var/periods/.p1.secret \
        --block 9527000 --block-hash 0x... [--current]

Seed (D-57): seed = H(secret || blockhash(B)), B produced after c was published, so neither we nor
anyone else could steer the period. The secret moves to <period>/secret.hex (mode 600) until the period
is revealed. An existing period directory is never overwritten.

Costs: generation ~$0.6 per 100 texts on OpenRouter (ledger caps apply). Scoring the human texts and the
owner's baseline king is a separate step (evaluate_submission.py --king-only), so this script spends
nothing on detectors.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from humanizer_eval.budget import SpendLedger
from humanizer_eval.config import get_settings
from humanizer_eval.llm.openrouter import ModelPricing, OpenRouter
from humanizer_eval.paths import ledger_path, var_dir
from humanizer_eval.period.build import build_period, save_period
from humanizer_eval.period.crypto import load_or_create_key
from humanizer_eval.period.generate import GenConfig
from humanizer_eval.period.pool_adapter import PoolAdapter
from humanizer_eval.period.seed import commitment, make_secret


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-secret", type=Path, default=None, help="only write a new secret there, print c")
    ap.add_argument("--period-id")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--pool", default="pool-full-v2", help="pool dir in var/")
    ap.add_argument("--block-hash", help="hash of block B (produced after the commitment was published)")
    ap.add_argument("--block", type=int, default=None, help="its number, for the manifest")
    ap.add_argument("--secret-file", type=Path, help="the secret whose commitment was published")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--current", action="store_true", help="make it the current period (periods/CURRENT)")
    args = ap.parse_args()
    if args.new_secret is not None:
        if args.new_secret.exists():
            raise SystemExit(f"{args.new_secret} exists")
        sec = make_secret()
        args.new_secret.parent.mkdir(parents=True, exist_ok=True)
        args.new_secret.write_text(sec.hex())
        args.new_secret.chmod(0o600)
        print(commitment(sec))
        return 0
    if not (args.period_id and args.block_hash and args.secret_file):
        raise SystemExit("give --period-id, --secret-file and --block-hash")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,40}", args.period_id):
        raise SystemExit("period id: lowercase letters, digits and dashes")
    if not re.fullmatch(r"(0x)?[0-9a-f]{64}", args.block_hash):
        raise SystemExit("--block-hash: 32-byte hex")
    out_dir = var_dir() / "periods" / args.period_id
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; a period is never rebuilt")

    s = get_settings()
    led = SpendLedger(ledger_path(), s.caps())
    llm = OpenRouter(s.openrouter_api_key, led, pricing=ModelPricing(var_dir() / "openrouter_models.json"),
                     log_path=var_dir() / "llm_calls.jsonl")  # fmt: skip
    cfg = GenConfig.from_dict(json.loads(Path("configs/period.json").read_text()))
    secret = bytes.fromhex(args.secret_file.read_text().strip())
    print("commitment c = H(secret):", commitment(secret), flush=True)
    pool = PoolAdapter.load(var_dir() / args.pool)
    spent0 = {k: v["spent"] for k, v in led.summary().items()}
    res = build_period(args.period_id, secret, args.block_hash.removeprefix("0x"), pool, llm, cfg, n=args.n,
                       workers=args.workers)  # fmt: skip
    res.manifest["block"] = args.block
    res.manifest["pool"] = args.pool
    save_period(res, out_dir, load_or_create_key(var_dir() / "keys" / "period.key"))
    sp = out_dir / "secret.hex"
    sp.write_text(secret.hex())
    sp.chmod(0o600)
    args.secret_file.unlink()
    if args.current:
        (var_dir() / "periods" / "CURRENT").write_text(args.period_id + "\n")
    spent = {k: round(v["spent"] - spent0.get(k, 0.0), 3) for k, v in led.summary().items()
             if v["spent"] - spent0.get(k, 0.0) > 0}  # fmt: skip
    print(json.dumps({"period": args.period_id, "dir": str(out_dir), "spent": spent,
                      "manifest": {k: v for k, v in res.manifest.items() if k != "config_sha256"}},
                     indent=1, default=str))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
