"""The payments treasury (billing/tao.py): what it holds on Base, what miners are owed, withdrawals.

    python scripts/tao_treasury.py                          # status (read-only)
    python scripts/tao_treasury.py --withdraw-usdc 500 --to 0xYourBaseAddress

The treasury is one EVM account (var/keys/tao_treasury.key, created by the intake server), the same address
on Bittensor EVM and on Base. On Base it receives the USDC of every TAO deposit, and it pays the refunds:
their swap back to TAO, their gas and their bridge fee in ETH (~0.0023 ETH each, charged to the miner in
USDC). Keep ~0.02 ETH on it on Base. Miners' balances are USDC the treasury owes: withdraw only what is
above them (the status shows both); native USDC on Base can go straight to an exchange such as Coinbase.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from humanizer_eval.billing.evm import ForeverMoneyBaseRoute, RouteConfig
from humanizer_eval.config import get_settings
from humanizer_eval.paths import var_dir


def owed_usdc(billing_db: Path) -> float:
    """All miners' balances (payments - charges - refunds), USDC."""
    c = sqlite3.connect(billing_db)
    paid = c.execute("SELECT COALESCE(SUM(net_cents), 0) FROM payments").fetchone()[0]
    charged = c.execute(
        "SELECT COALESCE(SUM(charged_cents), 0) FROM holds WHERE status = 'settled'"
    ).fetchone()[0]
    refunded = c.execute(
        "SELECT COALESCE(SUM(amount_cents), 0) FROM refunds WHERE status != 'failed'"
    ).fetchone()[0]
    return (paid - charged - refunded) / 100


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir", type=Path, default=None, help="as the intake server's (default var/intake)"
    )
    ap.add_argument(
        "--withdraw-usdc", type=float, default=None, help="USDC to send from the treasury on Base"
    )
    ap.add_argument("--to", default=None, help="Base address receiving the withdrawal")
    args = ap.parse_args()
    kp = var_dir() / "keys" / "tao_treasury.key"
    if not kp.exists():
        raise SystemExit(f"{kp} not found: start the intake server once")
    s = get_settings()
    rc = RouteConfig(**{k: v for k, v in (("bittensor_rpc", s.tao_rpc), ("base_rpc", s.base_rpc)) if v})
    route = ForeverMoneyBaseRoute(bytes.fromhex(kp.read_text().strip()), rc)
    state = args.state_dir or var_dir() / "intake"
    out = {"treasury": route.treasury_balances(), "tao_usdc": round(route.price(), 2),
           "eth_usdc": round(route.eth_usd(), 2), "bridge_ready": route.bridge_ready()}  # fmt: skip
    if (state / "billing.sqlite").exists():
        out["owed_to_miners_usdc"] = owed_usdc(state / "billing.sqlite")
        out["free_usdc"] = round(out["treasury"]["base_usdc"] - out["owed_to_miners_usdc"], 2)
    if (state / "tao.sqlite").exists():
        c = sqlite3.connect(state / "tao.sqlite")
        out["deposits_in_progress"] = c.execute(
            "SELECT COUNT(*) FROM inbound WHERE stage IN ('bridging','on_base','swapping')").fetchone()[0]  # fmt: skip
        out["refunds_in_progress"] = c.execute(
            "SELECT COUNT(*) FROM refunds WHERE stage IN ('new','buying','bought','bridging','sent')").fetchone()[0]  # fmt: skip
        out["stuck"] = c.execute("SELECT COUNT(*) FROM inbound WHERE stage = 'stuck'").fetchone()[0] + c.execute(
            "SELECT COUNT(*) FROM refunds WHERE stage = 'stuck'").fetchone()[0]  # fmt: skip
    print(json.dumps(out, indent=1))
    if args.withdraw_usdc is not None:
        if not args.to or not args.to.startswith("0x") or len(args.to) != 42:
            raise SystemExit("--to: a Base address 0x...")
        if args.withdraw_usdc > out.get("free_usdc", out["treasury"]["base_usdc"]):
            raise SystemExit("that is more than the treasury holds above what miners are owed")
        units = round(args.withdraw_usdc * 1e6)
        if input(f"send {units / 1e6} USDC to {args.to} on Base? type yes: ").strip() != "yes":
            return 1
        tx = route.usdc_transfer_tx(args.to, units)
        route.send("base", tx)
        print(f"sent: {tx.hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
