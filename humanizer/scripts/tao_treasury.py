"""The payments treasury (billing/tao.py): what it holds on Base, what miners are owed, withdrawals.

    python scripts/tao_treasury.py                          # status (read-only)
    python scripts/tao_treasury.py --withdraw-usdc 500 --to 0xYourBaseAddress

The treasury is one EVM account (var/keys/tao_treasury.key, created by the intake server), the same address
on Bittensor EVM and on Base. On Base it receives the USDC of every TAO deposit, and it pays the refunds:
their swap back to TAO, their gas and their bridge fee in ETH (~0.0023 ETH each, charged to the miner in
USDC). Keep ~0.02 ETH on it on Base, and ~0.02 TAO on it on Bittensor EVM: the bridge books a refund's
TAO for the treasury to claim (claimLiquid) and send on to the miner's ss58 address, and that gas is paid
there (charged to the miner in USDC too). Miners' balances are USDC the treasury owes: withdraw only what is
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
from humanizer_eval.billing.tao import _REFUND_OPEN
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


def refunds_unbought_usdc(tao_db: Path) -> float:
    if not tao_db.exists():
        return 0.0
    c = sqlite3.connect(tao_db)
    cents = c.execute(
        "SELECT COALESCE(SUM(cents), 0) FROM refunds WHERE stage IN ('new', 'buying')"
    ).fetchone()[0]
    return cents / 100


def payout(route, out: dict, s, state: Path) -> int:
    """Send what the treasury earned (above miners' money and the reserve) to PAYOUT_ADDRESS. One transfer
    at a time: while the previous one has no receipt, nothing more is sent."""
    from humanizer_eval.ops.alerts import from_settings

    to = s.payout_address
    if not to:
        print("payout: off (PAYOUT_ADDRESS empty)")
        return 0
    if not (to.startswith("0x") and len(to) == 42):
        raise SystemExit("PAYOUT_ADDRESS: a Base address 0x...")
    if "free_usdc" not in out:
        print("payout: no billing database here")
        return 0
    # money in flight is the miners' even before the books show it (a swap mined but not yet credited, a
    # refund that has not bought its TAO or got stuck): no payout until nothing is in flight
    busy = out.get("deposits_in_progress", 0) + out.get("refunds_in_progress", 0) + out.get("stuck", 0)
    if busy:
        print(f"payout: {busy} deposit(s)/refund(s) in flight or stuck; waiting")
        return 0
    last = state / "payout_last.json"
    if last.exists():
        prev = json.loads(last.read_text())
        if route.base.receipt(prev["tx"]) is None:
            print(f"payout: the previous transfer {prev['tx']} has no receipt yet")
            return 0
    amount = round(out["free_usdc"] - s.payout_reserve_usdc, 2)
    if amount < s.payout_min_usdc:
        print(f"payout: {amount} USDC above the reserve, under the minimum {s.payout_min_usdc}")
        return 0
    tx = route.usdc_transfer_tx(to, round(amount * 1e6))
    last.write_text(json.dumps({"tx": tx.hash, "usdc": amount, "to": to}))  # before it is broadcast
    alerts = from_settings(state_path=var_dir() / "ops" / "alerts_state.json")
    try:
        route.send("base", tx)
    except Exception as e:
        # it may have been broadcast before the error: the marker stays, so no later payout sends the same
        # earnings again; the owner checks the tx and deletes payout_last.json if it never went out
        alerts.send(f"payout of {amount} USDC: send failed ({type(e).__name__}: {e}); payouts wait. Check tx "
                    f"{tx.hash} on Base; if it never went out, delete {last}", dedupe="payout-fail")  # fmt: skip
        raise
    print(f"payout: sent {amount} USDC to {to}: {tx.hash}")
    alerts.send(f"payout: {amount} USDC sent to {to} on Base (tx {tx.hash})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state-dir", type=Path, default=None, help="as the intake server's (default var/intake)"
    )
    ap.add_argument(
        "--withdraw-usdc", type=float, default=None, help="USDC to send from the treasury on Base"
    )
    ap.add_argument("--to", default=None, help="Base address receiving the withdrawal")
    ap.add_argument("--payout", action="store_true",
                    help="send the earnings above the reserve to PAYOUT_ADDRESS (for a timer; no prompt)")  # fmt: skip
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
        # refunds already subtracted from the balances but not bought yet still need their USDC here
        out["refunds_unbought_usdc"] = refunds_unbought_usdc(state / "tao.sqlite")
        out["free_usdc"] = round(
            out["treasury"]["base_usdc"] - out["owed_to_miners_usdc"] - out["refunds_unbought_usdc"], 2
        )
    if (state / "tao.sqlite").exists():
        c = sqlite3.connect(state / "tao.sqlite")
        out["deposits_in_progress"] = c.execute(
            "SELECT COUNT(*) FROM inbound WHERE stage IN ('bridging','on_base','swapping')").fetchone()[0]  # fmt: skip
        out["refunds_in_progress"] = c.execute(
            f"SELECT COUNT(*) FROM refunds WHERE stage IN {_REFUND_OPEN}"
        ).fetchone()[0]
        out["stuck"] = c.execute("SELECT COUNT(*) FROM inbound WHERE stage = 'stuck'").fetchone()[0] + c.execute(
            "SELECT COUNT(*) FROM refunds WHERE stage = 'stuck'").fetchone()[0]  # fmt: skip
    print(json.dumps(out, indent=1))
    if args.payout:
        return payout(route, out, s, state)
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
