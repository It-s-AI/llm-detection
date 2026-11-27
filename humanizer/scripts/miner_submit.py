"""Submit a humanizer to the evaluation API (needs `pip install "bittensor==10.5.0" huggingface_hub`:
bittensor 11 changed its API; 10.5.0 is the version the SN32 validators run).

One command does it all: upload bundle.tar to your Hugging Face model repo, commit
"<repo>@<commit sha>" on chain from your hotkey, then submit (signed with the same hotkey):

    python scripts/miner_submit.py --api https://eval.example --wallet miner --hotkey h1 \
        publish --repo you/my-humanizer --file bundle.tar [--private --hf-token hf_...read-only...]

Or step by step: upload, commit, submit.

Money is USDC on your balance; you pay in TAO. If the balance does not cover the quote, the command prints
your hotkey's deposit address and the TAO to send before the deadline (the quote's lifetime, 60 min);
`--send` sends it from your wallet's coldkey right away (or use `btcli wallet transfer`, any wallet or an
exchange). It is bridged to Base and swapped to USDC (a few minutes); then:

    python scripts/miner_submit.py ... paid sub-0123456789abcdef [--wait]   # joins the queue once credited
    python scripts/miner_submit.py ... status sub-0123456789abcdef --wait
    python scripts/miner_submit.py ... deposit                     # your address, the rate, the balance
    python scripts/miner_submit.py ... balance
    python scripts/miner_submit.py ... refund --key my-refund-1 [--amount-usdc 20] [--quote]

A refund sends USDC of your balance back in TAO, at the rate of that moment, to `--to` (any ss58 address;
default: your wallet's coldkey); the cost of the way back (bridge fee, swap) comes out of it.

Rules (docs/billing-policy.md): only the exact commit you put on chain is evaluated (a branch such as
"main" is never used, so pushing to the repo later changes nothing); one submission per registration of
the hotkey; keep the commit downloadable until your result is final: a crown counts only after we have
copied the bundle, and if you win, the bundle is published from our copy 48 h later.
"""

from __future__ import annotations

import argparse
import json
import time

from humanizer_eval.intake.client import MinerClient, SeedSigner, file_digest
from humanizer_eval.intake.hf import BUNDLE_PATH, commitment_for

NETUID = 32


def wallet(args):
    from bittensor_wallet import Wallet

    return Wallet(name=args.wallet, hotkey=args.hotkey)


def signer(args):
    if args.dev_seed:
        return SeedSigner(bytes.fromhex(args.dev_seed))
    return wallet(args).hotkey


def upload(repo: str, path: str, token: str | None, private: bool) -> str:
    """Upload bundle.tar and return the full sha of the commit that contains it."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    except Exception:  # a token limited to existing repos may not create one: upload into it as it is
        if not api.repo_exists(repo, repo_type="model"):
            raise
    info = api.upload_file(path_or_fileobj=path, path_in_repo=BUNDLE_PATH, repo_id=repo, repo_type="model",
                           commit_message="humanizer bundle")  # fmt: skip
    return info.oid


def _bt():
    import bittensor as bt

    if not hasattr(bt.Subtensor, "set_commitment"):
        raise SystemExit(f"bittensor {getattr(bt, '__version__', '?')} is not supported (its API changed): "
                         'pip install "bittensor==10.5.0"')  # fmt: skip
    return bt


def commit_on_chain(args, repo: str, commit: str) -> str:
    bt = _bt()
    data = commitment_for(repo, commit)
    st = bt.Subtensor(network=args.network)
    r = st.set_commitment(wallet(args), args.netuid, data)
    if not r.success:
        raise SystemExit(f"commitment failed: {r.message}")
    return data


def send_tao(args, dest: str, amount_tao: str) -> None:
    """Transfer the TAO from the wallet's coldkey (asks for its password)."""
    bt = _bt()

    st = bt.Subtensor(network=args.network)
    r = st.transfer(wallet(args), dest, bt.Balance.from_tao(float(amount_tao)))
    if not getattr(r, "success", r):
        raise SystemExit(f"transfer failed: {getattr(r, 'message', r)}")


def payment_hint(args, out: dict) -> None:
    """The deposit address, the TAO to send, the deadline and the command to run after (on stderr)."""
    import sys

    pay = out.get("payment") or {}
    if out.get("state") != "awaiting_payment" or not pay.get("amount_tao"):
        return
    exp = pay.get("expires_ts")
    left = f" ({max(0, int((exp - time.time()) // 60))} min left)" if exp else ""
    until = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(exp)) if exp else "the deadline"
    who = f"--api {args.api} --wallet {args.wallet} --hotkey {args.hotkey}"
    print(f"\nYour balance lacks {pay['due_usdc']} USDC. Send {pay['amount_tao']} TAO by {until}{left} to your "
          f"deposit address\n  {pay['address']}\n(the rate now plus a {pay.get('buffer', 0):.0%} buffer and the "
          f"bridge's cost; what is left over stays on your balance), e.g.\n"
          f"  btcli wallet transfer --dest {pay['address']} --amount {pay['amount_tao']}\n"
          f"then run:\n  python scripts/miner_submit.py {who} paid {out['submission_id']}\n",
          file=sys.stderr)  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--wallet", default="default")
    ap.add_argument("--hotkey", default="default")
    ap.add_argument("--network", default="finney")
    ap.add_argument("--netuid", type=int, default=NETUID)
    ap.add_argument("--dev-seed", help="hex 32-byte seed instead of a wallet (tests only)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    pub = sp.add_parser("publish", help="upload + commit on chain + submit")
    up = sp.add_parser("upload", help="upload bundle.tar, print the commit sha")
    for p in (pub, up):
        p.add_argument("--repo", required=True, help="your model repo, e.g. you/my-humanizer")
        p.add_argument("--file", required=True)
        p.add_argument("--private", action="store_true")
        p.add_argument("--hf-write-token", default=None, help="for the upload (default: your HF login)")
    pub.add_argument("--hf-token", default=None, help="read-only token for a private repo (given to us)")
    pub.add_argument("--name", default="")
    pub.add_argument("--wait", action="store_true")

    cm = sp.add_parser("commit", help="put <repo>@<commit> on chain")
    cm.add_argument("--repo", required=True)
    cm.add_argument("--revision", required=True, help="full 40-hex commit sha")
    sh = sp.add_parser("submit", help="submit what your hotkey has committed on chain")
    sh.add_argument("--commitment", default=None, help="what you committed (detects a lagging snapshot)")
    sh.add_argument("--hf-token", default=None)
    sh.add_argument("--name", default="")
    sh.add_argument("--wait", action="store_true")
    for p in (pub, sh):
        p.add_argument("--send", action="store_true",
                       help="if the balance is short, send the TAO from your wallet's coldkey right away")  # fmt: skip
    pd = sp.add_parser("paid", help="after sending the TAO: check it, join the queue once credited")
    pd.add_argument("submission_id")
    pd.add_argument("--wait", action="store_true")
    st = sp.add_parser("status")
    st.add_argument("submission_id")
    st.add_argument("--wait", action="store_true")
    sp.add_parser("info")
    sp.add_parser("balance")
    sp.add_parser("deposit", help="your hotkey's TAO deposit address, the rate, the balance")
    c = sp.add_parser("cancel")
    c.add_argument("submission_id")
    r = sp.add_parser("refund")
    r.add_argument("--key", required=True)
    r.add_argument("--amount-usdc", type=float, default=None, help="default: everything available")
    r.add_argument("--quote", action="store_true", help="only show what would be sent now")
    r.add_argument(
        "--to", default=None, help="ss58 address receiving the TAO (default: your wallet's coldkey)"
    )
    rs = sp.add_parser("refund-status")
    rs.add_argument("--key", required=True)
    stm = sp.add_parser("statement")
    stm.add_argument("submission_id")
    args = ap.parse_args()

    if args.cmd == "upload":
        print(upload(args.repo, args.file, args.hf_write_token, args.private))
        return 0
    if args.cmd == "commit":
        print(commit_on_chain(args, args.repo, args.revision))
        return 0
    cli = MinerClient(args.api, None if args.cmd == "info" else signer(args))
    if args.cmd == "info":
        out = cli.info()
    elif args.cmd in ("publish", "submit"):
        commitment = getattr(args, "commitment", None)
        if args.cmd == "publish":
            sha, size = file_digest(args.file)
            commit = upload(args.repo, args.file, args.hf_write_token, args.private)
            commitment = commit_on_chain(args, args.repo, commit)
            print(f"uploaded {size} bytes (sha256 {sha}) as {commitment}")
        out = cli.submit_hf(commitment, args.hf_token, args.name)
        payment_hint(args, out)
        pay = out.get("payment") or {}
        if args.send and out.get("state") == "awaiting_payment" and pay.get("amount_tao"):
            send_tao(args, pay["address"], pay["amount_tao"])
            time.sleep(15)  # a block or two
            out = cli.paid(out["submission_id"])
        if args.wait:
            out = cli.wait(out["submission_id"])
    elif args.cmd == "paid":
        out = cli.paid(args.submission_id)
        payment_hint(args, out)
        if args.wait and out["state"] not in ("awaiting_payment", "expired", "cancelled"):
            out = cli.wait(args.submission_id)
    elif args.cmd == "status":
        out = cli.wait(args.submission_id) if args.wait else cli.status(args.submission_id)
    elif args.cmd == "statement":
        out = cli.statement(args.submission_id)
    elif args.cmd == "balance":
        out = cli.balance()
    elif args.cmd == "deposit":
        out = cli.deposit()
    elif args.cmd == "refund-status":
        out = cli.refund_status(args.key)
    elif args.cmd == "cancel":
        out = cli.cancel(args.submission_id)
    else:
        to = args.to or wallet(args).coldkeypub.ss58_address
        out = (
            cli.refund_quote(to, args.amount_usdc)
            if args.quote
            else cli.refund(args.key, to, args.amount_usdc)
        )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
