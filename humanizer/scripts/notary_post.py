"""Post one notary remark (period/notary.py) with a dedicated wallet; reads the text on stdin, prints the
receipt as one JSON line after finalization. Needs only bittensor (run it with the validator env).

    echo 'sn32-humanizer/1 {...}' | btenv/bin/python scripts/notary_post.py --wallet humeval-notary \
        --path var/keys/wallets --network finney
    btenv/bin/python scripts/notary_post.py --wallet humeval-notary --path var/keys/wallets --create
    btenv/bin/python scripts/notary_post.py --wallet humeval-notary --path var/keys/wallets --balance --network finney

The wallet holds only fees (~τ0.0001 per remark); its coldkey has no password so the timer can sign.
"""

from __future__ import annotations

import argparse
import json
import sys

PREFIX = "sn32-humanizer/1 "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--network", default="finney")
    ap.add_argument("--create", action="store_true", help="create the wallet if missing, print its address")
    ap.add_argument("--balance", action="store_true", help="print the address and its balance")
    args = ap.parse_args()
    import bittensor as bt

    w = bt.Wallet(name=args.wallet, path=args.path)
    if args.create:
        if not w.coldkey_file.exists_on_device():
            w.create_new_coldkey(use_password=False, overwrite=False, suppress=True)
        print(json.dumps({"address": w.coldkeypub.ss58_address}))
        return 0
    st = bt.Subtensor(network=args.network)
    if args.balance:
        bal = st.get_balance(w.coldkeypub.ss58_address)
        print(
            json.dumps(
                {"address": w.coldkeypub.ss58_address, "balance_tao": float(bal), "network": args.network}
            )
        )
        return 0
    text = sys.stdin.read().strip()
    if not text.startswith(PREFIX) or len(text.encode()) > 1024:
        raise SystemExit("refusing: not a sn32-humanizer remark (or longer than 1 KiB)")
    call = st.substrate.compose_call(call_module="System", call_function="remark_with_event",
                                     call_params={"remark": text})  # fmt: skip
    r = st.sign_and_send_extrinsic(call, w, sign_with="coldkey", wait_for_inclusion=True,
                                   wait_for_finalization=True)  # fmt: skip
    if not r.success:
        raise SystemExit(f"remark not included: {r.message}")
    rc = r.extrinsic_receipt
    pk = w.coldkeypub.public_key
    print(json.dumps({"block": int(st.substrate.get_block_number(rc.block_hash)), "block_hash": rc.block_hash,
                      "extrinsic_index": int(rc.extrinsic_idx), "extrinsic_hash": rc.extrinsic_hash,
                      "signer": w.coldkeypub.ss58_address, "signer_pubkey": pk.hex() if isinstance(pk, bytes) else pk,
                      "fee_tao": float(r.extrinsic_fee) if r.extrinsic_fee is not None else None,
                      "network": args.network}))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
