"""Runs ON THE POD next to a subtensor localnet: the launch steps for SN32 done by a subnet owner who is
NOT the chain's sudo key (as on finney). Bob owns the subnet; Alice (sudo) only sets the test tempo.

  1. owner sets max_allowed_uids 256 -> 128 (directly, not through Sudo.sudo)
  2. owner sets the mechanism count to 2
  3. owner sets the emission split 90/10
and what the bittensor helpers do when used as-is (they wrap the call in Sudo.sudo).
Writes /work/result.json.
"""

from __future__ import annotations

import json
import time
import traceback

import bittensor as bt
from bittensor.core.extrinsics.sudo import (
    sudo_set_mechanism_count_extrinsic,
    sudo_set_mechanism_emission_split_extrinsic,
)
from bittensor.core.extrinsics.utils import sudo_call_extrinsic
from bittensor.utils.weight_utils import convert_maybe_split_to_u16
from bittensor_wallet import Wallet

R: dict = {"steps": []}
WP = "/work/wallets"


def save():
    with open("/work/result.json", "w") as f:
        json.dump(R, f, indent=1, default=str)


def inner(resp):
    try:
        for ev in resp.extrinsic_receipt.triggered_events:
            e = ev.get("event", ev) if isinstance(ev, dict) else ev
            if isinstance(e, dict) and e.get("event_id") == "Sudid":
                return str(e.get("attributes"))[:300]
    except Exception:
        return None
    return None


def step(name, resp, **extra):
    ok = bool(getattr(resp, "success", resp))
    sr = inner(resp)
    if sr is not None:
        ok = ok and "Err" not in sr
    R["steps"].append({"step": name, "ok": ok, "msg": str(getattr(resp, "message", ""))[:300],
                       "sudo_result": sr, "t": round(time.time() - T0, 1), **extra})  # fmt: skip
    print(json.dumps(R["steps"][-1], default=str), flush=True)
    save()
    return ok


def max_uids(st, n):
    r = st.substrate.query("SubtensorModule", "MaxAllowedUids", [n])
    return getattr(r, "value", r)


def wait_blocks(st, n):
    b = st.get_current_block()
    while st.get_current_block() < b + n:
        time.sleep(0.25)


def owner_call(st, owner, fn, params):
    """Signed by the owner's coldkey and sent as is (bittensor: root_call=True means 'no Sudo.sudo')."""
    return sudo_call_extrinsic(st, owner, fn, params, root_call=True)


def main():
    st = bt.Subtensor(network="ws://127.0.0.1:9944")
    alice = Wallet(name="alice", hotkey="default", path=WP)
    alice.create_coldkey_from_uri("//Alice", use_password=False, overwrite=True, suppress=True)
    alice.create_hotkey_from_uri("//Alice", use_password=False, overwrite=True, suppress=True)
    bob = Wallet(name="bob", hotkey="default", path=WP)
    bob.create_coldkey_from_uri("//Bob", use_password=False, overwrite=True, suppress=True)
    bob.create_hotkey_from_uri("//Bob", use_password=False, overwrite=True, suppress=True)
    sudo_call_extrinsic(st, alice, "sudo_set_network_rate_limit", {"rate_limit": 0})
    before = set(st.get_all_subnets_netuid())
    step("bob_registers_subnet", st.register_subnet(bob))
    n = max(set(st.get_all_subnets_netuid()) - before)
    R["netuid"] = n
    R["owner_is_bob"] = st.get_subnet_owner_hotkey(n) == bob.hotkey.ss58_address
    for _ in range(120):
        r = st.start_call(bob, n)
        if r.success:
            break
        time.sleep(1)
    step("start_call", r)
    # as on finney: tempo long enough for the 10-block admin freeze window (finney: tempo 360, window 10)
    step("sudo_freeze_0", sudo_call_extrinsic(st, alice, "sudo_set_admin_freeze_window", {"window": 0}))
    step("sudo_tempo_40", sudo_call_extrinsic(st, alice, "sudo_set_tempo", {"netuid": n, "tempo": 40}))
    step("sudo_freeze_10", sudo_call_extrinsic(st, alice, "sudo_set_admin_freeze_window", {"window": 10}))
    R["tempo"] = st.tempo(n)
    R["admin_freeze_window"] = st.get_admin_freeze_window()
    R["max_uids_start"] = max_uids(st, n)
    R["mechanisms_start"] = st.get_mechanism_count(n)

    # what the stock helper does for a non-sudo owner (wraps in Sudo.sudo)
    step("helper_mech_count_2_as_owner", sudo_set_mechanism_count_extrinsic(st, bob, n, 2))
    R["mechanisms_after_helper"] = st.get_mechanism_count(n)

    # the launch steps, outside the admin freeze window at the end of a tempo
    def outside_window():
        for _ in range(200):
            if not st.is_in_admin_freeze_window(n):
                return
            wait_blocks(st, 1)

    outside_window()
    step("owner_max_uids_128", owner_call(st, bob, "sudo_set_max_allowed_uids",
                                          {"netuid": n, "max_allowed_uids": 128}))  # fmt: skip
    R["max_uids_after"] = max_uids(st, n)
    for attempt in range(6):  # an owner rate limit between hyperparameter changes, if any
        outside_window()
        ok = step(f"owner_mech_count_2_try{attempt}", owner_call(st, bob, "sudo_set_mechanism_count",
                                                                {"netuid": n, "mechanism_count": 2}))  # fmt: skip
        if st.get_mechanism_count(n) == 2:
            break
        if not ok:
            wait_blocks(st, 25)
    R["mechanisms_after_owner"] = st.get_mechanism_count(n)
    for attempt in range(6):
        outside_window()
        split = convert_maybe_split_to_u16([9, 1])
        step(f"owner_split_90_10_try{attempt}", owner_call(st, bob, "sudo_set_mechanism_emission_split",
                                                          {"netuid": n, "maybe_split": split}))  # fmt: skip
        if st.get_mechanism_emission_split(n) == [90, 10]:
            break
        wait_blocks(st, 25)
    R["split_after_owner"] = st.get_mechanism_emission_split(n)
    step("helper_split_as_owner", sudo_set_mechanism_emission_split_extrinsic(st, bob, n, [8, 2]))
    R["split_after_helper"] = st.get_mechanism_emission_split(n)
    save()


if __name__ == "__main__":
    T0 = time.time()
    try:
        main()
    except Exception:
        R["fatal"] = traceback.format_exc()[-3000:]
        save()
        raise
