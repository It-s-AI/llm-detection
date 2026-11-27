"""Runs ON THE POD next to a subtensor localnet (docker, ws://127.0.0.1:9944): checks that the validator's
mechanism-1 code (llm-detection detection/validator/humanizer_mechanism.py, uploaded next to this file)
puts the leaderboard weights on chain and that they earn incentive in mechanism 1.

Phases:
  A  commit-reveal off: mech-0 weights, then mech-1 weights from a signed leaderboard right after
     (same validator, same block window: is the rate limit shared?), read back, incentive per mechanism;
  B  commit-reveal on (as on SN32): mech-1 weights through set_mechanism_weights, wait for the reveal.
Writes /work/result.json.
"""

from __future__ import annotations

import json
import sys
import time
import traceback

sys.path.insert(0, "/work")

import bittensor as bt
import humanizer_mechanism as hm
from bittensor.core.extrinsics.sudo import (
    sudo_set_mechanism_count_extrinsic,
    sudo_set_mechanism_emission_split_extrinsic,
)
from bittensor.core.extrinsics.utils import sudo_call_extrinsic
from bittensor.utils.balance import Balance
from bittensor_wallet import Wallet
from nacl.signing import SigningKey

R: dict = {"steps": []}
WP = "/work/wallets"


def save():
    with open("/work/result.json", "w") as f:
        json.dump(R, f, indent=1, default=str)


def sudid(resp):
    """The inner result of a Sudo.sudo call (the outer extrinsic succeeds even when the call fails)."""
    try:
        for ev in resp.extrinsic_receipt.triggered_events:
            e = ev.get("event", ev) if isinstance(ev, dict) else ev
            e = getattr(e, "value", e)
            if isinstance(e, dict) and e.get("event_id") == "Sudid":
                return str(e.get("attributes"))[:300]
    except Exception as x:
        return f"n/a ({type(x).__name__})"
    return None


def step(name, resp=None, **extra):
    ok = None if resp is None else bool(getattr(resp, "success", resp))
    msg = None if resp is None else str(getattr(resp, "message", ""))[:400]
    if resp is not None and hasattr(resp, "extrinsic_receipt"):
        inner = sudid(resp)
        if inner is not None:
            extra["sudo_result"] = inner
            ok = ok and "Err" not in inner
    R["steps"].append({"step": name, "ok": ok, "msg": msg, "t": round(time.time() - T0, 1), **extra})
    print(json.dumps(R["steps"][-1], default=str), flush=True)
    save()
    return ok


def wallet(hotkey: str, uri: str, cold: str = "alice") -> Wallet:
    w = Wallet(name=cold, hotkey=hotkey, path=WP)
    w.create_hotkey_from_uri(uri, use_password=False, overwrite=True, suppress=True)
    return w


def wait_blocks(st, n):
    b = st.get_current_block()
    while st.get_current_block() < b + n:
        time.sleep(0.25)


def envelope(key, weights, now, netuid, ttl=3 * 3600):
    p = {"schema": hm.SCHEMA, "netuid": netuid, "mechid": 1, "seq": int(now * 1000), "issued_at": now,
         "expires_at": now + ttl, "weights": weights, "king": None, "history": []}  # fmt: skip
    body = json.dumps(p, sort_keys=True, separators=(",", ":"))
    return {"payload": body, "signature": key.sign(body.encode()).signature.hex(), "public_key": ""}


def max_uids(st, netuid):
    r = st.substrate.query("SubtensorModule", "MaxAllowedUids", [netuid])
    return getattr(r, "value", r)


def state(st, netuid):
    hp = st.get_subnet_hyperparameters(netuid)
    return {
        "tempo": hp.tempo,
        "weights_rate_limit": hp.weights_rate_limit,
        "commit_reveal": hp.commit_reveal_weights_enabled,
        "commit_reveal_period": hp.commit_reveal_period,
        "admin_freeze_window": st.get_admin_freeze_window(),
        "mechanisms": st.get_mechanism_count(netuid),
    }


def mech_view(st, netuid, mech):
    mg = st.metagraph(netuid, mechid=mech, lite=False)
    return {
        "incentive": [round(float(x), 4) for x in mg.incentive],
        "emission": [round(float(x), 6) for x in mg.emission],
        "weights": st.weights(netuid, mechid=mech),
    }


def main():
    st = bt.Subtensor(network="ws://127.0.0.1:9944")
    alice = Wallet(name="alice", hotkey="default", path=WP)
    alice.create_coldkey_from_uri("//Alice", use_password=False, overwrite=True, suppress=True)
    alice.create_hotkey_from_uri("//Alice", use_password=False, overwrite=True, suppress=True)
    # miners under another coldkey: the chain recycles miner emission of the subnet owner's own hotkeys
    bob = Wallet(name="bob", hotkey="default", path=WP)
    bob.create_coldkey_from_uri("//Bob", use_password=False, overwrite=True, suppress=True)
    val = wallet("val", "//Alice//val")
    m1, m2 = wallet("m1", "//Bob//m1", "bob"), wallet("m2", "//Bob//m2", "bob")
    R["chain_block0"] = st.get_current_block()

    step("freeze_window_0", sudo_call_extrinsic(st, alice, "sudo_set_admin_freeze_window", {"window": 0},
                                                ))  # fmt: skip
    step("network_rate_limit_0", sudo_call_extrinsic(st, alice, "sudo_set_network_rate_limit",
                                                     {"rate_limit": 0}))  # fmt: skip
    before = set(st.get_all_subnets_netuid())
    step("register_subnet", st.register_subnet(alice))
    netuid = max(set(st.get_all_subnets_netuid()) - before)
    R["netuid"] = netuid
    owner = st.get_subnet_owner_hotkey(netuid)
    R["owner_is_alice_hotkey"] = owner == alice.hotkey.ss58_address
    for _ in range(120):  # start_call waits for the start delay
        r = st.start_call(alice, netuid)
        if r.success:
            break
        time.sleep(1)
    step("start_call", r)
    step("tempo_20", sudo_call_extrinsic(st, alice, "sudo_set_tempo", {"netuid": netuid, "tempo": 20}))  # fmt: skip
    step("cr_off", sudo_call_extrinsic(st, alice, "sudo_set_commit_reveal_weights_enabled",
                                       {"netuid": netuid, "enabled": False}))  # fmt: skip
    step("rate_limit_100", sudo_call_extrinsic(st, alice, "sudo_set_weights_set_rate_limit",
                                               {"netuid": netuid, "weights_set_rate_limit": 100}))  # fmt: skip
    R["state_after_admin"] = state(st, netuid)
    R["max_uids_default"] = max_uids(st, netuid)
    step("mech_count_2_at_default_uids", sudo_set_mechanism_count_extrinsic(st, alice, netuid, 2))
    R["mechanism_count_at_default_uids"] = st.get_mechanism_count(netuid)
    if R["mechanism_count_at_default_uids"] < 2:
        step("max_uids_128", sudo_call_extrinsic(st, alice, "sudo_set_max_allowed_uids",
                                                 {"netuid": netuid, "max_allowed_uids": 128}))  # fmt: skip
        R["max_uids_after"] = max_uids(st, netuid)
        step("mech_count_2_at_128_uids", sudo_set_mechanism_count_extrinsic(st, alice, netuid, 2))
    step("mech_split_90_10", sudo_set_mechanism_emission_split_extrinsic(st, alice, netuid, [9, 1]))
    R["mechanism_count"] = st.get_mechanism_count(netuid)
    R["mechanism_split"] = st.get_mechanism_emission_split(netuid)
    for w in (val, m1, m2):
        step(f"register_{w.hotkey_str}", st.burned_register(w, netuid))
    step("stake_val", st.add_stake(alice, netuid, val.hotkey.ss58_address, Balance.from_tao(1000)))
    mg = st.metagraph(netuid)
    hk = list(mg.hotkeys)
    uid = {"owner": hk.index(owner), "val": hk.index(val.hotkey.ss58_address),
           "m1": hk.index(m1.hotkey.ss58_address), "m2": hk.index(m2.hotkey.ss58_address)}  # fmt: skip
    R["uids"] = uid
    # validator permit after an epoch
    for _ in range(240):
        mg = st.metagraph(netuid, lite=False)
        if bool(mg.validator_permit[uid["val"]]):
            break
        wait_blocks(st, 5)
    R["validator_permit"] = bool(mg.validator_permit[uid["val"]])
    step("validator_permit", None, permit=R["validator_permit"])

    key = SigningKey(b"\x07" * 32)
    pub = key.verify_key.encode().hex()
    served = {"w": {m2.hotkey.ss58_address: 0.85, hm.BURN_KEY: 0.15}}
    board = hm.HumanizerLeaderboard(["x"], pub, netuid, fetch=lambda u, t: envelope(key, served["w"], time.time(),
                                                                                   netuid))  # fmt: skip

    # --- phase A: commit-reveal off; the validator's order: mechanism 1, then mechanism 0 ------------
    R["state_phase_a"] = state(st, netuid)
    wait_blocks(st, 101)  # past the rate limit since registration

    def cycle(tag, block_time=12.0):
        ok1 = hm.set_mechanism_weights(board, st, val, st.metagraph(netuid), netuid, 0, block_time=block_time,
                                       info=lambda m: step(f"{tag}_info", None, info=m),
                                       error=lambda m: step(f"{tag}_error", None, error=m))  # fmt: skip
        step(f"{tag}_mech1_via_validator_code", ok1)
        step(f"{tag}_mech0_right_after", st.set_weights(val, netuid, [uid["m1"]], [1.0], mechid=0,
                                                        block_time=block_time, wait_for_inclusion=True,
                                                        wait_for_finalization=False))  # fmt: skip

    cycle("A")
    wait_blocks(st, 3)
    R["A_weights_mech0"] = st.weights(netuid, mechid=0)
    R["A_weights_mech1"] = st.weights(netuid, mechid=1)
    wait_blocks(st, 2 * 21 + 2)  # two epochs
    R["A_mech0"] = mech_view(st, netuid, 0)
    R["A_mech1"] = mech_view(st, netuid, 1)
    save()

    # --- the opposite order: what the SDK does with mechanism 1 right after mechanism 0 ------------------
    wait_blocks(st, 101)
    step("C_mech0_first", st.set_weights(val, netuid, [uid["m1"]], [1.0], mechid=0, wait_for_inclusion=True,
                                         wait_for_finalization=False))  # fmt: skip
    step("C_mech1_right_after_mech0", st.set_weights(val, netuid, [uid["m1"]], [1.0], mechid=1,
                                                     wait_for_inclusion=True,
                                                     wait_for_finalization=False))  # fmt: skip
    R["C_weights_mech1"] = st.weights(netuid, mechid=1)

    # --- phase B: commit-reveal on (as on SN32) --------------------------------------------------------
    step("cr_on", sudo_call_extrinsic(st, alice, "sudo_set_commit_reveal_weights_enabled",
                                      {"netuid": netuid, "enabled": True}))  # fmt: skip
    R["cr_enabled"] = st.commit_reveal_enabled(netuid)
    R["state_phase_b"] = state(st, netuid)
    wait_blocks(st, 101)
    served["w"] = {m1.hotkey.ss58_address: 0.6, m2.hotkey.ss58_address: 0.4}
    cycle("B", block_time=0.25)  # fast-block localnet
    revealed = False
    t_end = time.time() + 600
    while time.time() < t_end:
        w1 = dict(st.weights(netuid, mechid=1))
        row = dict(w1.get(uid["val"], []))
        if row.get(uid["m1"], 0) > 0 and row.get(uid["m2"], 0) > 0:
            revealed = True
            break
        wait_blocks(st, 10)
    R["B_revealed"] = revealed
    R["B_seconds_to_reveal"] = round(600 - (t_end - time.time()), 1)
    R["B_weights_mech1"] = st.weights(netuid, mechid=1)
    R["B_weights_mech0"] = st.weights(netuid, mechid=0)
    wait_blocks(st, 2 * 21 + 2)
    R["B_mech1"] = mech_view(st, netuid, 1)

    # --- phase D: what each hotkey actually receives (stake deltas over 5 epochs) ----------------------
    wait_blocks(st, 101)
    served["w"] = {m2.hotkey.ss58_address: 0.85, hm.BURN_KEY: 0.15}
    cycle("D", block_time=0.25)
    for _ in range(60):  # wait for the reveal of the new mechanism-1 weights
        row = dict(dict(st.weights(netuid, mechid=1)).get(uid["val"], []))
        if row.get(uid["m2"], 0) > 0 and row.get(uid["m1"], 0) == 0:
            break
        wait_blocks(st, 5)
    R["D_weights_mech1"] = st.weights(netuid, mechid=1)
    wait_blocks(st, 2 * 21)
    keys = {"owner": owner, "val": val.hotkey.ss58_address, "m1": m1.hotkey.ss58_address,
            "m2": m2.hotkey.ss58_address}  # fmt: skip

    cold = {"owner": alice, "val": alice, "m1": bob, "m2": bob}

    def stakes():
        return {k: float(st.get_stake(cold[k].coldkey.ss58_address, h, netuid)) for k, h in keys.items()}

    b0, s0 = st.get_current_block(), stakes()
    wait_blocks(st, 5 * 21)
    b1, s1 = st.get_current_block(), stakes()
    R["D_blocks"] = b1 - b0
    R["D_stake_delta"] = {k: round(s1[k] - s0[k], 6) for k in keys}
    d = R["D_stake_delta"]
    R["D_m2_over_m1"] = round(d["m2"] / d["m1"], 4) if d["m1"] else None
    R["D_expected_m2_over_m1"] = round((0.10 * 0.85) / (0.90 * 1.0), 4)
    step("done")


if __name__ == "__main__":
    T0 = time.time()
    try:
        main()
    except Exception:
        R["fatal"] = traceback.format_exc()[-3000:]
        save()
        raise
