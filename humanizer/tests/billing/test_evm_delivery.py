"""CCIP delivery tracking (billing/evm.py): accepted only from an OffRamp the destination router trusts."""

from eth_abi import encode

from humanizer_eval.billing.evm import _EXEC_STATE, ForeverMoneyBaseRoute, RouteConfig, selector

MID = "0x" + "c6" * 32
GOOD, EVIL = "0x16e577f1724ae2598f9b43a52c19e6f67ee13808", "0x" + "66" * 20
OLD = "0xf09afe78d3c7d359b334d7cb88995751f7ec5e13"
SRC = RouteConfig().bittensor_selector  # deliveries to Base come from Bittensor
BASE = RouteConfig().fm_base_selector  # and deliveries to Bittensor from Base


class FakeChain:
    def __init__(self, logs):
        self.logs, self.queries, self.calls = logs, [], 0

    def block_number(self):
        return 25_000

    def rpc(self, method, q):
        assert method == "eth_getLogs" and set(q["address"]) == {GOOD, OLD}
        self.queries.append(q)
        lo, hi = int(q["fromBlock"], 16), int(q["toBlock"], 16)
        # an endpoint that ignores the address filter: the route checks the address itself
        return [lg for lg in self.logs if lo <= lg["block"] <= hi and lg["topics"][3] == q["topics"][3]]

    def call(self, to, data):
        self.calls += 1
        assert data.removeprefix("0x").startswith(selector("getOffRamps()").hex())
        # the router's list: two OffRamps for our source (a CCIP upgrade keeps the old one), others
        pairs = [(SRC, GOOD), (7, EVIL), (SRC, OLD), (SRC, GOOD), (BASE, GOOD), (BASE, OLD)]
        return "0x" + encode(["(uint64,address)[]"], [pairs]).hex()


def log(addr, state, block=21_000, v16=True):
    data = ("00" * 32 + format(state, "064x")) if v16 else format(state, "064x")
    sig = _EXEC_STATE[0] if v16 else _EXEC_STATE[1]
    return {"address": addr, "block": block, "topics": [sig, "0x1", "0x2", MID], "data": "0x" + data}


def route(chain):
    r = ForeverMoneyBaseRoute.__new__(ForeverMoneyBaseRoute)
    r.cfg, r.base, r.bt_logs = RouteConfig(), chain, chain
    return r


def test_delivery_from_a_trusted_offramp_in_either_event_shape():
    for v16 in (True, False):
        assert route(FakeChain([log(GOOD, 2, v16=v16)])).arrived_on_base(MID, 1_000) is True
        assert route(FakeChain([log(GOOD, 3, v16=v16)])).arrived_on_base(MID, 1_000) is False


def test_a_fake_event_from_another_contract_is_ignored():
    chain = FakeChain([log(EVIL, 2)])
    assert route(chain).arrived_on_base(MID, 1_000) is None
    assert len(chain.queries) == 3  # windows of 10k blocks, up to the head


def test_not_yet_and_a_delivery_by_the_older_offramp():
    chain = FakeChain([log(GOOD, 1), log(GOOD, 1, block=22_000)])
    assert route(chain).arrived_on_base(MID, 1_000) is None
    assert route(FakeChain([log(OLD, 2)])).arrived_on_base(MID, 1_000) is True


def test_finney_outcome_reads_only_this_messages_part_of_a_batched_execution():
    from humanizer_eval.billing.evm import _DELIVERED_LIQUID

    claimable = "0x16a772d7" + "00" * 28

    def li(i, t0):
        return {"logIndex": hex(i), "topics": [t0]}

    for before, want in (([_DELIVERED_LIQUID], "liquid"), ([claimable], "claimable")):
        ours = {**log(GOOD, 2), "transactionHash": "0xexec", "logIndex": hex(5)}
        # another message executed first in the same tx: its liquid delivery must not count for ours
        other_liquid = [li(0, _DELIVERED_LIQUID), li(1, _EXEC_STATE[1])]
        mine = [li(2 + k, t) for k, t in enumerate(before)]
        chain = FakeChain([ours])
        chain.receipt = lambda h, logs=other_liquid + mine + [ours]: {"logs": logs} if h == "0xexec" else None
        r = route(chain)
        r.bt = chain
        assert r.finney_outcome(MID, 1_000) == want


def test_a_refund_exit_moves_whole_rao_and_asks_a_little_under_it():
    r = ForeverMoneyBaseRoute.__new__(ForeverMoneyBaseRoute)
    r.cfg, r.treasury = RouteConfig(), "0x" + "54" * 20
    _, amount, (_, fallback, liquid, min_out), _ = r._exit_args(61_909_322 * 10**9, b"\x01" * 32)
    assert amount == 61_909_322 * 10**9 and liquid and fallback == r.treasury
    # what the exit paid on mainnet for that amount: whole RAO, 23,270 wei short
    assert min_out <= amount - 23_270 and min_out >= amount * 0.99
    import pytest

    with pytest.raises(ValueError):
        r._exit_args(61_909_322_421_259_730, b"\x01" * 32)
