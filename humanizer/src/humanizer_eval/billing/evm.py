"""EVM plumbing for the TAO payments (billing/tao.py): a JSON-RPC client that signs locally, and the route
Bittensor EVM --(ForeverMoney bridge over Chainlink CCIP)--> Base --(Uniswap V3)--> USDC, and back.

Addresses and call shapes are ForeverMoney's deployment manifest 1.6.1 as Talisman ships it
(``apps/extension/src/core/domains/forevermoney``), checked on chain 2026-11-25: the outbound lane to Base
is allowed, the vault is not paused, ``quoteBridgeOut`` answered ~0.0015 TAO for 1 TAO and
``quoteBridgeToFinney`` ~0.0023 ETH. On Base, TAO -> WETH (Uniswap V3 0.3%) -> USDC (0.05%) paid 175.91
USD for 1 TAO against a market of ~176.3 (~0.4%). Bridged TAO on Base is backed by root-staked TAO in the
ForeverMoney vault on Finney; the bridge's admin is a 2-of-3 Safe (their security page).

Needs the extra ``tao`` (eth-account, which brings eth-abi).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
MAX_UINT = 2**256 - 1
RAO_WEI = 10**9


class ChainError(RuntimeError):
    pass


@dataclass(frozen=True)
class SignedTx:
    hash: str
    raw: str  # hex, no 0x


def selector(sig: str) -> bytes:
    from eth_utils import keccak

    return keccak(sig.encode())[:4]


def topic(sig: str) -> str:
    from eth_utils import keccak

    return "0x" + keccak(sig.encode()).hex()


def calldata(sig: str, types: list[str], args: list[Any]) -> str:
    from eth_abi import encode

    return (selector(sig) + encode(types, args)).hex()


def word(hexdata: str, i: int) -> int:
    h = hexdata.removeprefix("0x")
    return int(h[64 * i : 64 * (i + 1)], 16)


def ok(receipt: dict[str, Any]) -> bool:
    return int(receipt.get("status", "0x0"), 16) == 1


def transfers_to(receipt: dict[str, Any], token: str, to: str) -> int:
    """Sum of ERC-20 ``Transfer`` amounts of ``token`` to ``to`` in a receipt."""
    want = "0x" + to.lower().removeprefix("0x").rjust(64, "0")
    n = 0
    for lg in receipt.get("logs") or []:
        t = lg.get("topics") or []
        if (
            lg["address"].lower() == token.lower()
            and len(t) == 3
            and t[0] == TRANSFER_TOPIC
            and t[2].lower() == want
        ):
            n += int(lg["data"], 16)
    return n


class EvmClient:
    """JSON-RPC over httpx; legacy transactions signed locally, nonces from the node."""

    def __init__(self, rpc: str, chain_id: int, http: httpx.Client | None = None):
        from eth_account import Account

        self._Account = Account
        self.rpc_url = rpc
        self.chain_id = chain_id
        self.http = http or httpx.Client(timeout=30)
        self._id = 0

    def rpc(self, method: str, *params: Any) -> Any:
        self._id += 1
        for attempt in range(4):
            r = self.http.post(self.rpc_url, json={"jsonrpc": "2.0", "id": self._id, "method": method,
                                                   "params": list(params)})  # fmt: skip
            # publicnode answers a burst with 403, others with 429; a resent transaction is "already known"
            if r.status_code in (403, 429, 502, 503, 504) and attempt < 3:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                raise ChainError(f"{method}: {d['error']}")
            return d["result"]
        raise ChainError(f"{method}: rate limited")  # pragma: no cover

    def address(self, key: bytes) -> str:
        return self._Account.from_key(key).address

    def call(self, to: str, data: str, frm: str | None = None, value: int = 0) -> str:
        tx: dict[str, Any] = {"to": to, "data": "0x" + data}
        if frm:
            tx["from"] = frm
        if value:
            tx["value"] = hex(value)
        return self.rpc("eth_call", tx, "latest")

    def balance(self, address: str) -> int:
        return int(self.rpc("eth_getBalance", address, "latest"), 16)

    def erc20_balance(self, token: str, owner: str) -> int:
        return int(self.call(token, calldata("balanceOf(address)", ["address"], [owner])), 16)

    def gas_price(self) -> int:
        return int(self.rpc("eth_gasPrice"), 16)

    def block_number(self) -> int:
        return int(self.rpc("eth_blockNumber"), 16)

    def sign(self, key: bytes, to: str, data: str, value: int = 0, gas: int | None = None) -> SignedTx:
        frm = self.address(key)
        if gas is None:
            est = {"from": frm, "to": to, "data": "0x" + data, "value": hex(value)}
            gas = int(int(self.rpc("eth_estimateGas", est), 16) * 1.3) + 10_000
        tx = {"to": to, "data": "0x" + data, "value": value, "gas": gas, "gasPrice": self.gas_price(),
              "nonce": int(self.rpc("eth_getTransactionCount", frm, "pending"), 16),
              "chainId": self.chain_id}  # fmt: skip
        signed = self._Account.sign_transaction(tx, key)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        return SignedTx("0x" + bytes(signed.hash).hex().removeprefix("0x"), bytes(raw).hex())

    def send(self, tx: SignedTx) -> None:
        try:
            self.rpc("eth_sendRawTransaction", "0x" + tx.raw)
        except ChainError as e:  # sent before a restart: the node knows it
            if "already" not in str(e).lower() and "known" not in str(e).lower():
                raise

    def receipt(self, tx_hash: str) -> dict[str, Any] | None:
        return self.rpc("eth_getTransactionReceipt", tx_hash)

    def wait(self, tx_hash: str, timeout_s: float = 120.0) -> dict[str, Any]:
        end = time.time() + timeout_s
        while time.time() < end:
            r = self.receipt(tx_hash)
            if r is not None:
                return r
            time.sleep(2.0)
        raise ChainError(f"{tx_hash} not mined in {timeout_s:.0f} s")

    def ensure_allowance(self, key: bytes, token: str, spender: str, amount: int) -> None:
        owner = self.address(key)
        q = calldata("allowance(address,address)", ["address", "address"], [owner, spender])
        if int(self.call(token, q), 16) >= amount:
            return
        tx = self.sign(
            key, token, calldata("approve(address,uint256)", ["address", "uint256"], [spender, MAX_UINT])
        )
        self.send(tx)
        if not ok(self.wait(tx.hash)):
            raise ChainError("approve reverted")


# --- the route -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RouteConfig:
    bittensor_rpc: str = "https://archive.chain.opentensor.ai"  # the lite node prunes receipts within hours
    bittensor_logs_rpc: str = "https://lite.chain.opentensor.ai"  # the archive node refuses eth_getLogs
    bittensor_chain_id: int = 964
    base_rpc: str = "https://base-rpc.publicnode.com"  # mainnet.base.org rate-limits eth_getLogs
    base_chain_id: int = 8453
    # ForeverMoney (manifest 1.6.1)
    fm_gateway: str = "0xd5Fa238aa4177f6c1341491969d9cBeec94EEd69"  # Bittensor EVM, outbound
    fm_alpha_token: str = "0xC5b6C1632d34901239396F5E1BDe54B342900256"  # noqa: S105 (a contract address)
    fm_vault: str = "0x11837459896D96F821a8D88eC93a3C8D152033D4"
    fm_base_selector: int = 15971525489660198786
    fm_spoke_gateway: str = "0x1da2415229b614C787e145D1D7346eb496319C52"  # Base
    # CCIP deliveries: the destination's router names the OffRamps it trusts for a source chain (they
    # change with CCIP upgrades, several at once: a delivery is accepted from any of them)
    base_router: str = "0x881e3A65B4d4a04dD529061dd0071cf975F58bCD"
    bittensor_router: str = "0xD941FbEcd2b971D0f54b4C34286c95Fab52B60b8"
    bittensor_selector: int = 2135107236357186872  # Bittensor EVM as a CCIP source
    # Base's spoke delivers to the V5 hub on Bittensor EVM (its hub pointer is immutable). When the hub
    # cannot make the liquid exit on Finney it books the TAO for the exit's EVM fallback (the treasury),
    # which claims it and sends it on to the ss58 address with the balance-transfer precompile.
    fm_hub: str = "0xcd0C6d98D0A126B1c113d15b4c28F38321437787"
    balance_transfer: str = "0x0000000000000000000000000000000000000800"  # transfer(bytes32 pubkey)
    finney_claim_gas: int = 350_000  # claimLiquid, measured 230k
    finney_transfer_gas: int = 100_000
    bridge_out_gas: int = 700_000  # Talisman measured 615k
    bridge_in_dest_gas: int = 3_500_000  # the liquid exit on Finney (unstake), as the SDK sends
    ccip_fee_buffer: float = 0.02  # on top of the quote; the contracts refund what is not used
    # the liquid exit on Finney pays whole RAO, a few thousand wei under the amount: a minimum of exactly the
    # amount fails it with Slippage() and the hub books the TAO for the fallback instead (mainnet 2026-11-25)
    exit_tolerance: float = 0.005
    # Base: tokens and Uniswap V3
    wtao: str = "0xf3081494B87e8D5fb7960f066E931D1D0e6E3d67"  # TAO bridged by ForeverMoney, 18 decimals
    weth: str = "0x4200000000000000000000000000000000000006"
    usdc: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # native USDC (Circle)
    router: str = "0x2626664c2603336E57B271c5C0b26F421741e481"  # SwapRouter02
    quoter: str = "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a"  # QuoterV2
    tao_weth_fee: int = 3000
    weth_usdc_fee: int = 500


_BRIDGED_OUT = topic("BridgedOut(uint64,address,address,address,uint256,bytes32)")
_BRIDGED_TO_FINNEY = topic("BridgedToFinney(address,address,bytes32,uint256,bytes32)")
_EXEC_STATE = (
    topic("ExecutionStateChanged(uint64,uint64,bytes32,bytes32,uint8,bytes,uint256)"),  # data: hash, state
    topic("ExecutionStateChanged(uint64,uint64,bytes32,uint8,bytes)"),  # data: state
)
CCIP_SUCCESS, CCIP_FAILURE = 2, 3
#: the delivery is searched from this many blocks before we first saw the bridge call mined (it can be
#: seen late: a busy pass, a restart): ~1.5 h on Base, ~10 h on Bittensor; the search runs in 10k windows
DELIVERY_LOOKBACK_BLOCKS = 3000
_DELIVERED_LIQUID = topic("DeliveredLiquid(address,bytes32,uint256)")  # the hub's exit on Finney worked
_TO_FINNEY = "(address,uint256,(bytes32,address,bool,uint256),uint256)"
_TO_FINNEY_TYPES = ["address", "uint256", "(bytes32,address,bool,uint256)", "uint256"]


class ForeverMoneyBaseRoute:
    """What ``TaoRail`` moves money with. The treasury key is one EVM account, the same address on both
    chains: on Base it holds the USDC and some ETH (gas, and the CCIP fee of refunds)."""

    def __init__(self, treasury_key: bytes, cfg: RouteConfig | None = None):
        self.cfg = cfg or RouteConfig()
        self.bt = EvmClient(self.cfg.bittensor_rpc, self.cfg.bittensor_chain_id)
        self.bt_logs = EvmClient(self.cfg.bittensor_logs_rpc, self.cfg.bittensor_chain_id)
        self.base = EvmClient(self.cfg.base_rpc, self.cfg.base_chain_id)
        self._key = treasury_key
        self.treasury = self.base.address(treasury_key)

    # --- Bittensor side: a deposit account bridges its TAO to the treasury on Base ---------------------
    def address_of(self, key: bytes) -> str:
        return self.bt.address(key)

    def deposit_balance(self, address: str) -> int:
        return self.bt.balance(address)

    def bridge_ready(self) -> bool:
        c = self.cfg
        lane = word(
            self.bt.call(c.fm_gateway, calldata("allowedLane(uint64)", ["uint64"], [c.fm_base_selector])), 0
        )
        paused = word(self.bt.call(c.fm_vault, calldata("isPaused()", [], [])), 0)
        return bool(lane) and not paused

    def bridge_out_fee(self, amount_wei: int) -> int:
        c = self.cfg
        types = ["uint64", "address", "address", "uint256"]
        data = calldata("quoteBridgeOut(uint64,address,address,uint256)", types,
                        [c.fm_base_selector, c.fm_alpha_token, self.treasury, amount_wei])  # fmt: skip
        return int(self.bt.call(c.fm_gateway, data), 16)

    def bridge_out_cost(self, amount_wei: int) -> int:
        """What a deposit keeps back to bridge ``amount_wei``: the CCIP fee with its buffer, and gas."""
        fee = int(self.bridge_out_fee(amount_wei) * (1 + self.cfg.ccip_fee_buffer))
        gas = int(self.cfg.bridge_out_gas * self.bt.gas_price() * 1.2)
        return -(-(fee + gas) // RAO_WEI) * RAO_WEI

    def bridge_out_tx(self, key: bytes, amount_wei: int) -> SignedTx:
        c = self.cfg
        fee = int(self.bridge_out_fee(amount_wei) * (1 + c.ccip_fee_buffer))
        data = calldata(
            "bridgeOut(uint64,address,address,uint256,uint256,uint256)",
            ["uint64", "address", "address", "uint256", "uint256", "uint256"],
            [c.fm_base_selector, c.fm_alpha_token, self.treasury, amount_wei, 0, amount_wei],
        )
        return self.bt.sign(key, c.fm_gateway, data, value=amount_wei + fee, gas=c.bridge_out_gas)

    def bridge_out_result(self, tx_hash: str) -> dict[str, Any] | None:
        """None while unmined; else {"ok", "message_id", "from_block" (on Base)}."""
        r = self.bt.receipt(tx_hash)
        if r is None:
            return None
        mid = None
        for lg in r.get("logs") or []:
            if (lg.get("topics") or [None])[0] == _BRIDGED_OUT:
                mid = "0x" + lg["data"].removeprefix("0x")[128:192]  # data: (selector, minted, messageId)
        return {
            "ok": ok(r) and mid is not None,
            "message_id": mid,
            "from_block": self.base.block_number() - DELIVERY_LOOKBACK_BLOCKS,
        }

    # --- CCIP delivery ---------------------------------------------------------------------------------
    def _execution(self, client: EvmClient, router: str, source: int, message_id: str,
                   from_block: int) -> dict[str, Any] | None:  # fmt: skip
        """The terminal ExecutionStateChanged of ``message_id`` on the destination ("state" added),
        emitted by one of the OffRamps the destination's router names for ``source`` (scanned in windows:
        public endpoints cap the block range of eth_getLogs and want the addresses)."""
        ramps = self._offramps(client, router, source)
        head, step = client.block_number(), 10_000
        src = "0x" + format(source, "064x")
        for start in range(max(0, from_block), head + 1, step):
            q = {"address": ramps, "topics": [list(_EXEC_STATE), src, None, message_id],
                 "fromBlock": hex(start), "toBlock": hex(min(start + step - 1, head))}  # fmt: skip
            for lg in client.rpc("eth_getLogs", q):
                if lg["address"].lower() not in ramps:
                    continue
                state = word(lg["data"], 1 if lg["topics"][0] == _EXEC_STATE[0] else 0)
                if state in (CCIP_SUCCESS, CCIP_FAILURE):
                    return {**lg, "state": state}
        return None

    def _delivery(self, client: EvmClient, router: str, source: int, message_id: str,
                  from_block: int) -> bool | None:  # fmt: skip
        lg = self._execution(client, router, source, message_id, from_block)
        return None if lg is None else lg["state"] == CCIP_SUCCESS

    @staticmethod
    def _offramps(client: EvmClient, router: str, source: int) -> list[str]:
        """The router's OffRamps for deliveries from ``source`` (Router.getOffRamps, read each time: CCIP
        upgrades add new ones and keep the old ones for messages in flight)."""
        from eth_abi import decode

        out = client.call(router, calldata("getOffRamps()", [], []))
        pairs = decode(["(uint64,address)[]"], bytes.fromhex(out.removeprefix("0x")))[0]
        ramps = sorted({ramp.lower() for sel, ramp in pairs if sel == source})
        if not ramps:
            raise ChainError(f"router {router} has no OffRamp for source {source}")
        return ramps

    def arrived_on_base(self, message_id: str, from_block: int) -> bool | None:
        """True delivered, False failed (a manual retry on the CCIP explorer), None not yet."""
        c = self.cfg
        return self._delivery(self.base, c.base_router, c.bittensor_selector, message_id, from_block)

    def finney_outcome(self, message_id: str, from_block: int) -> str | None:
        """A refund's CCIP message on Bittensor: None not seen yet, "failed" (CCIP gave up), "liquid" (the
        hub made the exit: the TAO is on the ss58 address), "claimable" (the hub booked it for the
        treasury to claim)."""
        c = self.cfg
        try:
            lg = self._execution(self.bt_logs, c.bittensor_router, c.fm_base_selector, message_id, from_block)
            if lg is None:
                return None
            if lg["state"] != CCIP_SUCCESS:
                return "failed"
            r = self.bt.receipt(lg["transactionHash"])
        except ChainError:
            return None
        if r is None:
            return None
        # one execution can carry several messages: ours are the logs after the previous execution event
        logs = sorted(r.get("logs") or [], key=lambda x: int(x["logIndex"], 16))
        mine: list[str] = []
        for x in logs:
            t0 = (x.get("topics") or [None])[0]
            if int(x["logIndex"], 16) == int(lg["logIndex"], 16):
                break
            mine = [] if t0 in _EXEC_STATE else [*mine, t0]
        return "liquid" if _DELIVERED_LIQUID in mine else "claimable"

    # --- Bittensor EVM: the treasury claims a booked exit and sends the TAO on -------------------------
    def claimable_tao(self) -> int:
        c = self.cfg
        data = calldata(
            "claimableToken(address,address)", ["address", "address"], [c.fm_alpha_token, self.treasury]
        )
        return int(self.bt.call(c.fm_hub, data), 16)

    def claim_tx(self, amount_wei: int) -> SignedTx:
        """claimLiquid: the hub unwraps the booked ``amount_wei`` to the treasury's account."""
        c = self.cfg
        data = calldata("claimLiquid(address,uint256,address)", ["address", "uint256", "address"],
                        [c.fm_alpha_token, self._min_out(amount_wei), self.treasury])  # fmt: skip
        return self.bt.sign(self._key, c.fm_hub, data, gas=c.finney_claim_gas)

    def forward_tx(self, tao_wei: int, pubkey: bytes) -> SignedTx:
        """TAO from the treasury's EVM account to an ss58 account (whole RAO)."""
        c = self.cfg
        data = calldata("transfer(bytes32)", ["bytes32"], [pubkey])
        value = tao_wei // RAO_WEI * RAO_WEI
        return self.bt.sign(self._key, c.balance_transfer, data, value=value, gas=c.finney_transfer_gas)

    def bt_tx_ok(self, tx_hash: str) -> bool | None:
        r = self.bt.receipt(tx_hash)
        return None if r is None else ok(r)

    def treasury_usdc(self) -> int:
        """USDC (6 decimals) the treasury holds on Base."""
        return self.base.erc20_balance(self.cfg.usdc, self.treasury)

    def treasury_bt_balance(self) -> int:
        return self.bt.balance(self.treasury)

    def finney_gas_wei(self) -> int:
        """Bittensor EVM gas of a claim and the transfer that follows it (with a margin)."""
        c = self.cfg
        return int((c.finney_claim_gas + c.finney_transfer_gas) * self.bt.gas_price() * 1.2)

    # --- Base: swaps through Uniswap V3 -------------------------------------------------------------------
    def _path(self, sell_tao: bool) -> bytes:
        c = self.cfg
        hops: list[Any] = [c.wtao, c.tao_weth_fee, c.weth, c.weth_usdc_fee, c.usdc]
        if not sell_tao:
            hops = hops[::-1]
        return b"".join(bytes.fromhex(h[2:]) if isinstance(h, str) else h.to_bytes(3, "big") for h in hops)

    def _quote(self, path: bytes, amount: int) -> int:
        data = calldata("quoteExactInput(bytes,uint256)", ["bytes", "uint256"], [path, amount])
        return word(self.base.call(self.cfg.quoter, data), 0)

    def quote_sell(self, tao_wei: int) -> int:
        """USDC units the pools pay for ``tao_wei`` of bridged TAO now."""
        return self._quote(self._path(True), tao_wei)

    def quote_buy(self, usdc: int) -> int:
        return self._quote(self._path(False), usdc)

    def price(self) -> float:
        """USD per TAO for a small sale: the reference of the slippage guard and of quotes."""
        return self.quote_sell(10**16) / 10**6 * 100

    def eth_usd(self) -> float:
        c = self.cfg
        path = bytes.fromhex(c.weth[2:]) + c.weth_usdc_fee.to_bytes(3, "big") + bytes.fromhex(c.usdc[2:])
        return self._quote(path, 10**16) / 10**6 * 100

    def _swap_tx(self, sell_tao: bool, amount: int, min_out: int) -> SignedTx:
        c = self.cfg
        self.base.ensure_allowance(self._key, c.wtao if sell_tao else c.usdc, c.router, amount)
        data = calldata("exactInput((bytes,address,uint256,uint256))", ["(bytes,address,uint256,uint256)"],
                        [(self._path(sell_tao), self.treasury, amount, min_out)])  # fmt: skip
        return self.base.sign(self._key, c.router, data)

    def sell_tx(self, tao_wei: int, min_usdc: int) -> SignedTx:
        return self._swap_tx(True, tao_wei, min_usdc)

    def buy_tx(self, usdc: int, min_tao_wei: int) -> SignedTx:
        return self._swap_tx(False, usdc, min_tao_wei)

    def swap_result(self, tx_hash: str) -> dict[str, Any] | None:
        r = self.base.receipt(tx_hash)
        if r is None:
            return None
        return {"ok": ok(r), "usdc_in": transfers_to(r, self.cfg.usdc, self.treasury),
                "tao_in": transfers_to(r, self.cfg.wtao, self.treasury)}  # fmt: skip

    # --- Base -> Finney: refunds ----------------------------------------------------------------------
    def _min_out(self, tao_wei: int) -> int:
        return int(tao_wei * (1.0 - self.cfg.exit_tolerance))

    def _exit_args(self, tao_wei: int, pubkey: bytes) -> list[Any]:
        """``tao_wei`` in whole RAO (the bridge's rule); the exit pays at least ``_min_out`` of it."""
        if tao_wei % RAO_WEI:
            raise ValueError("a bridge to Finney moves whole RAO")
        exit_ = (pubkey, self.treasury, True, self._min_out(tao_wei))
        return [self.cfg.wtao, tao_wei, exit_, self.cfg.bridge_in_dest_gas]

    def bridge_in_fee(self, tao_wei: int, pubkey: bytes) -> int:
        """ETH (wei) the CCIP message of a refund costs, with its buffer."""
        data = calldata(
            f"quoteBridgeToFinney{_TO_FINNEY}", _TO_FINNEY_TYPES, self._exit_args(tao_wei, pubkey)
        )
        return int(int(self.base.call(self.cfg.fm_spoke_gateway, data), 16) * (1 + self.cfg.ccip_fee_buffer))

    def bridge_in_tx(self, tao_wei: int, pubkey: bytes) -> SignedTx:
        c = self.cfg
        self.base.ensure_allowance(self._key, c.wtao, c.fm_spoke_gateway, tao_wei)
        fee = self.bridge_in_fee(tao_wei, pubkey)
        data = calldata(f"bridgeToFinney{_TO_FINNEY}", _TO_FINNEY_TYPES, self._exit_args(tao_wei, pubkey))
        return self.base.sign(self._key, c.fm_spoke_gateway, data, value=fee)

    def bridge_in_result(self, tx_hash: str) -> dict[str, Any] | None:
        r = self.base.receipt(tx_hash)
        if r is None:
            return None
        mid = None
        for lg in r.get("logs") or []:
            if (lg.get("topics") or [None])[0] == _BRIDGED_TO_FINNEY:
                mid = "0x" + lg["data"].removeprefix("0x")[64:128]  # data: (amount, messageId)
        return {
            "ok": ok(r) and mid is not None,
            "message_id": mid,
            "from_block": self.bt.block_number() - DELIVERY_LOOKBACK_BLOCKS,
        }

    def send(self, chain: str, tx: SignedTx) -> None:
        (self.bt if chain == "bittensor" else self.base).send(tx)

    def treasury_balances(self) -> dict[str, Any]:
        b, c = self.base, self.cfg
        return {"address": self.treasury, "base_eth": b.balance(self.treasury) / 1e18,
                "base_usdc": b.erc20_balance(c.usdc, self.treasury) / 1e6,
                "base_tao": b.erc20_balance(c.wtao, self.treasury) / 1e18,
                "bittensor_tao": self.treasury_bt_balance() / 1e18,
                "bittensor_claimable_tao": self.claimable_tao() / 1e18}  # fmt: skip

    def usdc_transfer_tx(self, to: str, usdc: int) -> SignedTx:
        """The owner's withdrawal: USDC from the treasury to an address of theirs (Coinbase on Base, etc.)."""
        data = calldata("transfer(address,uint256)", ["address", "uint256"], [to, usdc])
        return self.base.sign(self._key, self.cfg.usdc, data)
