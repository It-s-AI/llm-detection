"""Public price card, quotes and statements (D-80).

A miner pays for what evaluating its submission costs, item by item: ``quantity x rate`` from a public,
versioned price card whose sha256 is pinned in every quote and statement, so anyone can recompute a
statement after the period is disclosed. Amounts are integer cents; every line item is rounded up.

* **Quote** (before the run): an upper bound. Detector words are bounded by the G0 length cap
  (output <= 1.3 x input words), judges by a per-output cap, Lium by a fixed number of hours. The
  payment processor's fee is added so that the net amount covers the quote.
* **Statement** (after the run): the metered quantities -- words actually sent to each detector (cache
  hits are free), tokens per judge model, pod hours -- times the same card's rates. The difference to the
  hold goes back to the miner's balance.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

LENGTH_CAP = 1.3  # G0 (gates/normalize.apply_length_limit)


def cents_up(usd: float) -> int:
    """USD -> integer cents, rounded up (a line item never undercharges by round-off)."""
    return math.ceil(round(usd * 100, 6))


@dataclass(frozen=True)
class PriceCard:
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> PriceCard:
        return cls(json.loads(Path(path).read_text()))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def version(self) -> str:
        return str(self.raw["version"])

    @property
    def detectors(self) -> dict[str, dict[str, Any]]:
        return self.raw["detectors"]

    def detector_cost_usd(self, detector: str, words: int) -> float:
        d = self.detectors[detector]
        units = words / d["per_words"]
        if d.get("rounding") == "ceil_per_text":
            units = math.ceil(units) if words > 0 else 0
        return units * d["rate_usd"]

    def judge_cost_usd(self, model: str, tokens_in: int, tokens_out: int) -> float:
        j = self.raw["judges"].get(model)
        if j is None:
            raise KeyError(f"judge model {model!r} is not on price card {self.version}")
        return (tokens_in * j["in_per_mtok"] + tokens_out * j["out_per_mtok"]) / 1e6


def card_archive(root: str | Path = "configs/price_cards") -> dict[str, PriceCard]:
    """Every price card ever published, by sha256 (a run is settled with the card of its quote)."""
    out: dict[str, PriceCard] = {}
    for p in sorted(Path(root).glob("*.json")):
        c = PriceCard.load(p)
        out[c.sha256] = c
    return out


@dataclass
class LineItem:
    item: str
    quantity: float
    unit: str
    rate: str  # human-readable rate from the card, e.g. "$0.05 / 100 words (per text, rounded up)"
    amount_cents: int


@dataclass
class Statement:
    kind: str  # "quote" | "statement"
    card_version: str
    card_sha256: str
    items: list[LineItem] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_cents(self) -> int:
        return sum(i.amount_cents for i in self.items)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_cents"] = self.total_cents
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Statement:
        return cls(
            kind=d["kind"],
            card_version=d["card_version"],
            card_sha256=d["card_sha256"],
            items=[LineItem(**i) for i in d["items"]],
            meta=d.get("meta", {}),
        )


def _detector_rate(card: PriceCard, det: str) -> str:
    d = card.detectors[det]
    per = f"{d['per_words']:,} words"
    return f"${d['rate_usd']} / {per}" + (
        " (per text, rounded up)" if d.get("rounding") == "ceil_per_text" else ""
    )


def _detector_items(card: PriceCard, words_by_detector: Mapping[str, Sequence[int]]) -> list[LineItem]:
    items = []
    for det in sorted(words_by_detector):
        words = list(words_by_detector[det])
        usd = sum(card.detector_cost_usd(det, w) for w in words)
        items.append(
            LineItem(
                f"detector:{det}",
                sum(words),
                "words",
                _detector_rate(card, det) + f", {len(words)} texts",
                cents_up(usd),
            )
        )
    return items


def quote(card: PriceCard, input_words: Sequence[int], detectors: Sequence[str] | None = None) -> Statement:
    """Upper-bound quote for one submission on a period whose inputs have ``input_words`` words."""
    dets = list(card.detectors) if detectors is None else list(detectors)
    caps = [math.floor(LENGTH_CAP * w) for w in input_words]
    st = Statement("quote", card.version, card.sha256, meta={"n_outputs": len(caps)})
    st.items += _detector_items(card, {d: caps for d in dets})
    jq = card.raw["judge_quote_per_output_usd"]
    st.items.append(
        LineItem("judges", len(caps), "outputs", f"up to ${jq} / output", cents_up(jq * len(caps)))
    )
    lium = card.raw["lium"]
    st.items.append(
        LineItem(
            "lium:h100",
            lium["quote_hours"],
            "hours",
            f"${lium['rate_per_hour_usd']} / hour",
            cents_up(lium["quote_hours"] * lium["rate_per_hour_usd"]),
        )
    )
    return st


def statement(
    card: PriceCard,
    *,
    detector_words: Mapping[str, Sequence[int]],
    judge_tokens: Mapping[str, tuple[int, int]],
    pod_hours: float,
    meta: dict[str, Any] | None = None,
) -> Statement:
    """Actual cost from metered quantities."""
    st = Statement("statement", card.version, card.sha256, meta=dict(meta or {}))
    st.items += _detector_items(card, detector_words)
    for model in sorted(judge_tokens):
        t_in, t_out = judge_tokens[model]
        j = card.raw["judges"][model]
        st.items.append(
            LineItem(
                f"judge:{model}",
                t_in + t_out,
                "tokens",
                f"${j['in_per_mtok']} / 1M in, ${j['out_per_mtok']} / 1M out ({t_in} in, {t_out} out)",
                cents_up(card.judge_cost_usd(model, t_in, t_out)),
            )
        )
    lium = card.raw["lium"]
    st.items.append(
        LineItem(
            "lium:h100",
            round(pod_hours, 4),
            "hours",
            f"${lium['rate_per_hour_usd']} / hour",
            cents_up(pod_hours * lium["rate_per_hour_usd"]),
        )
    )
    return st
