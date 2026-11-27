"""Which hotkeys are registered on the subnet, and since which block: a metagraph snapshot written by
the validator side.

The snapshot is ``{"netuid": 32, "block": ..., "ts": unix seconds, "hotkeys": [ss58, ...],
"registered_at": {ss58: block}, "commitments": {ss58: {"data": str, "block": int}}}``,
refreshed every few minutes (``scripts/refresh_metagraph.py`` with
``bittensor``). A stale snapshot refuses new submissions (HTTP 503) rather than letting an unregistered
hotkey pay for a crown it cannot receive.

A registration is identified by ``(hotkey, registered_at)``: one submission per registration (D-66), and
a king whose hotkey was deregistered (or registered again later) drops out of the payouts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


class MetagraphSnapshot:
    def __init__(self, path: Path | str, netuid: int = 32, max_age_s: float = 3600.0, now=time.time):
        self.path = Path(path)
        self.netuid = netuid
        self.max_age_s = max_age_s
        self._now = now
        self._cache: tuple[float, dict[str, int | None]] | None = None

    def _load(self) -> dict[str, int | None]:
        mtime = self.path.stat().st_mtime
        if self._cache is None or self._cache[0] != mtime:
            d = json.loads(self.path.read_text())
            if int(d["netuid"]) != self.netuid:
                raise RuntimeError(f"metagraph snapshot is for netuid {d['netuid']}, not {self.netuid}")
            self._ts = float(d["ts"])
            reg = d.get("registered_at") or {}
            self._cache = (
                mtime,
                {hk: (None if reg.get(hk) is None else int(reg[hk])) for hk in d["hotkeys"]},
            )
            self._commitments = {
                hk: (str(c["data"]), int(c["block"])) for hk, c in (d.get("commitments") or {}).items()
            }
        if self._now() - self._ts > self.max_age_s:
            raise RuntimeError("metagraph snapshot is stale")
        return self._cache[1]

    def __call__(self, hotkey: str) -> bool:
        return hotkey in self._load()

    def registration(self, hotkey: str) -> int | None:
        """Block the hotkey was registered at; None if it is not registered. Raises if stale."""
        hks = self._load()
        if hotkey not in hks:
            return None
        return hks[hotkey] if hks[hotkey] is not None else 0

    def uid(self, hotkey: str) -> int | None:
        """The hotkey's UID (its index in the snapshot's metagraph order); None if not registered."""
        hks = self._load()
        return list(hks).index(hotkey) if hotkey in hks else None

    def commitment(self, hotkey: str) -> tuple[str, int] | None:
        """The hotkey's current on-chain commitment ``(data, block)`` (registered hotkeys only)."""
        self._load()
        return self._commitments.get(hotkey)

    def is_registered(self, hotkey: str, registered_at: int | None) -> bool:
        """Still registered from that registration (``registered_at=None``: any registration)."""
        hks = self._load()
        if hotkey not in hks:
            return False
        return registered_at is None or hks[hotkey] is None or hks[hotkey] == registered_at


def write_snapshot(
    path: Path | str,
    netuid: int,
    block: int,
    hotkeys: list[str],
    registered_at: dict[str, int] | None = None,
    now=time.time,
    commitments: dict[str, tuple[str, int]] | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    body = {
        "netuid": netuid,
        "block": block,
        "ts": now(),
        "hotkeys": sorted(hotkeys),
        "registered_at": registered_at or {},
        "commitments": {hk: {"data": d, "block": b} for hk, (d, b) in (commitments or {}).items()},
    }
    tmp.write_text(json.dumps(body))
    tmp.replace(p)
