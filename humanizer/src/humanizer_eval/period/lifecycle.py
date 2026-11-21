"""Period lifecycle (D-56, D-57): one idempotent pass at a time (scripts/rotate_period.py, every 15 minutes).

    period t starts ── commit c_{t+1} = H(secret), declare the seed block B (build_lead before t ends)
      ── B finalized ── build D_{t+1} with seed H(secret || blockhash(B)), NEXT = t+1
      ── the server loads t+1 (H scored once, in the server) and re-runs the king on it ── READY
      ── t ends ── CURRENT = t+1 (without READY only after ``ready_grace_s``: the worker then re-runs the
         king itself and holds the queue meanwhile) ── no evaluation of t still running ── reveal t

B is fixed when the commitment is published, so the seed cannot be steered: we know the secret but not the
future block, everyone else learns the block but not the secret. ``schedule.json`` holds only public facts
(it is served at /v1/periods); the secret waits in ``.<pid>.secret`` (mode 600) until the build moves it
into the period directory, and is published with the reveal.
"""

from __future__ import annotations

import fcntl
import json
import math
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from humanizer_eval.period.notary import commitment_text, merkle_text
from humanizer_eval.period.seed import commitment, make_secret

SCHEDULE = "schedule.json"


@dataclass(frozen=True)
class LifecycleConfig:
    period_s: float = 4 * 86400.0
    build_lead_s: float = 12 * 3600.0  # B is produced this long before the period ends
    min_lead_blocks: int = 50  # B is at least this far ahead of the commitment (~10 min)
    finality_blocks: int = 12  # build only once B is this deep under the finalized head
    ready_grace_s: float = 3 * 3600.0
    block_s: float = 12.0
    netuid: int = 32  # named in the notary remarks
    notary_margin_blocks: int = 10  # the commitment remark must land this far before the seed block


def next_period_id(pid: str) -> str:
    m = re.fullmatch(r"(.*?)(\d+)", pid)
    if not m:
        raise ValueError(f"period id {pid!r} does not end in a number")
    return f"{m.group(1)}{int(m.group(2)) + 1}"


def _write(path: Path, text: str, mode: int | None = None) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    if mode is not None:
        tmp.chmod(mode)
    tmp.replace(path)


class Lifecycle:
    """``builder(pid, secret, block_hash, block) -> manifest`` writes ``<periods>/<pid>/``;
    ``revealer(pid) -> dict`` writes the reveal of a closed period; ``running(pid)`` counts evaluations of
    ``pid`` still running; ``publish(kind, pid)`` mirrors public files (the schedule, a reveal) outside."""

    def __init__(
        self,
        periods_dir: Path | str,
        chain: Any,
        builder: Callable[[str, bytes, str, int], dict[str, Any]],
        *,
        revealer: Callable[[str], dict[str, Any]] | None = None,
        running: Callable[[str], int] = lambda pid: 0,
        publish: Callable[[str, str], None] | None = None,
        notary: Any = None,  # period/notary.py SubprocessNotary: the remarks on chain (D-57)
        alert: Callable[..., Any] = lambda text, dedupe=None: None,
        cfg: LifecycleConfig | None = None,
        now: Callable[[], float] = time.time,
    ):
        self.dir = Path(periods_dir)
        self.chain, self.builder, self.revealer = chain, builder, revealer
        self.running, self.publish, self.alert, self.notary = running, publish, alert, notary
        self.cfg, self.now = cfg or LifecycleConfig(), now

    # --- state ---------------------------------------------------------------------------------
    def schedule(self) -> dict[str, Any]:
        p = self.dir / SCHEDULE
        return json.loads(p.read_text()) if p.exists() else {"periods": {}}

    def _save(self, sch: dict[str, Any]) -> None:
        sch["updated_at"] = self.now()
        _write(self.dir / SCHEDULE, json.dumps(sch, indent=1))

    def _read(self, name: str) -> str | None:
        p = self.dir / name
        return p.read_text().strip() or None if p.exists() else None

    def secret_path(self, pid: str) -> Path:
        return self.dir / f".{pid}.secret"

    @contextmanager
    def _locked(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / ".lifecycle.lock", "w") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("another rotation pass is running") from None
            yield

    # --- one pass --------------------------------------------------------------------------------
    def step(self) -> list[str]:
        """Do whatever is due, in order; returns what was done."""
        done: list[str] = []
        with self._locked():
            sch = self.schedule()
            for f in (self._adopt, self._switch, self._commit, self._notarize, self._build, self._reveal):
                out = f(sch)
                if out:
                    done.append(out)
                    self._save(sch)
        return done

    def _adopt(self, sch: dict[str, Any]) -> str | None:
        """The period in CURRENT when the rotation starts (the first one is built by hand)."""
        cur = self._read("CURRENT")
        if not cur or cur in sch["periods"]:
            return None
        m = json.loads((self.dir / cur / "manifest.json").read_text())
        sch["current"] = cur
        sch["periods"][cur] = {"commitment": m.get("commitment"), "seed_block": m.get("block"),
                               "block_hash": m.get("block_hash"), "merkle_root": m.get("merkle_root"),
                               "n_items": m.get("n_items"), "built_at": m.get("created_at"),
                               "started_at": self.now()}  # fmt: skip
        return f"adopted {cur} as the current period"

    def _switch(self, sch: dict[str, Any]) -> str | None:
        cur, nxt = sch.get("current"), sch.get("next")
        if not (cur and nxt) or not sch["periods"].get(nxt, {}).get("built_at"):
            return None
        rec_c, rec_n = sch["periods"][cur], sch["periods"][nxt]
        end = rec_c["started_at"] + self.cfg.period_s
        now = self.now()
        if now < end:
            return None
        if self.notary is not None and not rec_n.get("notary_merkle"):
            self.alert(f"period {cur} is over but the Merkle root of {nxt} is not on chain yet",
                       dedupe=f"no-merkle-{nxt}")  # fmt: skip
            return None
        ready = self.dir / nxt / "READY"
        if not ready.exists() and now < end + self.cfg.ready_grace_s:
            self.alert(f"period {cur} is over but {nxt} is not ready yet (the king's run on it)",
                       dedupe=f"not-ready-{nxt}")  # fmt: skip
            return None
        _write(self.dir / "CURRENT", nxt + "\n")
        (self.dir / "NEXT").unlink(missing_ok=True)
        rec_c["ended_at"] = rec_n["started_at"] = now
        rec_n["ready"] = ready.exists()
        sch["current"], sch["next"] = nxt, None
        self.alert(f"period {nxt} started ({rec_n.get('n_items')} texts); {cur} closed, reveal follows")
        return f"switched {cur} -> {nxt}"

    def _commit(self, sch: dict[str, Any]) -> str | None:
        cur = sch.get("current")
        if not cur or sch.get("next"):
            return None
        nxt = next_period_id(cur)
        if nxt in sch["periods"] or (self.dir / nxt).exists():
            raise RuntimeError(f"{nxt} already exists but is not the scheduled next period")
        sp = self.secret_path(nxt)
        if sp.exists():  # a pass that died after writing the secret: keep it (nothing was published)
            secret = bytes.fromhex(sp.read_text().strip())
        else:
            secret = make_secret()
            _write(sp, secret.hex(), mode=0o600)
        head = self.chain.head()
        starts_at = sch["periods"][cur]["started_at"] + self.cfg.period_s
        ahead = math.ceil((starts_at - self.cfg.build_lead_s - self.now()) / self.cfg.block_s)
        seed_block = head + max(self.cfg.min_lead_blocks, ahead)
        sch["periods"][nxt] = {"commitment": commitment(secret), "committed_at": self.now(),
                               "committed_block": head, "seed_block": seed_block,
                               "planned_start": starts_at}  # fmt: skip
        sch["next"] = nxt
        self._save(sch)  # public before anything else happens
        if self.publish:
            self.publish("schedule", nxt)
        self.alert(f"commitment for {nxt} published: c={commitment(secret)[:16]}..., seed block {seed_block}")
        return f"committed {nxt} (seed block {seed_block})"

    def _post(self, rec: dict[str, Any], field: str, text: str, pid: str) -> bool:
        try:
            rec[field] = {**self.notary.post(text), "text": text}
        except Exception as e:
            self.alert(f"notary remark for {pid} failed: {type(e).__name__}: {e}", dedupe=f"{field}-{pid}")
            return False
        return True

    def _notarize(self, sch: dict[str, Any]) -> str | None:
        """The commitment on chain, before its seed block; without it the period is not built."""
        nxt = sch.get("next")
        if self.notary is None or not nxt or sch["periods"][nxt].get("notary_commitment"):
            return None
        rec = sch["periods"][nxt]
        if self.chain.head() >= rec["seed_block"] - self.cfg.notary_margin_blocks:
            self.alert(f"the commitment for {nxt} did not reach the chain before its seed block; "
                       f"run rotate_period.py --recommit", dedupe=f"late-{nxt}")  # fmt: skip
            return None
        if not self._post(rec, "notary_commitment", commitment_text(self.cfg.netuid, nxt, rec["commitment"],
                                                                    rec["seed_block"]), nxt):  # fmt: skip
            return None
        if self.publish:
            self._save(sch)
            self.publish("schedule", nxt)
        return f"commitment of {nxt} on chain (block {rec['notary_commitment']['block']})"

    def recommit(self, reason: str) -> str:
        """Abandon the next period's commitment before it is built (e.g. its remark missed the seed block)
        and let the next pass publish a new one. The abandoned one stays in the schedule, with the reason."""
        with self._locked():
            sch = self.schedule()
            nxt = sch.get("next")
            if not nxt or sch["periods"][nxt].get("built_at"):
                raise RuntimeError("no unbuilt next period to abandon")
            rec = sch["periods"].pop(nxt)
            sch.setdefault("abandoned", []).append(
                {"period": nxt, **rec, "abandoned_at": self.now(), "reason": reason}
            )
            sch["next"] = None
            self.secret_path(nxt).unlink(missing_ok=True)  # never used: no seed was derived from it
            self._save(sch)
        self.alert(f"commitment for {nxt} abandoned: {reason}")
        return nxt

    def _build(self, sch: dict[str, Any]) -> str | None:
        nxt = sch.get("next")
        if not nxt:
            return None
        rec = sch["periods"][nxt]
        if rec.get("built_at"):
            if self.notary is not None and not rec.get("notary_merkle"):
                if not self._post(rec, "notary_merkle", merkle_text(self.cfg.netuid, nxt, rec["merkle_root"],
                                                                    rec["n_items"], rec["block_hash"]), nxt):  # fmt: skip
                    return None
                _write(self.dir / "NEXT", nxt + "\n")
                return f"Merkle root of {nxt} on chain (block {rec['notary_merkle']['block']})"
            if self._read("NEXT") != nxt:
                _write(self.dir / "NEXT", nxt + "\n")
            return None
        if self.notary is not None and not rec.get("notary_commitment"):
            return None  # the commitment must be on chain before its seed block
        head = self.chain.head()
        if head < rec["seed_block"] + self.cfg.finality_blocks:
            return None
        sp = self.secret_path(nxt)
        secret = bytes.fromhex(sp.read_text().strip())
        bh = self.chain.block_hash(rec["seed_block"])
        try:
            m = self.builder(nxt, secret, bh, rec["seed_block"])
        except Exception as e:
            self.alert(f"building {nxt} failed: {type(e).__name__}: {e}", dedupe=f"build-{nxt}")
            raise
        if m.get("commitment") != rec["commitment"] or m.get("block_hash") != bh:
            raise RuntimeError(f"{nxt}: the built manifest does not match the commitment and the seed block")
        rec.update(block_hash=bh, merkle_root=m.get("merkle_root"), n_items=m.get("n_items"),
                   built_at=self.now(), cost_usd=m.get("cost_usd"))  # fmt: skip
        sp.unlink(missing_ok=True)  # the builder keeps it in <pid>/secret.hex until the reveal
        notarized = self.notary is None or self._post(
            rec,
            "notary_merkle",
            merkle_text(self.cfg.netuid, nxt, rec["merkle_root"], rec["n_items"], bh),
            nxt,
        )
        if notarized:  # the server prepares the period only once its texts are fixed on chain
            _write(self.dir / "NEXT", nxt + "\n")
        if self.publish:
            self._save(sch)
            self.publish("schedule", nxt)
        self.alert(
            f"period {nxt} built: {m.get('n_items')} texts, ${m.get('cost_usd', 0):.2f}; server prepares it"
        )
        return f"built {nxt}"

    def _reveal(self, sch: dict[str, Any]) -> str | None:
        if self.revealer is None:
            return None
        for pid, rec in sch["periods"].items():
            if not rec.get("ended_at") or rec.get("revealed_at"):
                continue
            if self.running(pid):
                continue  # an evaluation started before the switch still uses these inputs
            info = self.revealer(pid)
            rec.update(revealed_at=self.now(), reveal=info)
            if self.publish:
                self._save(sch)
                self.publish("reveal", pid)
            self.alert(f"period {pid} revealed ({info.get('submissions', 0)} submissions)")
            return f"revealed {pid}"
        return None
