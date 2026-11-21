"""Crown decision (D-11, D-22, D-23, D-25) and king history with 85/10/5 payouts (D-12, D-13).

A challenger takes the crown iff the one-sided lower bound of the paired bootstrap, ``q_alpha(dS*)``,
is strictly above ``delta_min`` (D-23, D-25). Gate quality enters ``S`` per output (D-22: levels
1 / 0.5 / 0 weight each output's contribution); the old aggregate rule "at least 95% of outputs pass"
is off by default (``min_gate_pass = 0``, owner decision 2026-11-09) and kept only as an option.

Payouts: the current king gets 85%, the two previous kings with *distinct hotkeys* 10% and 5%.
Missing slots (fewer than three distinct kings) and slots held by the owner's baseline (D-13) go to
the burn sink ``BURN_KEY``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from humanizer_eval.scoring.bootstrap import DEFAULT_ALPHA, DEFAULT_N_BOOT, paired_bootstrap
from humanizer_eval.scoring.metric import (
    DEFAULT_P,
    DEFAULT_PROB_WEIGHT,
    DEFAULT_SEVERE_PENALTY,
    HumanReference,
    SubmissionRun,
)

#: Payout sink for shares that belong to nobody (missing kings, the owner's baseline).
BURN_KEY = "__burn__"
#: ``(hotkey, registered_at) -> bool``: is this hotkey still registered from that registration block?
IsRegistered = Callable[[str, "int | None"], bool]
#: Shares of the current king and the two previous distinct kings, in basis points (sum 10000).
PAYOUT_SHARES_BP: tuple[int, ...] = (8500, 1000, 500)
DEFAULT_MIN_GATE_PASS = 0.0  # D-22: quality is per output now; aggregate floor off
DEFAULT_DELTA_MIN = 0.0  # D-25 / Q-4: 0 until the first periods' data


@dataclass(frozen=True)
class CrownDecision:
    crown: bool
    reasons: tuple[str, ...]  # one line per criterion, pass or fail
    delta: float  # point estimate S_challenger - S_king
    lower: float  # alpha-quantile of the bootstrap replicates
    se: float
    s_king: float
    s_challenger: float
    king_gate_pass_rate: float
    challenger_gate_pass_rate: float
    alpha: float
    delta_min: float
    min_gate_pass: float
    n_boot: int
    seed: int | None


def crown_decision(
    king: SubmissionRun,
    challenger: SubmissionRun,
    human: HumanReference,
    *,
    alpha: float = DEFAULT_ALPHA,
    delta_min: float = DEFAULT_DELTA_MIN,
    min_gate_pass: float = DEFAULT_MIN_GATE_PASS,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int | np.random.SeedSequence = 0,
    p: float = DEFAULT_P,
    w_prob: float = DEFAULT_PROB_WEIGHT,
    penalty: float = DEFAULT_SEVERE_PENALTY,
) -> CrownDecision:
    """Decide whether ``challenger`` dethrones ``king`` on the same inputs and the same H."""
    if not 0 <= min_gate_pass <= 1:
        raise ValueError("min_gate_pass must be in [0, 1]")
    bs = paired_bootstrap(
        king, challenger, human, n_boot=n_boot, alpha=alpha, seed=seed, p=p, w_prob=w_prob, penalty=penalty
    )
    gate_rate = challenger.gate_pass_rate
    stat_ok = bs.lower > delta_min
    gate_ok = gate_rate >= min_gate_pass
    q = f"q_{alpha:g}(dS*)"
    reasons = (
        f"{'PASS' if stat_ok else 'FAIL'}: {q} = {bs.lower:+.4f} {'>' if stat_ok else '<='} "
        f"delta_min = {delta_min:+.4f} (dS = {bs.delta:+.4f}, se = {bs.se:.4f}, B = {n_boot})",
        f"{'PASS' if gate_ok else 'FAIL'}: challenger gate pass rate {gate_rate:.3f} "
        f"{'>=' if gate_ok else '<'} {min_gate_pass:.3f}",
    )
    return CrownDecision(
        crown=bool(stat_ok and gate_ok),
        reasons=reasons,
        delta=bs.delta,
        lower=bs.lower,
        se=bs.se,
        s_king=bs.s_king,
        s_challenger=bs.s_challenger,
        king_gate_pass_rate=king.gate_pass_rate,
        challenger_gate_pass_rate=gate_rate,
        alpha=alpha,
        delta_min=delta_min,
        min_gate_pass=min_gate_pass,
        n_boot=n_boot,
        seed=bs.seed,
    )


# --- king history and payouts -----------------------------------------------------------------
@dataclass(frozen=True)
class KingEntry:
    submission_id: str
    hotkey: str
    crowned_at: datetime  # timezone-aware
    delta_s: float | None  # dS at coronation; None for the owner's baseline
    is_baseline: bool = False
    #: block at which the king's hotkey was registered when it submitted (one submission per
    #: registration, D-66); a later deregistration (or re-registration) drops the entry from payouts
    registered_at: int | None = None

    def __post_init__(self) -> None:
        if not self.submission_id or not self.hotkey:
            raise ValueError("submission_id and hotkey must be non-empty")
        if self.hotkey == BURN_KEY:
            raise ValueError(f"{BURN_KEY!r} is reserved for the burn sink")
        if self.crowned_at.tzinfo is None or self.crowned_at.utcoffset() is None:
            raise ValueError("crowned_at must be timezone-aware")
        if self.delta_s is not None and not np.isfinite(self.delta_s):
            raise ValueError("delta_s must be finite")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["crowned_at"] = self.crowned_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> KingEntry:
        return cls(
            submission_id=str(d["submission_id"]),
            hotkey=str(d["hotkey"]),
            crowned_at=datetime.fromisoformat(d["crowned_at"]),
            delta_s=None if d.get("delta_s") is None else float(d["delta_s"]),
            is_baseline=bool(d.get("is_baseline", False)),
            registered_at=None if d.get("registered_at") is None else int(d["registered_at"]),
        )


@dataclass
class KingState:
    """Ordered history of coronations, oldest first; the last entry is the current king."""

    history: list[KingEntry] = field(default_factory=list)

    SCHEMA_VERSION = 1

    def __post_init__(self) -> None:
        ids = [e.submission_id for e in self.history]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate submission_id in king history")
        for a, b in zip(self.history, self.history[1:], strict=False):
            if b.crowned_at < a.crowned_at:
                raise ValueError("king history must be ordered by crowned_at")

    @classmethod
    def initial(cls, baseline_submission_id: str, owner_hotkey: str, crowned_at: datetime) -> KingState:
        """Start a track with the owner's baseline as king; its share is burned (D-13)."""
        return cls([KingEntry(baseline_submission_id, owner_hotkey, crowned_at, None, is_baseline=True)])

    @property
    def current(self) -> KingEntry | None:
        return self.history[-1] if self.history else None

    def apply_crown(
        self,
        submission_id: str,
        hotkey: str,
        crowned_at: datetime,
        delta_s: float,
        registered_at: int | None = None,
    ) -> KingEntry:
        """Record a new king (call only after ``crown_decision(...).crown`` is true)."""
        entry = KingEntry(submission_id, hotkey, crowned_at, float(delta_s), registered_at=registered_at)
        if any(e.submission_id == submission_id for e in self.history):
            raise ValueError(f"submission {submission_id!r} was already crowned")
        cur = self.current
        if cur is not None and crowned_at < cur.crowned_at:
            raise ValueError("crowned_at is earlier than the current king's coronation")
        self.history.append(entry)
        return entry

    def active_history(self, is_registered: IsRegistered | None = None) -> list[KingEntry]:
        """Coronations whose king still holds the registration it submitted from (the baseline always).

        A king that lost its registration drops out as if it had never been crowned: the kings after
        and before it close ranks (a fourth king moves back into the third payout slot), and if it was
        the current king, the crown returns to the previous one (D-12, D-66).
        """
        if is_registered is None:
            return list(self.history)
        return [e for e in self.history if e.is_baseline or is_registered(e.hotkey, e.registered_at)]

    def current_active(self, is_registered: IsRegistered | None = None) -> KingEntry | None:
        h = self.active_history(is_registered)
        return h[-1] if h else None

    def payout_slots(self, is_registered: IsRegistered | None = None) -> list[KingEntry]:
        """Current king, then previous kings with hotkeys not seen yet, newest first (at most 3)."""
        slots: list[KingEntry] = []
        seen: set[str] = set()
        for e in reversed(self.active_history(is_registered)):
            if e.hotkey in seen:
                continue
            seen.add(e.hotkey)
            slots.append(e)
            if len(slots) == len(PAYOUT_SHARES_BP):
                break
        return slots

    def payout_weights(self, is_registered: IsRegistered | None = None) -> dict[str, float]:
        """``{hotkey: weight}`` summing to 1; unowned shares under ``BURN_KEY`` (D-12, D-13).

        A previous-king slot that is missing or held by the owner's baseline rolls to the current
        king (as in SN34). The current king's own share is burned only when it is the baseline,
        and then the unowned residual is burned too.
        """
        bp: dict[str, int] = {}
        burn = 0
        slots = self.payout_slots(is_registered)
        current = slots[0] if slots and not slots[0].is_baseline else None
        residual = 0
        for i, share in enumerate(PAYOUT_SHARES_BP):
            e = slots[i] if i < len(slots) else None
            if e is None or e.is_baseline:
                if i == 0:
                    burn += share
                else:
                    residual += share
            else:
                bp[e.hotkey] = bp.get(e.hotkey, 0) + share
        if current is not None:
            bp[current.hotkey] += residual
        else:
            burn += residual
        if burn:
            bp[BURN_KEY] = burn
        total = sum(PAYOUT_SHARES_BP)
        return {k: v / total for k, v in bp.items()}

    # --- serialization ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"version": self.SCHEMA_VERSION, "history": [e.to_dict() for e in self.history]}

    @classmethod
    def from_dict(cls, d: dict) -> KingState:
        if d.get("version") != cls.SCHEMA_VERSION:
            raise ValueError(f"unsupported KingState version {d.get('version')!r}")
        return cls([KingEntry.from_dict(e) for e in d["history"]])

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False)

    @classmethod
    def from_json(cls, s: str) -> KingState:
        return cls.from_dict(json.loads(s))
