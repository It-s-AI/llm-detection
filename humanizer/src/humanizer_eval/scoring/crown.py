"""Crown decision (D-11, D-22, D-23, D-25) and king history with 85/10/5 payouts (D-12, D-13).

A challenger takes the crown iff the one-sided lower bound of the paired bootstrap, ``q_alpha(dS*)``,
is strictly above ``delta_min`` (D-23, D-25). Gate quality enters ``S`` per output (D-22: levels
1 / 0.5 / 0 weight each output's contribution); the old aggregate rule "at least 95% of outputs pass"
is off by default (``min_gate_pass = 0``, owner decision 2026-11-09) and kept only as an option.

Payouts: the current king gets 85%, the two previous kings with *distinct hotkeys* 10% and 5%.
Missing slots (fewer than three distinct kings) and slots held by the owner's baseline go to the
current king. Nothing is burned (owner's decision 2026-11-25): while the owner's baseline is king there
are no payout hotkeys and the validators give the competition's share to detection. The shares are
constants of the validator code; the evaluator only publishes who the kings are (``payout_hotkeys``).
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


def payout_shares(kings: list[str]) -> dict[str, float]:
    """85/10/5 over ``kings`` (newest first, distinct); the shares of missing slots go to the first."""
    kings = list(dict.fromkeys(kings))[: len(PAYOUT_SHARES_BP)]
    if not kings:
        return {}
    total = sum(PAYOUT_SHARES_BP)
    bp = {hk: PAYOUT_SHARES_BP[i] for i, hk in enumerate(kings)}
    bp[kings[0]] += total - sum(bp.values())
    return {hk: v / total for hk, v in bp.items()}


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
        """Start a track with the owner's baseline as king; it is paid nothing (D-13)."""
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

    def payout_hotkeys(self, is_registered: IsRegistered | None = None) -> list[str]:
        """Hotkeys of the current king and the previous distinct kings, newest first (at most 3), the
        owner's baseline left out; empty while the baseline is king. This is what the leaderboard signs."""
        slots = self.payout_slots(is_registered)
        if not slots or slots[0].is_baseline:
            return []
        return [e.hotkey for e in slots if not e.is_baseline]

    def payout_registrations(self, is_registered: IsRegistered | None = None) -> list[int | None]:
        """The registration block each of ``payout_hotkeys`` was crowned under, same order: a validator
        drops a king whose hotkey registered again since (D-66), even from a stale leaderboard."""
        slots = self.payout_slots(is_registered)
        if not slots or slots[0].is_baseline:
            return []
        return [e.registered_at for e in slots if not e.is_baseline]

    def payout_weights(self, is_registered: IsRegistered | None = None) -> dict[str, float]:
        """``{hotkey: share}`` of the competition (summing to 1, or empty while the baseline is king):
        the same rule as the validator's (D-12): 85/10/5, missing slots roll to the current king."""
        return payout_shares(self.payout_hotkeys(is_registered))

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
