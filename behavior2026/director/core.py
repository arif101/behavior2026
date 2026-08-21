"""Director core state — every variable printable, no hidden memory (spec v1.1/v1.2).

Ledger states, literal specs, arm state, macro FSM state, config. Pure data; all
decision logic lives in reflex.py. Times are in ticks (caller defines tick=frame).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LState(Enum):
    UNSAT = "unsat"
    SAT_LATCHED = "sat_latched"      # evidence-latched; revocable only via rollback rule
    BANKED = "banked"                # latched + geometry-confirmed; audit may un-bank
    ABANDONED = "abandoned"          # budget exhausted after cross-arm escalation
    BLOCKED = "blocked"              # a transitive prerequisite was abandoned


@dataclass(frozen=True)
class Literal:
    lid: int
    predicate: str                   # inside / ontop / toggled_on / ...
    target_cat: str
    target_instance: str | None      # belief hypothesis id once bound
    reference: str | None
    deps: tuple[int, ...] = ()       # prerequisite lids (e.g. open(fridge))
    skill_type: str = "uncoordinated"  # navigation | uncoordinated | coordinated
    est_ticks: int = 600             # demo-median completion estimate
    credit: float = 1.0


@dataclass
class MacroState:
    kind: str                        # TRUNCATE_REPLAN | RE_GROUND | PLACE_DOWN_SAFE | ...
    lid: int
    entered: int


@dataclass
class ArmState:
    assigned: int | None = None
    attached: str | None = None      # instance id held by this gripper
    phase: str = "idle"              # idle | nav | contact | transition (from stage head)
    stage_age: int = 0
    macro: MacroState | None = None
    last_macro_exit: int = -(10**9)
    reattempt_pending_lid: int | None = None  # macro done; next same-literal failure costs budget
    pfail_streak: int = 0


@dataclass
class Config:
    theta_sat: float = 0.8
    latch_windows: int = 3           # consecutive windows of p_sat > theta to latch
    rollback_windows: int = 3        # consecutive VISIBLE contradictions to un-latch
    alpha_pfail: float = 0.30
    pfail_sustain: int = 3
    stage_age_kappa: float = 3.0     # macro trigger at kappa x est_ticks
    conf_tau: float = 0.7            # grounding confidence gate
    p_exists_min: float = 0.2
    sigma_max: float = 1.0
    refractory: int = 50             # ticks after macro exit before same trigger may refire
    retry_budget: int = 3
    endgame_reserve: int = 100       # do-no-harm window (ticks)
    colocate_radius: float = 3.0     # 2nd arm claims only targets within this of 1st arm's
    contact_prefix: int = 6
    nav_prefix: int = 24
    transition_prefix: int = 12
    contact_phases: tuple[str, ...] = ("contact",)


@dataclass
class DirectorState:
    lstate: dict[int, LState]
    budgets: dict[int, int]
    sat_streak: dict[int, int] = field(default_factory=dict)
    contra_streak: dict[int, int] = field(default_factory=dict)
    escalated: set[int] = field(default_factory=set)
    arms: dict[str, ArmState] = field(default_factory=lambda: {"L": ArmState(), "R": ArmState()})
    base_owner: str | None = None
    base_frozen: bool = False
    epoch: int = 0
    tick: int = 0
    terminal: bool = False

    @classmethod
    def fresh(cls, literals: list[Literal], cfg: Config) -> "DirectorState":
        return cls(
            lstate={l.lid: LState.UNSAT for l in literals},
            budgets={l.lid: cfg.retry_budget for l in literals},
            sat_streak={l.lid: 0 for l in literals},
            contra_streak={l.lid: 0 for l in literals},
        )


def dependents_of(lid: int, literals: dict[int, Literal]) -> set[int]:
    """Transitive dependents: literals whose prerequisite chain contains lid."""
    out: set[int] = set()
    frontier = {lid}
    while frontier:
        nxt = {l.lid for l in literals.values() if set(l.deps) & frontier} - out
        out |= nxt
        frontier = nxt
    return out
