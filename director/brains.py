"""DirectorBrain — the deliberative tier interface (spec v1.2).

The reflex tier consults the brain at events (episode start, subtask boundary, macro
dispatch, ambiguity); the brain returns a typed decision dict or None. None / timeout /
invalid schema all fall through to the reflex tier's programmed defaults — an API
failure degrades to boring-but-sane, never burns a prescribed rollout.

Decision schema (validated by the reflex tier before any effect):
  {"assign": {"L": lid|None, "R": lid|None},          # optional overrides
   "macro": {"arm": "L"|"R", "kind": str},            # optional
   "query_override": {"L": category, "R": category},  # optional
   "note": str}                                       # logged, never executed
"""
from __future__ import annotations


class DirectorBrain:
    def consult(self, situation: dict) -> dict | None:  # pragma: no cover - interface
        raise NotImplementedError


class ProgrammedBrain(DirectorBrain):
    """The fallback and A/B control arm: defers everything to reflex defaults."""

    def consult(self, situation: dict) -> dict | None:
        return None


class OpusBrain(DirectorBrain):
    """VLM deliberative tier (Claude Opus). Build order: offline prompt harness over
    recorded demo situations FIRST (strategy-agreement vs human demonstrators), then
    live async consults at chunk boundaries. Requirements carried from spec v1.2:
    strict timeout -> ProgrammedBrain answer; temperature 0; decision cache keyed by
    situation digest; every prompt/response logged; circuit breaker after N failures.
    """

    def __init__(self, client=None, timeout_s: float = 4.0):
        self.client = client
        self.timeout_s = timeout_s

    def consult(self, situation: dict) -> dict | None:
        if self.client is None:
            return None  # offline mode == programmed behavior
        raise NotImplementedError("wired in the offline-harness milestone")
