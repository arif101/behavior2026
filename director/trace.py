"""One trace schema for both worlds (user requirement: full visibility everywhere).

The SAME record is emitted per tick whether the director is driven by the trace-replay
harness (recorded episodes) or by the live serve loop at rollout. The viewer renders
either — demo and production telemetry are one artifact.

Record (one JSON line per tick):
  {t, obs: {stage, p_sat, pfail, grounding_conf}, ledger, q, arms: {assigned, phase,
   macro, attached}, payload: {target_points, mask, meta, stage_tokens, exec_prefix,
   epoch}, events, base_frozen, terminal, brain: {consulted, decision, latency_ms}|null}
"""
from __future__ import annotations

import json


class TraceWriter:
    def __init__(self, path: str):
        self.f = open(path, "w")

    def rec(self, obs, tick_out, brain_info=None):
        row = {
            "t": tick_out["tick"],
            "obs": {
                "stage": obs["stage"],
                "p_sat": obs["p_sat"],
                "pfail": obs["pfail"],
                "grounding_conf": {a: g.get("conf") for a, g in obs.get("grounding", {}).items()},
                "time_remaining": obs["time_remaining"],
            },
            "ledger": tick_out["ledger"],
            "q": tick_out["q"],
            "payload": tick_out["payload"],
            "events": tick_out["events"],
            "base_frozen": tick_out["base_frozen"],
            "terminal": tick_out["terminal"],
            "brain": brain_info,
        }
        self.f.write(json.dumps(row) + "\n")

    def close(self):
        self.f.close()
