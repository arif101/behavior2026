"""Reflex tier — the per-tick decision function (spec v1.1 rules, v1.2 two-tier split).

tick(obs) consumes head measurements and returns the policy payload + events. It never
emits joint commands and never steers head inference. A DirectorBrain (brains.py) may
override the deliberative decisions (assignment, macro choice); absent/timed-out brains
fall through to the programmed defaults implemented here.

obs schema (all numbers, producible by oracle stubs or live heads):
  time_remaining: int ticks
  stage:   {arm: {phase: idle|nav|contact|transition, progress: float, entropy: float}}
  p_sat:   {lid: float}          instantaneous, from stage head
  visible: {lid: bool}           target-of-literal currently visible (occlusion gate)
  grounding: {arm: {point: (x,y,z)|None, conf: float, instance: str|None}}
  belief:  {instance: {mu:(x,y,z), sigma_tr: float, p_exists: float, age: float,
                       attached_to: None|"L"|"R"}}
  pfail:   {arm: float}
  ee:      {arm: (x,y,z)}
"""
from __future__ import annotations

from .core import ArmState, Config, DirectorState, LState, Literal, MacroState, dependents_of

CONTACTY = ("contact",)


class Reflex:
    def __init__(self, literals: list[Literal], cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.literals = {l.lid: l for l in literals}
        self.st = DirectorState.fresh(literals, self.cfg)
        self._bound: dict[int, str] = {}   # lid -> bound instance id (no-double-claim at ID level)

    # ------------------------------------------------------------------ ledger
    def _update_ledger(self, obs, events):
        cfg, st = self.cfg, self.st
        for lid, lit in self.literals.items():
            s = st.lstate[lid]
            p = obs["p_sat"].get(lid, 0.0)
            vis = obs["visible"].get(lid, True)
            if s == LState.UNSAT:
                st.sat_streak[lid] = st.sat_streak[lid] + 1 if p > cfg.theta_sat else 0
                if st.sat_streak[lid] >= cfg.latch_windows:
                    st.lstate[lid] = LState.SAT_LATCHED
                    events.append(("latch", lid))
                    self._free_arm_of(lid, events)
            elif s in (LState.SAT_LATCHED, LState.BANKED):
                # visibility-gated rollback: occluded evidence FREEZES, never contradicts
                if vis and p < (1 - cfg.theta_sat):
                    st.contra_streak[lid] += 1
                else:
                    st.contra_streak[lid] = 0
                enough_time = obs["time_remaining"] > self.literals[lid].est_ticks
                if st.contra_streak[lid] >= cfg.rollback_windows and enough_time:
                    st.lstate[lid] = LState.UNSAT
                    st.sat_streak[lid] = 0
                    st.contra_streak[lid] = 0
                    events.append(("rollback", lid))

    def _free_arm_of(self, lid, events):
        for name, arm in self.st.arms.items():
            if arm.assigned == lid:
                arm.assigned = None
                arm.stage_age = 0
                arm.reattempt_pending_lid = None
                self.st.epoch += 1
                events.append(("free_arm", name, lid))

    # ------------------------------------------------------------------ macros
    def _macro_lifecycle(self, obs, events):
        cfg, st = self.cfg, self.st
        for name, arm in st.arms.items():
            if arm.macro is None:
                continue
            m, done = arm.macro, False
            if m.kind == "TRUNCATE_REPLAN":
                done = True
            elif m.kind == "RE_GROUND":
                g = obs["grounding"].get(name, {})
                done = g.get("conf", 0.0) > cfg.conf_tau or st.tick - m.entered > 120
            elif m.kind == "PLACE_DOWN_SAFE":
                done = arm.attached is None
            if done:
                arm.macro = None
                arm.last_macro_exit = st.tick
                arm.stage_age = 0
                arm.reattempt_pending_lid = m.lid   # next failure on this literal costs budget
                st.epoch += 1
                events.append(("macro_done", name, m.kind, m.lid))

    def _maybe_dispatch(self, obs, events, brain_decision):
        cfg, st = self.cfg, self.st
        for name, arm in st.arms.items():
            if arm.macro is not None or arm.assigned is None:
                continue
            if st.tick - arm.last_macro_exit < cfg.refractory:
                continue                                   # trigger refractory
            if obs["time_remaining"] <= cfg.endgame_reserve:
                continue                                   # do-no-harm window
            arm.pfail_streak = arm.pfail_streak + 1 if obs["pfail"].get(name, 0.0) > cfg.alpha_pfail else 0
            lit = self.literals[arm.assigned]
            aged = arm.stage_age > cfg.stage_age_kappa * lit.est_ticks
            if arm.pfail_streak >= cfg.pfail_sustain or aged:
                kind = None
                if brain_decision and brain_decision.get("macro", {}).get("arm") == name:
                    kind = brain_decision["macro"]["kind"]
                elif obs["grounding"].get(name, {}).get("conf", 1.0) < cfg.conf_tau and arm.phase == "nav":
                    kind = "RE_GROUND"
                elif arm.phase != "idle":
                    kind = "TRUNCATE_REPLAN"               # universal safe macro for stagnation
                if kind is None:
                    continue                               # idle/unmapped -> continue nominal
                # attachment gate: a holding arm may only place down safely, never retreat
                if arm.attached is not None and kind not in ("PLACE_DOWN_SAFE", "TRUNCATE_REPLAN"):
                    kind = "PLACE_DOWN_SAFE"
                if arm.reattempt_pending_lid == arm.assigned:
                    st.budgets[arm.assigned] -= 1          # failed genuine re-attempt
                    arm.reattempt_pending_lid = None
                    events.append(("budget", arm.assigned, st.budgets[arm.assigned]))
                arm.macro = MacroState(kind, arm.assigned, st.tick)
                arm.pfail_streak = 0
                st.epoch += 1
                events.append(("dispatch", name, kind, arm.assigned))
                if st.budgets[arm.assigned] <= 0:
                    self._abandon(arm.assigned, name, events)

    def _abandon(self, lid, arm_name, events):
        st, cfg = self.st, self.cfg
        other = "L" if arm_name == "R" else "R"
        if lid not in st.escalated and st.arms[other].assigned is None and st.arms[other].attached is None:
            st.escalated.add(lid)
            st.budgets[lid] = max(1, cfg.retry_budget // 2)
            self._free_arm_of(lid, events)
            st.arms[other].assigned = lid                  # cross-arm escalation, once
            st.epoch += 1
            events.append(("escalate", lid, other))
            return
        st.lstate[lid] = LState.ABANDONED
        self._free_arm_of(lid, events)
        for dep in dependents_of(lid, self.literals):
            if st.lstate[dep] == LState.UNSAT:
                st.lstate[dep] = LState.BLOCKED
                events.append(("blocked", dep, lid))
        events.append(("abandon", lid))

    # -------------------------------------------------------------- assignment
    def _eligible(self, obs):
        st, cfg = self.st, self.cfg
        claimed = {a.assigned for a in st.arms.values() if a.assigned is not None}
        out = []
        for lid, lit in self.literals.items():
            if st.lstate[lid] != LState.UNSAT or lid in claimed or st.budgets[lid] <= 0:
                continue
            if any(st.lstate[d] not in (LState.SAT_LATCHED, LState.BANKED) for d in lit.deps):
                continue
            if lit.est_ticks > obs["time_remaining"] - cfg.endgame_reserve:
                continue                                   # end-game: only finishable work
            out.append(lit)
        fanin = {lid: len(dependents_of(lid, self.literals)) for lid in self.literals}
        out.sort(key=lambda l: -(l.credit + fanin[l.lid]))  # subtree-weighted priority
        return out

    def _assign(self, obs, events, brain_decision):
        cfg, st = self.cfg, self.st
        elig = self._eligible(obs)
        if not elig:
            if all(a.assigned is None and a.macro is None for a in st.arms.values()) and any(
                s in (LState.UNSAT, LState.BLOCKED) for s in st.lstate.values()
            ):
                if not st.terminal:
                    st.terminal = True
                    events.append(("terminal_park",))
            return
        anchor = None                                      # co-location anchor target position
        for name in ("R", "L"):
            arm = st.arms[name]
            if arm.assigned is not None or arm.macro is not None:
                if arm.assigned is not None:
                    anchor = anchor or self._candidate_mu(self.literals[arm.assigned], obs)
                continue
            pick = None
            if brain_decision and name in brain_decision.get("assign", {}):
                want = brain_decision["assign"][name]
                pick = next((l for l in elig if l.lid == want), None)
            if pick is None:
                for cand in elig:
                    if arm.attached is not None:
                        # a holding arm may only take the literal OF the held instance
                        if not (cand.target_instance == arm.attached
                                or self._bound.get(cand.lid) == arm.attached):
                            continue
                    if cand.skill_type == "coordinated":
                        both_free = all(
                            a.assigned in (None, cand.lid) and a.macro is None and a.phase not in CONTACTY
                            for a in st.arms.values()
                        )
                        if not both_free:
                            continue
                    mu = self._candidate_mu(cand, obs)
                    if anchor is not None and mu is not None:
                        d = sum((a - b) ** 2 for a, b in zip(mu, anchor)) ** 0.5
                        if d > cfg.colocate_radius:
                            continue                       # base-aware batching
                    pick = cand
                    break
            if pick is None:
                continue
            arm.assigned = pick.lid
            arm.stage_age = 0
            elig = [l for l in elig if l.lid != pick.lid]
            inst = pick.target_instance or self._bind_instance(pick, obs)
            if inst:
                self._bound[pick.lid] = inst
            anchor = anchor or self._candidate_mu(pick, obs)
            st.epoch += 1
            events.append(("assign", name, pick.lid))
            if pick.skill_type == "coordinated":
                for oname, oarm in st.arms.items():
                    if oname != name and oarm.assigned is None:
                        oarm.assigned = pick.lid
                        events.append(("assign", oname, pick.lid))

    def _bind_instance(self, lit: Literal, obs) -> str | None:
        """Bind literal -> belief hypothesis: same category, not held, not bound to a
        BANKED/latched literal, nearest to either gripper. The ONLY writer of binding."""
        taken = {inst for l, inst in self._bound.items()
                 if self.st.lstate[l] in (LState.SAT_LATCHED, LState.BANKED) or l != lit.lid and l in
                 {a.assigned for a in self.st.arms.values()}}
        best, best_d = None, float("inf")
        for inst, h in obs["belief"].items():
            if not inst.startswith(lit.target_cat) or inst in taken or h.get("attached_to"):
                continue
            d = min(sum((a - b) ** 2 for a, b in zip(h["mu"], e)) ** 0.5 for e in obs["ee"].values())
            if d < best_d:
                best, best_d = inst, d
        return best

    def _target_mu(self, lid, obs):
        inst = self._bound.get(lid)
        h = obs["belief"].get(inst) if inst else None
        return h["mu"] if h else None

    def _candidate_mu(self, lit: Literal, obs):
        """Best position estimate for a literal's target: bound hypothesis, else the
        nearest unattached belief instance of the category (pre-binding estimate)."""
        mu = self._target_mu(lit.lid, obs)
        if mu is not None:
            return mu
        if lit.target_instance and lit.target_instance in obs["belief"]:
            return obs["belief"][lit.target_instance]["mu"]
        best, best_d = None, float("inf")
        for inst, h in obs["belief"].items():
            if not inst.startswith(lit.target_cat) or h.get("attached_to"):
                continue
            d = min(sum((a - b) ** 2 for a, b in zip(h["mu"], e)) ** 0.5
                    for e in obs["ee"].values())
            if d < best_d:
                best, best_d = h["mu"], d
        return best

    # ----------------------------------------------------------------- payload
    def _payload(self, obs):
        cfg, st = self.cfg, self.st
        pay = {"target_points": {}, "mask": {}, "meta": {}, "stage_tokens": {},
               "exec_prefix": None, "epoch": st.epoch}
        active_prefixes = []
        for name, arm in st.arms.items():
            hold = arm.attached is not None and arm.phase not in CONTACTY
            phase = obs["stage"][name]["phase"]
            if hold:
                pay["mask"][name] = 0
                pay["meta"][name] = {"attached_to": arm.attached, "mode": "HOLD"}
            else:
                src, pt, extra = self._point_for(name, arm, obs)
                pay["mask"][name] = 1 if pt is not None else 0
                pay["target_points"][name] = pt
                pay["meta"][name] = {"source": src, **extra}
                active_prefixes.append(
                    cfg.contact_prefix if phase in CONTACTY
                    else cfg.nav_prefix if phase == "nav" else cfg.transition_prefix
                )
            pay["stage_tokens"][name] = obs["stage"][name]
        pay["exec_prefix"] = min(active_prefixes) if active_prefixes else cfg.nav_prefix
        return pay

    def _point_for(self, name, arm, obs):
        cfg = self.cfg
        g = obs["grounding"].get(name, {})
        ee = obs["ee"][name]
        if g.get("point") is not None and g.get("conf", 0.0) > cfg.conf_tau:
            p = g["point"]
            return "grounding", tuple(a - b for a, b in zip(p, ee)), {"conf": g["conf"]}
        inst = self._bound.get(arm.assigned) if arm.assigned is not None else None
        h = obs["belief"].get(inst) if inst else None
        if h and h["p_exists"] >= cfg.p_exists_min and h["sigma_tr"] <= cfg.sigma_max:
            return "belief", tuple(a - b for a, b in zip(h["mu"], ee)), \
                {"age": h["age"], "sigma_tr": h["sigma_tr"]}
        return "none", None, {"scan_requested": True}

    # -------------------------------------------------------------------- tick
    def tick(self, obs, brain_decision=None):
        st = self.st
        st.tick += 1
        events: list[tuple] = []
        for name, arm in st.arms.items():
            arm.phase = obs["stage"][name]["phase"]
            arm.stage_age += 1
            arm.attached = next(
                (i for i, h in obs["belief"].items() if h.get("attached_to") == name), None)
            if arm.attached is not None and arm.assigned is not None:
                lit = self.literals[arm.assigned]
                held_is_target = (self._bound.get(arm.assigned) == arm.attached
                                  or lit.target_instance == arm.attached)
                if not held_is_target:
                    # holding something unrelated to the assignment (e.g. carrying the
                    # container): release the literal for a free arm to claim
                    lid = arm.assigned
                    arm.assigned = None
                    arm.stage_age = 0
                    st.epoch += 1
                    events.append(("release_assign", name, lid))
        self._update_ledger(obs, events)
        self._macro_lifecycle(obs, events)
        self._maybe_dispatch(obs, events, brain_decision)
        self._assign(obs, events, brain_decision)
        st.base_frozen = any(a.phase in CONTACTY for a in st.arms.values())
        st.base_owner = None if st.base_frozen else next(
            (n for n, a in st.arms.items() if a.phase == "nav"), None)
        q = sum(self.literals[l].credit for l, s in st.lstate.items()
                if s in (LState.SAT_LATCHED, LState.BANKED))
        total = sum(l.credit for l in self.literals.values())
        return {"payload": self._payload(obs), "events": events, "q": q / total if total else 0.0,
                "ledger": {l: s.value for l, s in st.lstate.items()},
                "arms": {n: {"assigned": a.assigned, "attached": a.attached,
                             "macro": a.macro.kind if a.macro else None,
                             "phase": a.phase, "bound": self._bound.get(a.assigned)
                             if a.assigned is not None else None}
                         for n, a in st.arms.items()},
                "base_frozen": st.base_frozen, "terminal": st.terminal, "tick": st.tick}
