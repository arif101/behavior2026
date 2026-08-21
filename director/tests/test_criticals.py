"""The 7 critical design-review findings (wf_ba3610aa) as deterministic regression tests.

Each test reconstructs the failure scenario from the review; if any of these ever go
red again, the corresponding livelock/safety bug has been reintroduced.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from director.core import Config, LState, Literal  # noqa: E402
from director.reflex import Reflex  # noqa: E402


def base_obs(time_remaining=5000, phases=None, pfail=None, psat=None, visible=None,
             belief=None, grounding=None):
    phases = phases or {"L": "idle", "R": "nav"}
    return {
        "time_remaining": time_remaining,
        "stage": {a: {"phase": p, "progress": 0.5, "entropy": 0.1} for a, p in phases.items()},
        "p_sat": psat or {},
        "visible": visible or {},
        "grounding": grounding or {},
        "belief": belief or {},
        "pfail": pfail or {"L": 0.0, "R": 0.0},
        "ee": {"L": (0.0, 0.2, 0.4), "R": (0.0, -0.2, 0.4)},
    }


def fridge_task():
    """open(fridge) gates 3 inside() literals — the review's canonical dependency case."""
    return [
        Literal(0, "open", "fridge", "fridge_1", None, deps=(), est_ticks=200),
        Literal(1, "inside", "milk", "milk_1", "fridge_1", deps=(0,), est_ticks=200),
        Literal(2, "inside", "eggs", "eggs_1", "fridge_1", deps=(0,), est_ticks=200),
        Literal(3, "inside", "butter", "butter_1", "fridge_1", deps=(0,), est_ticks=200),
    ]


def drain_budget(rf, obs_fn, lid):
    """Drive repeated dispatch->macro-done->redispatch cycles until lid's budget is gone."""
    guard = 0
    while rf.st.budgets[lid] > 0 and guard < 500:
        guard += 1
        arm = next((n for n, a in rf.st.arms.items() if a.assigned == lid), None)
        if arm is None:
            break
        rf.st.arms[arm].stage_age = 10**6            # stagnation trigger
        rf.st.arms[arm].last_macro_exit = -(10**9)   # skip refractory for the drain loop
        rf.tick(obs_fn())
        rf.st.arms[arm].macro = None                 # macro completes instantly
        rf.st.arms[arm].last_macro_exit = rf.st.tick
        rf.st.arms[arm].reattempt_pending_lid = lid
    assert guard < 500, "budget never drained — dispatch loop broken"


def test_1_abandon_no_livelock_and_cascade():
    rf = Reflex(fridge_task())
    obs = lambda: base_obs(phases={"L": "nav", "R": "nav"})
    rf.tick(obs())
    assert rf.st.arms["R"].assigned == 0, "fridge (fan-in 3) must win priority"
    # drain through BOTH arms: budget exhausts -> cross-arm escalation (once) -> abandon
    drain_budget(rf, obs, 0)
    assert 0 in rf.st.escalated, "cross-arm escalation must have been offered"
    r = rf.tick(obs())
    assert rf.st.lstate[0] == LState.ABANDONED
    # dependents blocked, never re-assigned (the livelock the review found)
    for dep in (1, 2, 3):
        assert rf.st.lstate[dep] == LState.BLOCKED
    for _ in range(5):
        r = rf.tick(obs())
    assert all(a.assigned is None for a in rf.st.arms.values())
    assert r["terminal"], "empty assignable set with blocked literals must park, not spin"


def test_2_base_frozen_during_contact():
    rf = Reflex(fridge_task())
    r = rf.tick(base_obs(phases={"L": "contact", "R": "nav"}))
    assert r["base_frozen"] is True
    r = rf.tick(base_obs(phases={"L": "idle", "R": "nav"}))
    assert r["base_frozen"] is False and rf.st.base_owner == "R"


def test_2b_colocation_batching():
    lits = [
        Literal(0, "inside", "can", None, "bin", est_ticks=100),
        Literal(1, "inside", "cup", None, "sink", est_ticks=100),
    ]
    belief = {
        "can_1": {"mu": (0.5, 0.0, 0.1), "sigma_tr": 0.1, "p_exists": 0.9, "age": 0.1, "attached_to": None},
        "cup_1": {"mu": (9.0, 9.0, 0.1), "sigma_tr": 0.1, "p_exists": 0.9, "age": 0.1, "attached_to": None},
    }
    rf = Reflex(lits, Config(colocate_radius=3.0))
    rf.tick(base_obs(phases={"L": "idle", "R": "idle"}, belief=belief))
    a = [x.assigned for x in rf.st.arms.values()]
    assert a.count(None) == 1, f"far-apart literals must not be claimed simultaneously: {a}"


def test_3_instance_binding_no_double_claim():
    lits = [
        Literal(0, "inside", "sock", None, "hamper", est_ticks=100),
        Literal(1, "inside", "sock", None, "hamper", est_ticks=100),
    ]
    belief = {
        "sock_1": {"mu": (0.5, 0.1, 0.0), "sigma_tr": 0.1, "p_exists": 0.9, "age": 0.1, "attached_to": None},
        "sock_2": {"mu": (0.6, -0.1, 0.0), "sigma_tr": 0.1, "p_exists": 0.9, "age": 0.1, "attached_to": None},
    }
    rf = Reflex(lits)
    rf.tick(base_obs(phases={"L": "idle", "R": "idle"}, belief=belief))
    bound = list(rf._bound.values())
    assert len(bound) == len(set(bound)), f"two literals bound to the same physical sock: {bound}"


def test_4_macro_lifecycle_refractory_and_budget():
    lits = [Literal(0, "toggled_on", "radio", "radio_1", None, est_ticks=100)]
    rf = Reflex(lits, Config(refractory=50, pfail_sustain=1))
    obs = lambda pf: base_obs(phases={"L": "idle", "R": "contact"}, pfail={"L": 0.0, "R": pf})
    rf.tick(obs(0.0))
    assert rf.st.arms["R"].assigned == 0
    rf.tick(obs(0.9))
    assert rf.st.arms["R"].macro is not None, "sustained pfail must dispatch"
    b0 = rf.st.budgets[0]
    rf.tick(obs(0.9))                                  # trigger masked while macro active
    assert rf.st.budgets[0] == b0, "no budget decrement on dispatch itself"
    rf.st.arms["R"].macro = None                       # macro completes
    rf.st.arms["R"].last_macro_exit = rf.st.tick
    rf.st.arms["R"].reattempt_pending_lid = 0
    rf.tick(obs(0.9))
    assert rf.st.arms["R"].macro is None, "refractory must block immediate refire"
    rf.st.arms["R"].last_macro_exit = -(10**9)
    rf.tick(obs(0.9))
    assert rf.st.budgets[0] == b0 - 1, "failed genuine re-attempt costs exactly one budget"


def test_5_endgame_and_terminal_do_no_harm():
    lits = [Literal(0, "inside", "can", "can_1", "bin", est_ticks=1000)]
    rf = Reflex(lits, Config(endgame_reserve=100))
    rf.tick(base_obs(time_remaining=500))              # 1000-tick literal, 500 left
    assert rf.st.arms["R"].assigned is None, "must not start unfinishable work"
    rf2 = Reflex(lits, Config(endgame_reserve=100, pfail_sustain=1))
    rf2.tick(base_obs(time_remaining=5000))
    rf2.st.arms["R"].stage_age = 10**6
    rf2.tick(base_obs(time_remaining=50, phases={"L": "idle", "R": "contact"},
                      pfail={"L": 0, "R": 0.9}))
    assert rf2.st.arms["R"].macro is None, "no macro dispatch inside the do-no-harm window"


def test_6_attachment_gate_place_down_safe():
    lits = [Literal(0, "inside", "can", "can_1", "bin", est_ticks=100)]
    rf = Reflex(lits, Config(pfail_sustain=1, refractory=0))
    belief = {"can_1": {"mu": (0.5, 0, 0.1), "sigma_tr": 0.1, "p_exists": 0.9, "age": 0.1,
                        "attached_to": "R"}}
    rf.tick(base_obs(phases={"L": "idle", "R": "nav"}, belief=belief,
                     grounding={"R": {"point": None, "conf": 0.1}}))
    rf.st.arms["R"].stage_age = 10**6
    rf.tick(base_obs(phases={"L": "idle", "R": "nav"}, belief=belief,
                     grounding={"R": {"point": None, "conf": 0.1}}))
    m = rf.st.arms["R"].macro
    assert m is not None and m.kind == "PLACE_DOWN_SAFE", \
        f"holding arm must place down safely, never retreat/re-ground blind: {m}"


def test_7_latch_hysteresis_and_occlusion_freeze():
    lits = [Literal(0, "toggled_on", "radio", "radio_1", None, est_ticks=100)]
    rf = Reflex(lits, Config(latch_windows=3))
    obs_hi = lambda vis: base_obs(psat={0: 0.95}, visible={0: vis})
    obs_lo = lambda vis: base_obs(psat={0: 0.02}, visible={0: vis})
    rf.tick(obs_hi(True))
    assert rf.st.lstate[0] == LState.UNSAT, "one window must not latch"
    rf.tick(obs_hi(True)); rf.tick(obs_hi(True))
    assert rf.st.lstate[0] == LState.SAT_LATCHED
    for _ in range(10):
        rf.tick(obs_lo(False))                         # occluded contradiction: FROZEN
    assert rf.st.lstate[0] == LState.SAT_LATCHED, "occluded p_sat collapse must not roll back"
    rf.tick(obs_lo(True)); rf.tick(obs_lo(True)); rf.tick(obs_lo(True))
    assert rf.st.lstate[0] == LState.UNSAT, "visible contradiction x3 must roll back"
