"""Step 0 — per-family replay-fidelity sweep for task-62 under eval-exact physics.

For each demo and each scoring-relevant manipulation segment (organizers' annotations), restore
the demo state at (segment start − pre) and replay the demo's own 23-d actions through the
segment (+ slack). Score the family predicate at the end. Record, for grasps, the closure frame
and EE→object distance / AG engagement so the "magnetic rig" hypothesis is measured, not assumed.

Modes:
  isolated   cold sequential restore per segment (post-slice segments: restore chop anchor,
             replay through the slice, then try to restore the segment anchor; on failure,
             continue replaying from the slice = 'chain' fallback, recorded as such)
  continuous one restore at the first segment's anchor, replay straight through to the end of
             the episode, evaluate each segment's predicate at its end frame (drift exposure)

Single sim process per box. Usage:
  env -u DISPLAY OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg \
    PYTHONPATH=/root/behavior2026 python -u task62/step0_replay_sweep.py \
    --demos 620010,620030,620040 --mode isolated --out /root/step0
"""
import os, sys, json, time, argparse, traceback
import numpy as np
import h5py
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task62.env import build_env, goal_status

RAW = "/root/rawdemos/task-0062"
ANN = "/root/t62_annotations/annotations/task-0062"
GL, GR = 14, 22                      # 23-d layout: grip_L=14, grip_R=22, ±1 (closed = -1)
PRE, SLACK = 30, 60                  # frames before segment start / after segment end
SCOPE = {"egg": "hard-boiled_egg.n.01_1", "knife": "carving_knife.n.01_1", "plate": "plate.n.04_1",
         "sink": "sink.n.01_1", "fridge": "electric_refrigerator.n.01_1", "board": "chopping_board.n.01_1",
         "half1": "half__hard-boiled_egg.n.01_1", "half2": "half__hard-boiled_egg.n.01_2"}
FAMILIES = [  # (skill_description, object_id[0][0], tag)
    ("open door", "fridge", "open_fridge"), ("pick up from", "hard_boiled_egg", "pick_egg"),
    ("close door", "fridge", "close_fridge"), ("pick up from", "carving_knife", "pick_knife"),
    ("place on", "hard_boiled_egg", "place_egg_board"), ("chop", "carving_knife", "chop"),
    ("place in", "carving_knife", "place_knife_sink"), ("pick up from", "half_hard_boiled_egg", "pick_half"),
    ("place on", "half_hard_boiled_egg", "place_half_plate"),
]


def log(*a):
    print("[step0]", *a, flush=True)


# ----------------------------------------------------------------------------- demo I/O
def load_demo(ep):
    f = h5py.File(f"{RAW}/episode_{ep:08d}.hdf5", "r")
    keys = [k for k in f["data"] if k.startswith("demo_")]
    k = max(keys, key=lambda k: f["data"][k]["action"].shape[0])
    g = f["data"][k]
    ann = json.load(open(f"{ANN}/episode_{ep:08d}.json"))["skill_annotation"]
    return dict(ep=ep, action=g["action"][:], reward=g["reward"][:], state=g["state"],
                state_size=g["state_size"][:], ann=ann, f=f)


def segments(d):
    """Ordered list of scoring-relevant segments with their tags (post-slice flagged)."""
    out = []; seen_chop = False
    for s in d["ann"]:
        k = s["skill_description"][0]; o = s["object_id"][0][0] if s["object_id"] else None
        for sk, so, tag in FAMILIES:
            if k == sk and o == so:
                a, b = s["frame_duration"]
                out.append(dict(tag=tag, start=a, end=b, post_slice=seen_chop, idx=s["skill_idx"]))
        if k == "chop": seen_chop = True
    return out


def chop_spike(d):
    nz = np.nonzero(d["reward"] > 0.39)[0]
    return int(nz[0]) if len(nz) else None


def closure_frames(a, lo, hi):
    """Gripper open→closed crossings per arm within [lo, hi)."""
    res = {}
    for arm, ch in (("left", GL), ("right", GR)):
        s = np.sign(a[lo:hi, ch]); c = np.nonzero((s[1:] < 0) & (s[:-1] >= 0))[0] + 1 + lo
        res[arm] = c.tolist()
    return res


# ----------------------------------------------------------------------------- sim helpers
class Sim:
    def __init__(self):
        import omnigibson as og
        self.og = og
        t0 = time.time()
        self.env, self.hs = build_env(robot_name="robot")
        self.robot = self.env.robots[0]
        log(f"env booted in {time.time()-t0:.0f}s")
        self.step_ms = []

    def obj(self, key):
        return self.env.task.object_scope.get(SCOPE.get(key, key))

    def board(self):
        return self.obj("board")

    def halves_exist(self):
        return any(self.obj(k) is not None and getattr(self.obj(k), "name", None) for k in ("half1", "half2"))

    def ensure_pre_slice_composition(self):
        """Snapshots only load into a matching object set; after an in-sim slice the egg is gone
        and halves exist. env.reset() + purge of transition-created object info restores it."""
        if not self.halves_exist():
            return True
        try:
            self.env.reset()
            try:
                api = self.env.scene.transition_rule_api
                if api is not None:
                    api.obj_init_info = dict()
            except Exception as e:
                log("purge warning:", repr(e))
            self.og.sim.update_handles()
            ok = not self.halves_exist()
            log("composition reset ->", "pre-slice OK" if ok else "HALVES STILL PRESENT")
            return ok
        except Exception as e:
            log("composition reset FAILED:", repr(e)); return False

    def restore_sequential(self, d, upto, start=0, stride=5):
        idx = list(range(start, upto, stride)) + [upto]
        for t in idx:
            s = th.from_numpy(np.asarray(d["state"][t][: int(d["state_size"][t])]))
            self.og.sim.load_state(s, serialized=True)
        self.og.sim.update_handles()

    closed_loop = False          # set by --replay closed
    TRACK_TOL, MAX_SUB = 0.03, 8  # playbook: hold each absolute target <=8 substeps until |q-cmd|<0.03

    def _tracking_err(self, a23):
        r = self.robot; q = r.get_joint_positions()
        q = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
        errs = [np.abs(q[r.trunk_control_idx.cpu().numpy() if hasattr(r.trunk_control_idx, "cpu") else r.trunk_control_idx] - a23[3:7]).max()]
        for arm, sl in (("left", slice(7, 14)), ("right", slice(15, 22))):
            idx = r.arm_control_idx[arm]; idx = idx.cpu().numpy() if hasattr(idx, "cpu") else idx
            errs.append(np.abs(q[idx] - a23[sl]).max())
        return float(max(errs))

    def step(self, a23):
        """One demo frame. Open-loop = one env.step. Closed-loop = re-issue the same absolute target
        until the arms/trunk track it (drift fix from SYSTEM_PLAYBOOK §2.6)."""
        a23 = np.asarray(a23, dtype=np.float32)
        ts = time.time()
        self.env.step(th.from_numpy(a23))
        if self.closed_loop:
            for _ in range(self.MAX_SUB - 1):
                if self._tracking_err(a23) < self.TRACK_TOL: break
                self.env.step(th.from_numpy(a23))
        self.step_ms.append((time.time() - ts) * 1000)

    # ---- measurements
    def ag(self):
        return {arm: (o.name if o is not None else None) for arm, o in self.robot._ag_obj_in_hand.items()}

    def eef_pos(self, arm):
        return np.asarray(self.robot.get_eef_position(arm).cpu() if hasattr(self.robot.get_eef_position(arm), "cpu") else self.robot.get_eef_position(arm))

    def obj_pos(self, o):
        p = o.get_position_orientation()[0]
        return np.asarray(p.cpu() if hasattr(p, "cpu") else p)

    def dist(self, arm, o):
        return float(np.linalg.norm(self.eef_pos(arm) - self.obj_pos(o)))

    def finger_contact(self, arm, o):
        try:
            contacts = self.robot._find_gripper_contacts(arm)
            paths = contacts[0] if isinstance(contacts, tuple) else contacts
            paths = set(str(p) for p in (paths.keys() if isinstance(paths, dict) else paths))
            link_paths = set(l.prim_path for l in o.links.values())
            return any(p in link_paths or o.name in p for p in paths)
        except Exception:
            return None

    def fridge_open(self):
        from omnigibson.object_states import Open
        try: return bool(self.obj("fridge").states[Open].get_value())
        except Exception: return None

    def ontop(self, o, ref):
        from omnigibson.object_states import OnTop
        try: return bool(o.states[OnTop].get_value(ref))
        except Exception: return None

    def inside(self, o, ref):
        from omnigibson.object_states import Inside
        try: return bool(o.states[Inside].get_value(ref))
        except Exception: return None

    def q(self):
        q, s, u = goal_status(self.env); return q, list(s)


# ----------------------------------------------------------------------------- scoring
def halves(sim):
    return [sim.obj(k) for k in ("half1", "half2") if sim.obj(k) is not None and getattr(sim.obj(k), "name", None)]


def target_for(tag, sim, arm=None):
    """For the half families the target is whichever half is NOT already held (or, given an arm,
    the half nearest that arm's EE) — the two halves are interchangeable in the BDDL goal."""
    if tag in ("pick_egg", "place_egg_board"): return sim.obj("egg")
    if tag in ("pick_knife", "chop", "place_knife_sink"): return sim.obj("knife")
    if tag in ("pick_half", "place_half_plate"):
        hs = halves(sim)
        if not hs: return None
        held = set(v for v in sim.ag().values() if v)
        free = [h for h in hs if h.name not in held] or hs
        if arm is not None:
            return min(free, key=lambda h: sim.dist(arm, h))
        return free[0]
    return None


def score(tag, sim, held_before):
    q, sat = sim.q(); ag = sim.ag()
    r = dict(q=q, sat=sat, ag=ag)
    if tag == "open_fridge": r["success"] = sim.fridge_open() is True
    elif tag == "close_fridge": r["success"] = sim.fridge_open() is False
    elif tag == "pick_half":
        names = [h.name for h in halves(sim)]
        newly = [v for v in ag.values() if v in names and v not in held_before]
        r["success"] = len(newly) > 0; r["newly_held"] = newly
    elif tag.startswith("pick_"):
        o = target_for(tag, sim); r["success"] = o is not None and o.name in ag.values()
    elif tag == "place_egg_board":
        o, b = sim.obj("egg"), sim.board(); r["ontop_board"] = sim.ontop(o, b) if (o and b) else None
        r["success"] = bool(r["ontop_board"]) and (o.name not in ag.values())
    elif tag == "chop": r["success"] = 0 in sat and 1 in sat
    elif tag == "place_knife_sink": r["success"] = 4 in sat
    elif tag == "place_half_plate":
        gained = [i for i in (2, 3) if i in sat and i not in held_before]
        r["success"] = len(gained) > 0; r["gained"] = gained
    return r


def replay_segment(sim, d, seg, out_rows, mode, held_before):
    """Replay frames [seg.start-PRE, seg.end+SLACK) (already restored/positioned) and score."""
    a = d["action"]; lo = max(0, seg["start"] - PRE); hi = min(len(a), seg["end"] + SLACK)
    tgt = target_for(seg["tag"], sim)
    cf = closure_frames(a, lo, hi)
    grasp = dict(closure=None, arm=None, d_at_closure=None, contact_at_closure=None, ag_frame=None, d_at_ag=None, z0=None, z_end=None)
    if tgt is not None and seg["tag"].startswith("pick_"):
        grasp["z0"] = float(sim.obj_pos(tgt)[2])
    ag_before = sim.ag(); held_before_names = set(v for v in ag_before.values() if v)
    for t in range(lo, hi):
        sim.step(a[t])
        if tgt is not None and seg["tag"].startswith("pick_"):
            for arm in ("left", "right"):
                if t in cf[arm] and grasp["closure"] is None:
                    tgt = target_for(seg["tag"], sim, arm=arm) or tgt      # half nearest the closing arm
                    grasp.update(closure=t, arm=arm, target=tgt.name, d_at_closure=sim.dist(arm, tgt),
                                 contact_at_closure=sim.finger_contact(arm, tgt), z0=float(sim.obj_pos(tgt)[2]))
                cur = sim.ag().get(arm)
                if grasp["ag_frame"] is None and cur and cur not in held_before_names and cur != ag_before.get(arm):
                    o = tgt if cur == tgt.name else next((h for h in halves(sim) if h.name == cur), tgt)
                    grasp.update(ag_frame=t, d_at_ag=sim.dist(arm, o), arm=grasp["arm"] or arm, target=o.name)
        # early exit for chop once sliced
        if seg["tag"] == "chop" and t >= seg["start"] and (t - lo) % 10 == 0:
            qq, ss = sim.q()
            if 0 in ss and 1 in ss:
                grasp["sliced_at"] = t; break
    if tgt is not None and seg["tag"].startswith("pick_") and getattr(tgt, "name", None):
        grasp["z_end"] = float(sim.obj_pos(tgt)[2])
    res = score(seg["tag"], sim, held_before)
    row = dict(ep=d["ep"], mode=mode, replay="closed" if sim.closed_loop else "open", tag=seg["tag"], idx=seg["idx"],
               start=seg["start"], end=seg["end"], replayed=[lo, hi], post_slice=seg["post_slice"], **res, grasp=grasp)
    out_rows.append(row); log(json.dumps(row))
    return res


def run_isolated(sim, d, rows):
    segs = segments(d); spike = chop_spike(d)
    if not sim.ensure_pre_slice_composition():
        rows.append(dict(ep=d["ep"], mode="isolated", error="composition")); return
    held = set()
    # pre-slice segments: cold restore each
    for seg in [s for s in segs if not s["post_slice"]]:
        try:
            sim.ensure_pre_slice_composition()
            t0 = time.time(); sim.restore_sequential(d, max(0, seg["start"] - PRE)); rt = time.time() - t0
            log(f"ep{d['ep']} {seg['tag']}: restored to {seg['start']-PRE} in {rt:.0f}s, ag={sim.ag()}, q={sim.q()[0]:.2f}")
            r = replay_segment(sim, d, seg, rows, "isolated", held); held |= set(r.get("sat", []))
        except Exception as e:
            rows.append(dict(ep=d["ep"], mode="isolated", tag=seg["tag"], error=repr(e))); log("ERROR", seg["tag"], traceback.format_exc()[-800:])
    # post-slice segments: slice in-episode first, then try cold restore of the anchor
    post = [s for s in segs if s["post_slice"]]
    if not post or spike is None: return
    try:
        sim.ensure_pre_slice_composition()
        sim.restore_sequential(d, max(0, spike - 60))
        for t in range(max(0, spike - 60), spike + 15): sim.step(d["action"][t])
        q, sat = sim.q(); log(f"ep{d['ep']} in-episode slice: q={q:.2f} sat={sat} halves={sim.halves_exist()}")
        held = set(sat); cur = spike + 15
        for seg in post:
            anchor = max(cur, seg["start"] - PRE); restored = False
            try:
                sim.restore_sequential(d, anchor, start=cur + 1); restored = True
                log(f"ep{d['ep']} {seg['tag']}: post-slice cold restore to {anchor} OK")
            except Exception as e:
                log(f"ep{d['ep']} {seg['tag']}: post-slice restore FAILED ({type(e).__name__}); chain-replaying {cur}->{anchor}")
                for t in range(cur, anchor): sim.step(d["action"][t])
            r = replay_segment(sim, d, seg, rows, "isolated" if restored else "chain", held); held |= set(r.get("sat", []))
            cur = min(len(d["action"]), seg["end"] + SLACK)
    except Exception as e:
        rows.append(dict(ep=d["ep"], mode="isolated", tag="post_slice", error=repr(e))); log("ERROR post-slice", traceback.format_exc()[-800:])


def run_continuous(sim, d, rows):
    segs = segments(d)
    if not sim.ensure_pre_slice_composition() or not segs:
        rows.append(dict(ep=d["ep"], mode="continuous", error="composition")); return
    first = max(0, segs[0]["start"] - PRE); sim.restore_sequential(d, first)
    a = d["action"]; held = set(); t = first; qtrace = []
    for seg in segs:
        hi = min(len(a), seg["end"] + SLACK)
        while t < hi:
            sim.step(a[t]); t += 1
            if t % 200 == 0: qtrace.append((t, sim.q()[0]))
        r = score(seg["tag"], sim, held); held |= set(r.get("sat", []))
        row = dict(ep=d["ep"], mode="continuous", tag=seg["tag"], idx=seg["idx"], end=seg["end"], at=t, **r)
        rows.append(row); log(json.dumps(row))
    rows.append(dict(ep=d["ep"], mode="continuous", tag="q_trace", q_trace=qtrace))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="620010,620030,620040")
    ap.add_argument("--mode", default="isolated", choices=["isolated", "continuous", "both"])
    ap.add_argument("--out", default="/root/step0")
    ap.add_argument("--replay", default="open", choices=["open", "closed"], help="open-loop (1 env.step/frame) or closed-loop tracking")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sim = Sim(); sim.closed_loop = (a.replay == "closed"); rows = []
    outp = f"{a.out}/replay_sweep_{a.mode}_{a.replay}_{a.demos.replace(',', '_')[:40]}.jsonl"
    for ep in [int(x) for x in a.demos.split(",")]:
        d = load_demo(ep); t0 = time.time()
        if a.mode in ("isolated", "both"): run_isolated(sim, d, rows)
        if a.mode in ("continuous", "both"): run_continuous(sim, d, rows)
        with open(outp, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        log(f"ep{ep} done in {(time.time()-t0)/60:.1f} min; step_ms median={np.median(sim.step_ms):.0f}")
    log(f"SWEEP_DONE rows={len(rows)} -> {outp}")
    os._exit(0)


if __name__ == "__main__":
    main()
