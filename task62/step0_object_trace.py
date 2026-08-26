"""Step 0 — object-displacement trace: WHEN and WHY does the target object move during replay?

Restore at an anchor, replay to the closure frame, log the target object's position every N
frames plus whether any robot link is in contact with it and whether the object is asleep.
Also logs the object's position drift over a pure SETTLE (no actions) after the restore, to
separate restore artifacts from robot contact.

Usage:
  ... python -u task62/step0_object_trace.py --ep 620010 --tag pick_knife --anchor-off 230 --out /root/step0
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task62.step0_replay_sweep import Sim, load_demo, segments, closure_frames, target_for, log, PRE


def contacts_with(sim, obj):
    """Names of robot links currently in contact with obj (RigidContactAPI via obj.states or links)."""
    try:
        names = set()
        for link in obj.links.values():
            for c in link.contact_list():
                for p in (c.body0, c.body1):
                    p = str(p)
                    if "robot" in p and p not in names: names.add(p.split("/")[-1])
        return sorted(names)
    except Exception as e:
        return [f"err:{type(e).__name__}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=620010)
    ap.add_argument("--tag", default="pick_knife")
    ap.add_argument("--anchor-off", type=int, default=230)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--settle", type=int, default=60)
    ap.add_argument("--out", default="/root/step0")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sim = Sim(); d = load_demo(a.ep); act = d["action"]
    seg = next(s for s in segments(d) if s["tag"] == a.tag)
    lo = max(0, seg["start"] - PRE); hi = min(len(act), seg["end"] + 60)
    cf = closure_frames(act, lo, hi); c, arm = min((t, arm) for arm in ("left", "right") for t in cf[arm])
    anchor = max(0, c - a.anchor_off)
    sim.ensure_pre_slice_composition(); sim.restore_sequential(d, anchor)
    tgt = target_for(a.tag, sim, arm=arm); p0 = sim.obj_pos(tgt).copy()
    log(f"ep{a.ep} {a.tag} anchor {anchor} closure {c} arm {arm} obj0={np.round(p0,4).tolist()}")
    # 1) pure settle: does the object move with NO actions? (restore artifact)
    settle = []
    for i in range(a.settle):
        sim.og.sim.step_physics()
        if (i + 1) % 10 == 0:
            settle.append((i + 1, float(np.linalg.norm(sim.obj_pos(tgt) - p0) * 100)))
    log(f"settle drift (cm) over {a.settle} physics steps: {settle}")
    # re-restore so the replay starts from the same anchor state
    sim.ensure_pre_slice_composition(); sim.restore_sequential(d, anchor)
    tgt = target_for(a.tag, sim, arm=arm); p0 = sim.obj_pos(tgt).copy()
    # 2) replay with per-frame object trace + robot contacts
    trace = []; last = p0.copy(); jumps = []
    for t in range(anchor, c):
        sim.step(act[t])
        p = sim.obj_pos(tgt); step_move = float(np.linalg.norm(p - last) * 100); last = p.copy()
        if step_move > 0.2:
            cts = contacts_with(sim, tgt); jumps.append((t, round(step_move, 2), cts))
            log(f"  JUMP t={t}: {step_move:.2f} cm this frame, cumulative {np.linalg.norm(p-p0)*100:.2f} cm, robot contacts={cts}")
        if (t - anchor) % a.every == 0:
            trace.append(dict(t=t, cum_cm=round(float(np.linalg.norm(p - p0) * 100), 3), pos=np.round(p, 4).tolist(),
                              contacts=contacts_with(sim, tgt), eef_cm=round(sim.dist(arm, tgt) * 100, 2)))
    log(f"total displacement at closure: {np.linalg.norm(last-p0)*100:.2f} cm; jumps={jumps[:10]}")
    json.dump(dict(ep=a.ep, tag=a.tag, anchor=anchor, closure=c, arm=arm, settle=settle, jumps=jumps, trace=trace),
              open(f"{a.out}/object_trace_{a.ep}_{a.tag}_off{a.anchor_off}.json", "w"), indent=1)
    log("OBJECT_TRACE_DONE"); os._exit(0)


if __name__ == "__main__":
    main()
