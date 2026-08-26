"""Step 0 — closure-state probe: was a demo grasp rig-assisted, or does replay drift lose it?

For a demo and a grasp segment, RESTORE the recorded state (no replay) at closure-30, closure,
closure+10, +30, +60 and read: AG object per arm, EE->object distance, finger contact, gripper
qpos. The recording is the human's ground truth: if AG appears within a few frames of closure
while the EE is at contact distance, the grasp was physically honest. Then a SHORT replay
(closure-30 -> closure+60) from the restored state tells whether 90 frames of open-loop control
already lose it (drift), independent of the long pre-segment approach.

Usage:
  ... python -u task62/step0_closure_probe.py --ep 620010 --tags pick_knife,pick_half --out /root/step0
"""
import os, sys, json, argparse, time
import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task62.step0_replay_sweep import (Sim, load_demo, segments, closure_frames, target_for, halves,
                                       chop_spike, log)

OFFS = [-30, -10, 0, 5, 10, 20, 30, 60]


def gripper_qpos(sim, arm):
    q = sim.robot.get_joint_positions()
    idx = sim.robot.gripper_control_idx[arm]
    v = q[idx]
    return [float(x) for x in (v.cpu() if hasattr(v, "cpu") else v)]


def snapshot(sim, arm, tgt):
    return dict(ag=sim.ag(), d=sim.dist(arm, tgt) if tgt is not None else None,
                contact=sim.finger_contact(arm, tgt) if tgt is not None else None,
                grip=gripper_qpos(sim, arm), z=float(sim.obj_pos(tgt)[2]) if tgt is not None else None)


def probe_segment(sim, d, seg, rows):
    a = d["action"]; lo = max(0, seg["start"] - 30); hi = min(len(a), seg["end"] + 60)
    cf = closure_frames(a, lo, hi)
    cands = [(t, arm) for arm in ("left", "right") for t in cf[arm]]
    if not cands:
        rows.append(dict(ep=d["ep"], tag=seg["tag"], error="no closure in segment")); return
    spike = chop_spike(d)
    for c, arm in sorted(cands):
        if c > 0 and seg["post_slice"]:
            # bring halves into existence first (slice in-episode), then restore forward
            sim.ensure_pre_slice_composition()
            sim.restore_sequential(d, max(0, spike - 60))
            for t in range(max(0, spike - 60), spike + 15): sim.step(a[t])
            base = spike + 15
        else:
            sim.ensure_pre_slice_composition(); base = 0
        states = {}
        for off in OFFS:
            f = min(len(a) - 1, max(base, c + off))
            sim.restore_sequential(d, f, start=base if off == OFFS[0] else prev + 1)
            prev = f
            tgt = target_for(seg["tag"], sim, arm=arm)
            states[off] = dict(frame=f, target=getattr(tgt, "name", None), **snapshot(sim, arm, tgt))
        log(f"ep{d['ep']} {seg['tag']} closure@{c} arm={arm}: " + " | ".join(
            f"{k:+d}: ag={v['ag'].get(arm)} d={v['d']:.3f} c={v['contact']} z={v['z']:.3f}" if v['d'] is not None else f"{k:+d}: -"
            for k, v in states.items()))
        # short replay from closure-30
        sim.ensure_pre_slice_composition()          # pre-slice state needs the pre-slice object set
        if seg["post_slice"]:
            sim.restore_sequential(d, max(0, spike - 60))
            for t in range(max(0, spike - 60), spike + 15): sim.step(a[t])
            sim.restore_sequential(d, c - 30, start=spike + 16)
        else:
            sim.restore_sequential(d, c - 30)
        tgt = target_for(seg["tag"], sim, arm=arm); trace = []
        for t in range(c - 30, min(len(a), c + 60)):
            sim.step(a[t])
            if t in (c - 1, c, c + 5, c + 10, c + 20, c + 30, c + 59):
                s = snapshot(sim, arm, tgt); s["t"] = t; trace.append(s)
        ag_engaged = any(s["ag"].get(arm) for s in trace)
        log(f"ep{d['ep']} {seg['tag']} short-replay from {c-30}: ag_engaged={ag_engaged} " + " | ".join(
            f"t{s['t']-c:+d}: d={s['d']:.3f} c={s['contact']} ag={s['ag'].get(arm)}" for s in trace if s['d'] is not None))
        rows.append(dict(ep=d["ep"], tag=seg["tag"], idx=seg["idx"], closure=c, arm=arm,
                         recorded={str(k): v for k, v in states.items()}, short_replay=trace, short_replay_ag=ag_engaged))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=620010)
    ap.add_argument("--tags", default="pick_knife,pick_half")
    ap.add_argument("--out", default="/root/step0")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sim = Sim(); d = load_demo(a.ep); rows = []
    for seg in segments(d):
        if seg["tag"] in a.tags.split(","):
            try: probe_segment(sim, d, seg, rows)
            except Exception as e:
                import traceback; log("ERROR", seg["tag"], traceback.format_exc()[-600:]); rows.append(dict(ep=a.ep, tag=seg["tag"], error=repr(e)))
    with open(f"{a.out}/closure_probe_{a.ep}.json", "w") as f: json.dump(rows, f, indent=1)
    log("PROBE_DONE"); os._exit(0)


if __name__ == "__main__":
    main()
