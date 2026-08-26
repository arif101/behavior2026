"""Step 0 — drift attribution: WHICH part of the state diverges during open-loop replay?

For a demo and a grasp segment: (1) restore at anchor = segment start − PRE and replay the demo's
actions to the closure frame c; record base position/yaw, trunk qpos, arm qpos, EE position, and
the target object's pose. (2) Restore the RECORDED state at c and record the same. Report the
per-component deltas. Repeat for several anchors (closure − {30, 60, 120, 230, segment}) to see
how drift grows with replay length.


Usage:
  ... python -u task62/step0_drift_probe.py --ep 620010 --tag pick_knife --out /root/step0
"""
import os, sys, json, argparse
import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task62.step0_replay_sweep import Sim, load_demo, segments, closure_frames, target_for, chop_spike, log, PRE
import omnigibson.utils.transform_utils as T


def _np(x):
    return np.asarray(x.cpu() if hasattr(x, "cpu") else x, dtype=np.float64)


def pose_snapshot(sim, arm, tgt):
    r = sim.robot
    bp, bq = r.get_position_orientation()
    yaw = float(T.quat2euler(bq)[2]) if hasattr(T, "quat2euler") else float(T.mat2euler(T.quat2mat(bq))[2])
    q = _np(r.get_joint_positions())
    ia = r.arm_control_idx[arm]; it = r.trunk_control_idx
    op, oq = tgt.get_position_orientation()
    return dict(base=_np(bp).tolist(), yaw=yaw, trunk=q[_np(it).astype(int)].tolist(), arm=q[_np(ia).astype(int)].tolist(),
                eef=_np(r.get_eef_position(arm)).tolist(), obj=_np(op).tolist(), obj_quat=_np(oq).tolist(),
                ag=sim.ag())


def diff(a, b):
    d = {}
    for k in ("base", "eef", "obj"):
        d[k + "_cm"] = round(float(np.linalg.norm(np.array(a[k]) - np.array(b[k])) * 100), 2)
    d["yaw_deg"] = round(float(np.degrees(abs(a["yaw"] - b["yaw"]))), 2)
    d["trunk_rad_max"] = round(float(np.max(np.abs(np.array(a["trunk"]) - np.array(b["trunk"])))), 4)
    d["arm_rad_max"] = round(float(np.max(np.abs(np.array(a["arm"]) - np.array(b["arm"])))), 4)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=620010)
    ap.add_argument("--tag", default="pick_knife")
    ap.add_argument("--offsets", default="30,60,120,230")
    ap.add_argument("--out", default="/root/step0")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    sim = Sim(); d = load_demo(a.ep); act = d["action"]
    seg = next(s for s in segments(d) if s["tag"] == a.tag)
    lo = max(0, seg["start"] - PRE); hi = min(len(act), seg["end"] + 60)
    cf = closure_frames(act, lo, hi); c, arm = min((t, arm) for arm in ("left", "right") for t in cf[arm])
    log(f"ep{a.ep} {a.tag}: segment [{seg['start']},{seg['end']}] closure {c} arm {arm}")
    assert not seg["post_slice"], "drift probe currently supports pre-slice segments only"

    # recorded state at closure (ground truth)
    sim.ensure_pre_slice_composition(); sim.restore_sequential(d, c)
    tgt = target_for(a.tag, sim, arm=arm); rec = pose_snapshot(sim, arm, tgt)
    log(f"recorded@{c}: base={np.round(rec['base'],3).tolist()} yaw={rec['yaw']:.3f} eef={np.round(rec['eef'],3).tolist()} obj={np.round(rec['obj'],3).tolist()}")
    # also recorded state at the anchor frames (to separate restore error from replay drift)
    rows = []
    for off in [int(x) for x in a.offsets.split(",")] + [c - lo]:
        anchor = max(0, c - off)
        sim.ensure_pre_slice_composition(); sim.restore_sequential(d, anchor)
        tgt = target_for(a.tag, sim, arm=arm); at_anchor = pose_snapshot(sim, arm, tgt)
        for t in range(anchor, c):
            sim.step(act[t])
        rep = pose_snapshot(sim, arm, tgt)
        # continue to closure+25 to see whether AG fires
        for t in range(c, min(len(act), c + 25)): sim.step(act[t])
        ag_after = sim.ag().get(arm)
        dd = diff(rep, rec)
        row = dict(ep=a.ep, tag=a.tag, anchor=anchor, replay_frames=c - anchor, closure=c, arm=arm,
                   delta_vs_recorded=dd, ag_after_closure=ag_after,
                   at_anchor=at_anchor, replayed=rep, recorded=rec)
        rows.append(row)
        log(f"anchor {anchor} ({c-anchor} frames): {json.dumps(dd)} ag_after={ag_after}")
    json.dump(rows, open(f"{a.out}/drift_probe_{a.ep}_{a.tag}.json", "w"), indent=1)
    log("DRIFT_PROBE_DONE"); os._exit(0)


if __name__ == "__main__":
    main()
