"""Step 0 — random-policy floor per family from the SAME anchors the replay sweep used (Law 0:
no bar means anything without its floor). Radio's floor audit showed shallow anchors can be
solved by noise (press-10 floor 70%); this measures that for task-62's grasp/place families.

Policy: uniform random joint-position deltas on the ACTIVE arm + trunk (scale = the radio skill
wrapper's per-joint clip, ~0.03-0.08 rad/step), random gripper command in {-1,+1} with 5-step
persistence, base zeroed. Budget = the annotated segment length + 60. Trials per anchor: --trials.

Usage:
  ... python -u task62/step0_random_floor.py --demos 620010,620030,620040 --trials 3 --out /root/step0
"""
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task62.step0_replay_sweep import Sim, load_demo, segments, closure_frames, chop_spike, score, log, PRE

A_TRUNK, A_ARM, A_GRIP = slice(3, 7), {"left": slice(7, 14), "right": slice(15, 22)}, {"left": 14, "right": 22}
SCALE_ARM = np.array([0.0261, 0.0138, 0.0470, 0.0413, 0.0774, 0.0334, 0.08], np.float32)   # skill_wrapper_scale.json
SCALE_TRUNK = np.full(4, 0.01, np.float32)
FAMS = ("pick_egg", "pick_knife", "pick_half", "place_egg_board", "place_knife_sink", "place_half_plate", "chop")


def active_arm(a, lo, hi):
    cf = closure_frames(a, lo, hi)
    cands = [(t, arm) for arm in ("left", "right") for t in cf[arm]]
    return min(cands)[1] if cands else "left"


def random_rollout(sim, d, seg, arm, budget, rng, held_before):
    a0 = d["action"][max(0, seg["start"] - PRE)].copy()
    q = sim.robot.get_joint_positions(); q = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
    r = sim.robot
    idx_arm = r.arm_control_idx[arm]; idx_arm = idx_arm.cpu().numpy() if hasattr(idx_arm, "cpu") else idx_arm
    idx_tr = r.trunk_control_idx; idx_tr = idx_tr.cpu().numpy() if hasattr(idx_tr, "cpu") else idx_tr
    act = a0.copy(); act[0:3] = 0.0
    act[A_ARM[arm]] = q[idx_arm]; act[A_TRUNK] = q[idx_tr]
    grip = 1.0
    for t in range(budget):
        if t % 5 == 0: grip = rng.choice([-1.0, 1.0])
        act[A_ARM[arm]] += rng.uniform(-1, 1, 7) * SCALE_ARM
        act[A_TRUNK] += rng.uniform(-1, 1, 4) * SCALE_TRUNK
        act[A_GRIP[arm]] = grip
        sim.step(act)
        if t % 25 == 0:
            res = score(seg["tag"], sim, held_before)
            if res.get("success"): return True, t
    return bool(score(seg["tag"], sim, held_before).get("success")), budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="620010,620030,620040")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/step0")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed); sim = Sim(); rows = []
    outp = f"{a.out}/random_floor_{a.demos.replace(',', '_')[:40]}.jsonl"
    for ep in [int(x) for x in a.demos.split(",")]:
        d = load_demo(ep); spike = chop_spike(d); segs = [s for s in segments(d) if s["tag"] in FAMS]
        for seg in segs:
            lo = max(0, seg["start"] - PRE); hi = min(len(d["action"]), seg["end"] + 60)
            arm = active_arm(d["action"], lo, hi); budget = hi - lo
            for k in range(a.trials):
                try:
                    sim.ensure_pre_slice_composition()
                    if seg["post_slice"]:
                        sim.restore_sequential(d, max(0, spike - 60))
                        for t in range(max(0, spike - 60), spike + 15): sim.step(d["action"][t])
                        sim.restore_sequential(d, lo, start=spike + 16)
                    else:
                        sim.restore_sequential(d, lo)
                    q0, sat0 = sim.q(); held_before = set(sat0)
                    ok, steps = random_rollout(sim, d, seg, arm, budget, rng, held_before)
                    row = dict(ep=ep, tag=seg["tag"], idx=seg["idx"], arm=arm, trial=k, budget=budget, success=ok, steps=steps, q0=q0)
                except Exception as e:
                    row = dict(ep=ep, tag=seg["tag"], idx=seg["idx"], trial=k, error=repr(e)[:200])
                rows.append(row); log(json.dumps(row))
                with open(outp, "w") as f:
                    for r in rows: f.write(json.dumps(r) + "\n")
    log(f"RANDOM_FLOOR_DONE rows={len(rows)} -> {outp}"); os._exit(0)


if __name__ == "__main__":
    main()
