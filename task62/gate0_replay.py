"""Gate 0 — replay fidelity for task-62 (halve_an_egg), the invariant every physics/scene/restore
change must re-pass.

Protocol: boot the eval-exact env (robot named "robot"), restore a demo's serialized states
SEQUENTIALLY from frame 0 to the anchor (stride 5), then replay the demo's own 23-d actions
from the anchor through the recorded chop spike + slack, and report whether/when the slice
fires. Also measures env.step wall time and the restore drift.

Usage (single sim process per box!):
  env -u DISPLAY OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 XDG_RUNTIME_DIR=/tmp/xdg \
    python -u task62/gate0_replay.py --ep 620010 --offset 100 --slack 60
"""
import os, sys, json, time, argparse
import numpy as np
import h5py
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task62.env import build_env, goal_status, TASK

RAW = "/root/rawdemos/task-0062"
ANN = "/root/t62_annotations/annotations/task-0062"


def load_demo(ep):
    f = h5py.File(f"{RAW}/episode_{ep:08d}.hdf5", "r")
    keys = [k for k in f["data"] if k.startswith("demo_")]
    k = max(keys, key=lambda k: f["data"][k]["action"].shape[0])
    g = f["data"][k]
    return dict(action=g["action"][:], reward=g["reward"][:], state=g["state"], state_size=g["state_size"][:], f=f)


def chop_spike(reward):
    nz = np.nonzero(reward > 0.39)[0]
    return int(nz[0]) if len(nz) else None


def restore_sequential(state, state_size, upto, stride=5):
    """Only early frames carry a full scene state; later frames serialize only awake bodies.
    Loading every k-th frame in order leaves every object at its last-awake pose."""
    import omnigibson as og
    idx = list(range(0, upto, stride)) + [upto]
    for t in idx:
        s = th.from_numpy(np.asarray(state[t][: int(state_size[t])]))
        og.sim.load_state(s, serialized=True)
    og.sim.update_handles()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=620010)
    ap.add_argument("--offset", type=int, default=100, help="anchor = chop spike - offset frames")
    ap.add_argument("--slack", type=int, default=60, help="replay past the recorded spike by this many frames")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default="/root/gate0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import omnigibson as og
    t0 = time.time()
    env, hs = build_env(robot_name="robot")
    print(f"[gate0] env booted in {time.time()-t0:.0f}s; human mean length {hs['length']:.0f}", flush=True)
    robot = env.robots[0]

    d = load_demo(a.ep)
    spike = chop_spike(d["reward"])
    assert spike is not None, "no chop spike in this demo"
    anchor = max(0, spike - a.offset)
    print(f"[gate0] ep {a.ep}: T={len(d['action'])} chop spike @ {spike}, anchor @ {anchor}", flush=True)

    t0 = time.time()
    restore_sequential(d["state"], d["state_size"], anchor, stride=a.stride)
    print(f"[gate0] sequential restore ({anchor//a.stride + 1} frames) in {time.time()-t0:.1f}s", flush=True)

    # what does the restored world look like?
    q0, sat0, unsat0 = goal_status(env)
    ag = {arm: (obj.name if obj is not None else None) for arm, obj in robot._ag_obj_in_hand.items()}
    scope = {k: getattr(v, "name", None) for k, v in env.task.object_scope.items()}
    print(f"[gate0] restored: q={q0:.2f} sat={sat0} ag_in_hand={ag}", flush=True)
    print(f"[gate0] scope: {scope}", flush=True)

    # replay own actions closed-loop-free (raw), watch for the slice
    egg_keys = [k for k in scope if "egg" in k and "half" not in k]
    half_keys = [k for k in scope if "half" in k]
    sliced_at = None; step_times = []; qs = []
    for t in range(anchor, min(spike + a.slack, len(d["action"]))):
        ts = time.time()
        env.step(th.from_numpy(d["action"][t]))
        step_times.append(time.time() - ts)
        if (t - anchor) % 25 == 0 or t >= spike - 5:
            q, sat, _ = goal_status(env); qs.append((t, q))
            if sliced_at is None and q >= 0.39:
                sliced_at = t
                print(f"[gate0] SLICED at t={t} (recorded {spike}, delta {t - spike}) q={q:.2f} sat={sat}", flush=True)
                break
    res = dict(ep=a.ep, spike=spike, anchor=anchor, sliced_at=sliced_at,
               delta=(sliced_at - spike) if sliced_at is not None else None,
               q_restored=q0, ag_in_hand=ag, egg_keys=egg_keys, half_keys=half_keys,
               step_ms=float(np.median(step_times) * 1000) if step_times else None,
               n_steps=len(step_times), q_trace=qs)
    print("[gate0] RESULT", json.dumps(res), flush=True)
    json.dump(res, open(f"{a.out}/gate0_ep{a.ep}_off{a.offset}.json", "w"), indent=1)
    print("GATE0_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
