"""STATE-HARVEST PASS (spec §4 [A-2026-08-13]): run the FROZEN VLA from train-demo
restores and snapshot every sim state it visits inside the 10 cm shell -> the
policy-visited start bank (primary source for the gate-boundary curriculum stage).

TRAIN demos only — never instance 301, never the Run-2 eval campaign (pinned).

Per demo: restore to the demo's pre-shell anchor (stage-3 bank entry = quasi-static
~press-100), hand control to the VLA policy server (websocket, same serving stack), step
up to --budget steps, and dump og.sim state (serialized) whenever
dist(closest EE, metalink) < --shell (default 0.10 m), at most every --min-gap steps.
Snapshots carry: demo, step, dmin, AG state, gripper qpos, ToggledOn (must be False —
toggled snapshots are dropped), for wrapper-side validity filtering.

Output: /root/harvest_states/d{demo}_s{step}.npz (state, state_size, meta) +
/root/harvest_bank.json (entry list in the start-bank format with "state_file" set —
skill_env_wrapper gains a load-from-npz path for these).

Serving: expects the run-2 policy server on 127.0.0.1:--port (default 8901; provider
nginx squats 8000). GPU sharing with the policy server is tight on the 24 GB A5000 —
if the sim OOMs, restart the server with JAX_PLATFORMS=cpu (π0.5 CPU inference ~10-30 s
per h=32 chunk -> a 600-step rollout still finishes in minutes).

Run INSIDE the behavior env, one demo per process (Isaac env-reuse leaks):
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    python -u harvest_start_states.py --demo-id 20
"""

import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

P_OFF_PATH = "/root/behavior2026/labels/metalink_offset.json"
EE_L, EE_R = slice(17, 20), slice(42, 45)


def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _np(v):
    return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-id", type=int, required=True)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--shell", type=float, default=0.10)
    ap.add_argument("--min-gap", type=int, default=5)
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--out", default="/root/harvest_states")
    a = ap.parse_args()

    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from omnigibson.eval.utils.network_utils import WebsocketClientPolicy

    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame

    bank = json.load(open("/root/skill_start_bank.json"))
    assert a.demo_id not in bank["held_out_demos"], \
        f"demo {a.demo_id} is HELD OUT -- harvest train demos only"
    anchors = [e for e in bank["entries"]
               if e["demo"] == a.demo_id and e["stage"] == max(
                   x["stage"] for x in bank["entries"] if x["demo"] == a.demo_id)]
    assert anchors, f"no anchor for demo {a.demo_id}"
    anchor = anchors[0]

    p_off = np.asarray(json.load(open(P_OFF_PATH))["offset_pos_root_frame"])
    h5 = f"/root/rawdemos/task-0000/episode_{a.demo_id:08d}.hdf5"
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=h5, output_path=f"/root/harvest_tmp_{a.demo_id}.hdf5",
        robot_obs_modalities=("rgb", "proprio"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS,
    )
    env = wrapper.env
    robot = wrapper.scene.robots[0]
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    epid = int(sorted(k.split("_")[1] for k in wrapper.input_hdf5["data"].keys()
                      if k.startswith("demo_"))[0])
    restore_to_frame(wrapper, epid, anchor["frame"])
    print(f"HARVEST demo {a.demo_id}: restored f{anchor['frame']} "
          f"(holding={anchor['holding_arm']})", flush=True)

    policy = WebsocketClientPolicy(host="127.0.0.1", port=a.port)
    os.makedirs(a.out, exist_ok=True)
    entries, last_snap = [], -10**9
    obs = env.get_obs()[0]

    def dmin_now(prop61):
        tp, tq = radio.get_position_orientation()
        mw = _np(tp) + q2r(_np(tq)) @ p_off
        bp, bq = robot.get_position_orientation()
        mb = q2r(_np(bq)).T @ (mw - _np(bp))
        return float(min(np.linalg.norm(mb - prop61[EE_L]),
                         np.linalg.norm(mb - prop61[EE_R])))

    def prop61(o):
        node = o
        for k in ("robot_r1", "robot_r1::proprio", "proprio"):
            if isinstance(node, dict) and k in node:
                node = node[k]
        return _np(node).reshape(-1)

    from omnigibson.object_states import ToggledOn
    for t in range(a.budget):
        act = policy.act(obs)
        out = env.step(act)
        obs = out[0]
        p = prop61(obs)
        d = dmin_now(p)
        if d < a.shell and t - last_snap >= a.min_gap:
            if bool(radio.states[ToggledOn].get_value()):
                print(f"  t={t} d={d:.3f} SKIP (already toggled)", flush=True)
                continue
            st = og.sim.dump_state(serialized=True)
            st = _np(st)
            ag = getattr(robot, "_ag_obj_constraint_params", {})
            holding = [k for k, v in ag.items() if v is not None]
            fp = f"{a.out}/d{a.demo_id}_s{t}.npz"
            np.savez_compressed(fp, state=st, state_size=np.int64(len(st)))
            entries.append(dict(demo=a.demo_id, state_file=fp, step=int(t),
                                dmin=round(d, 4),
                                holding_arm=(holding[0] if holding else None),
                                active_arm=anchor["active_arm"], stage="boundary",
                                source="vla_harvest"))
            last_snap = t
            print(f"  t={t} d={d:.3f} SNAP ({len(entries)})", flush=True)
    bank_fp = "/root/harvest_bank.json"
    existing = json.load(open(bank_fp)) if os.path.exists(bank_fp) else {"entries": []}
    existing["entries"] = [e for e in existing["entries"]
                           if e["demo"] != a.demo_id] + entries
    json.dump(existing, open(bank_fp, "w"), indent=2)
    print(f"HARVEST_DONE demo={a.demo_id} snapshots={len(entries)}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
