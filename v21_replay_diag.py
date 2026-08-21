"""Replay-divergence diagnostic: step demo actions from the d70 s0 snapshot and compare
the achieved joint trajectory against the demo's own recorded trajectory, per step."""
import json, os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

def main():
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    import torch as th, h5py
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
    from skill_env_wrapper import P, _np
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000070.hdf5",
        output_path="/root/replaydiag_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    man = json.load(open("/root/snapshot_bank_v21/manifest_d70.json"))
    s0 = [r for r in man["entries"] if str(r["stage"]) == "0"][0]
    f0 = s0["frame"]
    z = np.load(s0["snapshot"]); snap = th.as_tensor(z["state"])
    og.sim.load_state(snap, serialized=True)

    with h5py.File(wrapper.input_hdf5.filename, "r") as f:
        key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
        acts = f[f"data/{key}/action"][f0:f0+30]
        # demo's own recorded proprio at those frames
        demo_obs = f[f"data/{key}/obs"]["proprio"][f0:f0+31] if "obs" in f[f"data/{key}"] \
            else None
    def qpos():
        obs = wrapper.env.get_obs()[0]
        def find(node, sub):
            if isinstance(node, dict):
                for k, v in node.items():
                    r = find(v, sub)
                    if r is not None: return r
                    if sub in str(k): return v
            return None
        return _np(find(obs, "proprio")).reshape(-1)
    print(f"REPLAYDIAG f0={f0} demo_obs={'YES' if demo_obs is not None else 'NO'}", flush=True)
    s = qpos()
    armL = P["left"]["arm_qpos"]
    for i, a in enumerate(acts):
        cmd = np.asarray(a, np.float32)
        wrapper.env.step(cmd)
        s = qpos()
        ach = s[armL]
        cmd_arm = cmd[7:14]
        track_err = float(np.abs(ach - cmd_arm).max())
        line = f"REPLAYDIAG step{i:02d} cmd-vs-achieved maxerr={track_err:.4f}"
        if demo_obs is not None:
            demo_arm = np.asarray(demo_obs[i+1]).reshape(-1)[armL]
            line += f" achieved-vs-demo maxerr={float(np.abs(ach-demo_arm).max()):.4f}"
        if i % 5 == 0 or i == 29:
            print(line, flush=True)
    print("REPLAYDIAG_DONE", flush=True)
    os._exit(0)

main()
