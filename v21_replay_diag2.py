"""Replay from PLAYBACK restore (not snapshot): does tracking hold and the press fire?"""
import json, os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

def main():
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    import h5py
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from omnigibson.object_states import ToggledOn
    from skill_env_wrapper import P, _np
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000070.hdf5",
        output_path="/root/replaydiag2_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    man = json.load(open("/root/snapshot_bank_v21/manifest_d70.json"))
    s0 = [r for r in man["entries"] if str(r["stage"]) == "0"][0]
    f0 = s0["frame"]
    restore_to_frame(wrapper, 0, f0)
    with h5py.File(wrapper.input_hdf5.filename, "r") as f:
        key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
        acts = f[f"data/{key}/action"][f0:f0+80]
    try:
        if radio.states[ToggledOn].get_value():
            radio.states[ToggledOn].set_value(False)
    except KeyError:
        pass
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
    armL = P["left"]["arm_qpos"]
    succ = False
    for i, a in enumerate(acts):
        cmd = np.asarray(a, np.float32)
        wrapper.env.step(cmd)
        if i % 5 == 0 or i == len(acts)-1:
            err = float(np.abs(qpos()[armL] - cmd[7:14]).max())
            print(f"RD2 step{i:02d} trackerr={err:.4f}", flush=True)
        if radio.states[ToggledOn].get_value():
            succ = True
            print(f"RD2 PRESSED at step {i}", flush=True)
            break
    print(f"RD2_VERDICT playback_restore_replay_press={succ}", flush=True)
    os._exit(0)

main()
