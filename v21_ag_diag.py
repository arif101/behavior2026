"""Does a serialized V2.1 snapshot restore the AG constraint natively — and does
re-establishing on top break it? Test on d70 s0 (grip-fidelity-perfect state)."""
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
    from omnigibson.object_states import ToggledOn
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000070.hdf5",
        output_path="/root/agdiag_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    robot = wrapper.scene.robots[0]
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    man = json.load(open("/root/snapshot_bank_v21/manifest_d70.json"))
    s0 = [r for r in man["entries"] if str(r["stage"]) == "0"][0]
    z = np.load(s0["snapshot"])
    snap = th.as_tensor(z["state"])

    for variant in ("no_repin", "with_repin"):
        og.sim.load_state(snap, serialized=True)
        ag0 = getattr(robot, "_ag_obj_constraint_params", {}).get("left")
        if variant == "with_repin":
            # replicate the wrapper's re-pin
            from skill_env_wrapper_v2 import SkillCommitEnvV2
            env = SkillCommitEnvV2.__new__(SkillCommitEnvV2)
            env.robot, env.target, env.w = robot, radio, wrapper
            env._reestablish_ag("left")
        ag1 = getattr(robot, "_ag_obj_constraint_params", {}).get("left")
        # replay demo actions -> should press within 80 steps from press-5
        with h5py.File(wrapper.input_hdf5.filename, "r") as f:
            key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
            acts = f[f"data/{key}/action"][s0["frame"]:s0["frame"] + 80]
        try:
            if radio.states[ToggledOn].get_value():
                radio.states[ToggledOn].set_value(False)
        except KeyError:
            pass
        succ = False
        for a in acts:
            wrapper.env.step(np.asarray(a, np.float32))
            if radio.states[ToggledOn].get_value():
                succ = True
                break
        print(f"AGDIAG {variant}: ag_after_load={'YES' if ag0 else 'NO'} "
              f"ag_after={'YES' if ag1 else 'NO'} demo_replay_press={succ}", flush=True)
    print("AGDIAG_DONE", flush=True)
    os._exit(0)

main()
