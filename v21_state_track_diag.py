"""Do recorded states track the radio through the lift? And what gripper controller
does the playback env build?"""
import os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

def main():
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000030.hdf5",
        output_path="/root/statetrack_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    print("ST controllers:", {k: type(v).__name__ for k, v in rob.controllers.items()}, flush=True)
    for t in [909, 929, 945, 955, 970, 1000, 1100, 1280]:
        restore_to_frame(wrapper, 0, t)
        rp = _np(radio.get_position_orientation()[0])
        ags = {a: bool(rob._ag_obj_constraint_params.get(a)) for a in rob.arm_names}
        eef = {a: round(float(np.linalg.norm(
            _np(rob.eef_links[a].get_position_orientation()[0]) - rp)), 3)
            for a in rob.arm_names}
        print(f"ST t={t} radio_z={float(rp[2]):.4f} eefd={eef} ag={ags}", flush=True)
    os._exit(0)

main()
