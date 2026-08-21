"""Is the 929->945 approach base translation? Print base pose, right EEF, radio across restores."""
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
        output_path="/root/basediag_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    for t in [909, 929, 937, 945, 955]:
        restore_to_frame(wrapper, 0, t)
        bp = _np(rob.get_position_orientation()[0])
        ep = _np(rob.eef_links["right"].get_position_orientation()[0])
        rp = _np(radio.get_position_orientation()[0])
        print(f"BD t={t} base={np.round(bp,3)} eefR={np.round(ep,3)} radio={np.round(rp,3)} "
              f"eefd={float(np.linalg.norm(ep-rp)):.3f}", flush=True)
    os._exit(0)

main()
