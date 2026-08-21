"""Measure the mapping between bank frames (LeRobot-indexed) and raw-hdf5 state indices:
restore at a sweep of raw indices, compare restored 61-d proprio to LeRobot obs.state
at the bank's closure/anchor frames for demo 30."""
import os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

def main():
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    import pyarrow.parquet as pq
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np
    t = pq.read_table("/root/b1k_radio_map/data/chunk-000/file-000.parquet",
                      columns=["episode_index", "observation.state"])
    m = t["episode_index"].to_numpy() == 2   # demo 30
    L = np.stack(t["observation.state"].to_numpy()[m])
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000030.hdf5",
        output_path="/root/offsetdiag_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
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
    targets = {"closure929": L[929], "anchor945": L[945]}
    for traw in range(849, 1031, 10):
        restore_to_frame(wrapper, 0, traw)
        p = qpos()
        d = {k: float(np.linalg.norm(p - v)) for k, v in targets.items()}
        print(f"FO raw{traw}: d_closure929={d['closure929']:.3f} "
              f"d_anchor945={d['anchor945']:.3f}", flush=True)
    # fine sweep around whichever coarse point was best is left to reading the output
    os._exit(0)

main()
