"""d110 seeding diagnostic: replay demo actions from each snapshot with randomize OFF.
Also report restored radio pose vs the manifest's (captured at bank build)."""
import json, os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

def main():
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
    from skill_env_wrapper_v2 import SkillCommitEnvV2, load_snapshot_bank
    from skill_env_wrapper import _np
    ents = [e for e in load_snapshot_bank() if e["demo"] == 110]
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000110.hdf5",
        output_path="/root/seed_diag_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    env = SkillCommitEnvV2(wrapper, ents, l2_on=True, seed=0, randomize=False)
    man = json.load(open("/root/snapshot_bank/manifest_d110.json"))["scene"]
    for e in ents:
        trans, ok = env.seed_from_demo(e, max_steps=150)
        tp, _ = env.target.get_position_orientation()
        print(f"SEEDDIAG d110 s{e['stage']} f{e['frame']}: success={ok} steps={len(trans)} "
              f"dist_end={env._dist(env._proprio61()):.3f} "
              f"radio_now={[round(float(x),3) for x in _np(tp)]} "
              f"radio_manifest={man['radio_pos']}", flush=True)
    print("SEEDDIAG_DONE", flush=True)
    os._exit(0)

main()
