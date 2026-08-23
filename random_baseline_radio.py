"""Frozen RANDOM-POLICY baselines for the radio rungs (retroactive; task-62 discipline).

Uniform random a~U[-1,1]^12 per step, current wrapper semantics (still-hold grasp,
assist at the persisted per-scene radius for G, honest press). Pre-registered floors:
a rung is discriminative only if its trained rate >> its random floor.

Run (pauses training):  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
    python -u random_baseline_radio.py --n 20
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
    from skill_env_wrapper_v2 import SkillCommitEnvV2, load_snapshot_bank

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000020.hdf5",
        output_path="/root/rb_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    bank = [e for e in load_snapshot_bank("/root/snapshot_bank_v21")
            if e["demo"] in (20, 30)]
    env = SkillCommitEnvV2(wrapper, bank, l2_on=True, seed=7)
    try:
        ar = json.load(open("/root/v22a_run/stats.json")).get("assist_range", {})
    except FileNotFoundError:
        ar = {}
    rng = np.random.default_rng(7)
    results = {}
    for stage in ("0", "1", "3", "G"):
        ents = [e for e in bank if str(e["stage"]) == stage]
        if not ents:
            continue
        succ, n = 0, 0
        for k in range(a.n):
            e = ents[k % len(ents)]
            env.ag_assist_range = (float(ar.get(str(e["demo"]), 0.30))
                                   if e["family"] == "pick_up_from" else 0.0)
            env.reset(entry=e)
            done, info = False, {}
            while not done:
                act = rng.uniform(-1, 1, 12).astype(np.float32)
                _, _, done, info = env.step(act)
            succ += int(info["success"]); n += 1
            print(f"RB s{stage} ep{k}: {'S' if info['success'] else 'f'} "
                  f"dist={info['dist']:.3f}", flush=True)
        results[stage] = (succ, n)
        print(f"RB_RUNG s{stage}: {succ}/{n} = {succ/max(n,1):.3f}", flush=True)
    print("RB_SUMMARY", json.dumps({k: f"{s}/{n}" for k, (s, n) in results.items()}),
          flush=True)
    os._exit(0)


main()
