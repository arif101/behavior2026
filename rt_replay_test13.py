"""Round 13: does the DEMO'S OWN held pose pass native assisted-grasp checks?

R12: replayed cage reaches inside the radio's AABB but the between-fingers
raycast (inhand) fires on ZERO frames. Discriminator: restore the recording's
ground-truth frames around the held window and ask the native checks directly.
  inhand True on GT  -> demo pose native-eligible; alignment work continues.
  inhand False on GT -> the rig welded a pose native AG would never grant;
                        replay wing closes for this demo's grasp, verdict final.

Per frame closure-2..closure+30: restore, one physics step (contacts need it),
log contact / inhand / per-link AABB gap / grip qpos. No trials, no film.

Run (training paused):  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root \
    OMNIGIBSON_HEADLESS=1 python -u rt_replay_test13.py --demo 30
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=30)
    a = ap.parse_args()

    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame

    from skill_env_wrapper import _np

    man = json.load(open(f"/root/snapshot_bank_v21/manifest_d{a.demo}.json"))
    G = next(r for r in man["entries"] if str(r["stage"]) == "G")
    closure = G["grasp_closure_frame"]

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path="/root/rtt13_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]

    rows = []
    for t in range(closure - 2, closure + 31):
        restore_to_frame(wrapper, 0, t)
        og.sim.step_physics()
        try:
            cs, _ = rob._find_gripper_contacts(arm="right")
            contact = any(radio.name in c for c in cs)
        except Exception:  # noqa: BLE001
            contact = None
        try:
            ih = rob._calculate_in_hand_object(arm="right")
            inhand = None if ih is None else str(ih[0].name)
        except Exception as e:  # noqa: BLE001
            inhand = f"ERR:{type(e).__name__}"
        try:
            rc = rob._find_gripper_raycast_collisions(arm="right")
            ray_hits = sorted({c.split("/")[-1] for c in rc})[:4]
        except Exception:  # noqa: BLE001
            ray_hits = None
        lo, hi = (_np(x) for x in radio.aabb)
        gaps = [float(np.linalg.norm(np.maximum(
            np.maximum(lo - _np(l.get_position_orientation()[0]), 0.0),
            _np(l.get_position_orientation()[0]) - hi)))
            for l in rob.finger_links["right"]]
        try:
            gq = float(np.mean(_np(
                rob.get_joint_positions()[rob.gripper_control_idx["right"]])))
        except Exception:  # noqa: BLE001
            gq = float("nan")
        row = dict(rel=t - closure, contact=contact, inhand=inhand,
                   ray_hits=ray_hits, min_gap=round(min(gaps), 4),
                   grip=round(gq, 4))
        rows.append(row)
        print("GTPROBE", json.dumps(row), flush=True)
    n_inhand = sum(1 for r in rows if r["inhand"] and "radio" in str(r["inhand"]).lower())
    print("VERDICT inhand_radio_frames:", n_inhand, "/", len(rows), flush=True)
    json.dump(rows, open(f"/root/gtprobe_d{a.demo}.json", "w"), indent=1)
    print("DONE", flush=True)
    os._exit(0)


main()
