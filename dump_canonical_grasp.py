"""Dump the CANONICAL rail-grasp geometry from a certified demo (default d20):
hand pose + finger positions relative to the radio at closure+t0off, plus the
gripper's approach (tine) axis in the radio frame. Same radio model everywhere,
so this geometry is a valid grasp at ANY rest attitude — demos whose rig pull
rotated the radio (d40: rests on its back) get a consistent target.
Run (sim free, ~5 min):  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
    python -u /root/dump_canonical_grasp.py --demo 20
"""
import argparse, json, os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", type=int, default=20); a = ap.parse_args()
    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np
    from scipy.spatial.transform import Rotation as R
    man = json.load(open(f"/root/snapshot_bank_v21/manifest_d{a.demo}.json"))
    closure = next(r for r in man["entries"] if str(r["stage"]) == "G")["grasp_closure_frame"]
    t0off = 6
    try:
        fm = json.load(open(f"/root/factory_clips/d{a.demo:03d}_meta.json")); t0off = int(fm["t0"] - fm["closure"])
    except FileNotFoundError:
        pass
    w = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5", output_path=f"/root/dcg_tmp_{a.demo}.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in w.scene.objects if "radio" in o.name.lower()][0]; rob = w.scene.robots[0]
    restore_to_frame(w, 0, closure + t0off)
    pE, qE = rob.eef_links["right"].get_position_orientation(); pR, qR = radio.get_position_orientation()
    pE, pR = _np(pE), _np(pR); RE, RR = R.from_quat(_np(qE)), R.from_quat(_np(qR))
    fingers = [_np(l.get_position_orientation()[0]) for l in rob.finger_links["right"]]
    rel_f = [RR.inv().apply(f - pR) for f in fingers]
    # tine axis: from the eef origin toward the finger midpoint (the direction the tines point)
    tine_w = np.mean(fingers, axis=0) - pE; tine_w /= np.linalg.norm(tine_w)
    canon = dict(demo=a.demo, t_post=closure + t0off,
                 rel_p=RR.inv().apply(pE - pR).tolist(), rel_R_quat=(RR.inv() * RE).as_quat().tolist(),
                 rel_f=[f.tolist() for f in rel_f], rel_fm=np.mean(rel_f, axis=0).tolist(),
                 gap=float(np.linalg.norm(rel_f[0] - rel_f[1])),
                 tine_axis_radio=RR.inv().apply(tine_w).tolist(), tine_axis_world_at_post=tine_w.tolist(),
                 radio_up_world_at_post=RR.apply([0, 0, 1]).tolist())
    json.dump(canon, open(f"/root/canonical_grasp_d{a.demo}.json", "w"), indent=1)
    print("CANON", json.dumps(canon), flush=True); os._exit(0)

main()
