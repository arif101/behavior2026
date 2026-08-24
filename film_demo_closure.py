"""Film demo 30's grasp closure — GROUND-TRUTH pixels, no re-simulation.

Restores each RECORDED frame around grasp_closure_frame=929 and renders the
right gripper + radio close-up from two angles. Per frame, records:
  - min fingertip-link-origin -> radio-AABB-surface distance (fair "to-surface" proxy)
  - fingertip-midpoint -> radio-center distance (the SUSPECT metric from rt proto)
  - right gripper joint qpos (closure amount)
  - radio position (z reveals the frame where the rig's weld starts carrying it)
  - contact API result AFTER one physics step (contacts need a step to populate;
    the 1-step drift at recorded state is ~0.3mm, noted not hidden)

Pixels are grabbed from the PURE restored state, before that physics step.

Run (training must be paused):  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root \
    OMNIGIBSON_HEADLESS=1 python -u film_demo_closure.py --demo 30
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

OUT = "/root/demo_closure_film"


def lookat_quat(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.stack([r, u, -f], axis=1)
    w = np.sqrt(max(1e-9, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return np.array([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w),
                     (m[1, 0] - m[0, 1]) / (4 * w), w])


def pt_aabb_dist(p, lo, hi):
    d = np.maximum(np.maximum(lo - p, 0.0), p - hi)
    return float(np.linalg.norm(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=30)
    a = ap.parse_args()

    import torch as th
    import imageio
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np

    os.makedirs(OUT, exist_ok=True)
    man = json.load(open(f"/root/snapshot_bank_v21/manifest_d{a.demo}.json"))
    G = next(r for r in man["entries"] if str(r["stage"]) == "G")
    closure = G["grasp_closure_frame"]

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path="/root/film_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    cam = og.sim.viewer_camera

    def fingertip_links():
        return [(l.name, _np(l.get_position_orientation()[0]))
                for l in rob.finger_links["right"]]

    def grab(offset, target):
        pos = np.asarray(target) + np.asarray(offset)
        cam.set_position_orientation(
            th.as_tensor(pos, dtype=th.float32),
            th.as_tensor(lookat_quat(pos, target), dtype=th.float32))
        for _ in range(3):
            og.sim.render()
        obs, _ = cam.get_obs()
        return np.asarray(obs["rgb"])[..., :3].astype(np.uint8)

    # dense through the closure window, sparse on the shoulders
    frames = sorted(set(
        list(range(closure - 20, closure - 8, 4)) +
        list(range(closure - 8, closure + 13)) +
        list(range(closure + 14, closure + 31, 4))))

    rows, prev_radio, prev_ft = [], None, None
    vidA, vidB = [], []
    for t in frames:
        restore_to_frame(wrapper, 0, t)
        rp = _np(radio.get_position_orientation()[0])
        lo, hi = (_np(x) for x in radio.aabb)
        fts = fingertip_links()
        ftmid = np.mean([p for _, p in fts], axis=0)
        surf_gaps = {n: pt_aabb_dist(p, lo, hi) for n, p in fts}
        min_surf = min(surf_gaps.values())
        center_gap = float(np.linalg.norm(ftmid - rp))
        try:
            gq = _np(rob.get_joint_positions()[rob.gripper_control_idx["right"]])
            grip = float(np.mean(gq))
        except Exception:  # noqa: BLE001
            grip = float("nan")
        tracking = None
        if prev_radio is not None:
            dr = np.linalg.norm(rp - prev_radio)
            df = np.linalg.norm(ftmid - prev_ft)
            tracking = bool(dr > 0.004 and df > 0.004 and
                            np.linalg.norm((rp - prev_radio) - (ftmid - prev_ft)) < 0.006)
        prev_radio, prev_ft = rp.copy(), ftmid.copy()

        target = 0.5 * (ftmid + rp)
        imgA = grab([0.32, -0.30, 0.20], target)
        imgB = grab([-0.36, 0.22, 0.34], target)
        imageio.imwrite(f"{OUT}/f{t:04d}_A.png", imgA)
        imageio.imwrite(f"{OUT}/f{t:04d}_B.png", imgB)
        vidA.append(imgA); vidB.append(imgB)

        # contact read needs one physics step; taken AFTER pixels (pure state above)
        og.sim.step_physics()
        try:
            cs, _ = rob._find_gripper_contacts(arm="right")
            contact = any(radio.name in c for c in cs)
            ncontacts = len(cs)
        except Exception:  # noqa: BLE001
            contact, ncontacts = None, -1
        row = dict(frame=t, rel=t - closure, min_surf_gap=round(min_surf, 4),
                   center_gap=round(center_gap, 4), grip_q=round(grip, 4),
                   radio_z=round(float(rp[2]), 4), tracking=tracking,
                   contact=contact, ncontacts=ncontacts,
                   per_link={n: round(g, 4) for n, g in surf_gaps.items()})
        rows.append(row)
        print("CLOSURE", json.dumps(row), flush=True)

    json.dump(dict(demo=a.demo, closure=closure,
                   aabb_extent=[round(float(x), 4) for x in (hi - lo)],
                   rows=rows),
              open(f"{OUT}/numerics_d{a.demo}.json", "w"), indent=1)
    imageio.mimsave(f"{OUT}/closure_d{a.demo}_A.mp4", vidA, fps=4)
    imageio.mimsave(f"{OUT}/closure_d{a.demo}_B.mp4", vidB, fps=4)
    print("DONE frames:", len(rows), flush=True)
    os._exit(0)


main()
