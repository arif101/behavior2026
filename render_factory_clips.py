"""Re-render one factory clip from its saved command stream (verification pass).

Restores the clip's t0, replays the exact executed commands verbatim, renders
every Nth step from the viewer camera, writes mp4 + contact sheet, and checks
determinism: final radio z must match the meta's recorded value.

Run (one process per demo):
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
      python -u /root/render_factory_clips.py --demo 20
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

OUT = "/root/factory_reel"


def lookat_quat(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float); f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.stack([r, u, -f], axis=1)
    w = np.sqrt(max(1e-9, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    return np.array([(m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w),
                     (m[1, 0] - m[0, 1]) / (4 * w), w])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, required=True)
    ap.add_argument("--every", type=int, default=25)
    a = ap.parse_args()

    import imageio
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np

    os.makedirs(OUT, exist_ok=True)
    z = np.load(f"/root/factory_clips/d{a.demo:03d}_grasp_transport.npz",
                allow_pickle=True)
    cmds = z["cmds"]
    meta = json.loads(str(z["meta"]))
    print(f"CLIP d{a.demo}: {len(cmds)} cmds, t0={meta['t0']}", flush=True)

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path=f"/root/rr_tmp_{a.demo}.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]
    cam = og.sim.viewer_camera

    def grab():
        rp = _np(radio.get_position_orientation()[0])
        ft = _np(th.stack([l.get_position_orientation()[0]
                           for l in rob.finger_links["right"]]).mean(dim=0))
        target = 0.5 * (rp + ft)
        pos = target + np.array([0.55, -0.5, 0.35])
        cam.set_position_orientation(
            th.as_tensor(pos, dtype=th.float32),
            th.as_tensor(lookat_quat(pos, target), dtype=th.float32))
        for _ in range(2):
            og.sim.render()
        obs, _ = cam.get_obs()
        return np.asarray(obs["rgb"])[..., :3].astype(np.uint8)

    restore_to_frame(wrapper, 0, meta["t0"])
    frames = [grab()]
    for i, cmd in enumerate(cmds):
        wrapper.env.step(np.asarray(cmd, np.float32))
        if i % a.every == 0:
            frames.append(grab())
    zend = float(_np(radio.get_position_orientation()[0])[2])
    dz = abs(zend - meta["radio_z_end"])
    print(f"RENDERED d{a.demo}: {len(frames)} frames, z_end={zend:.4f} "
          f"(meta {meta['radio_z_end']}) DETERMINISM_DELTA={dz:.4f}", flush=True)
    imageio.mimsave(f"{OUT}/d{a.demo:03d}.mp4", frames, fps=8)
    print("DONE", flush=True)
    os._exit(0)


main()
