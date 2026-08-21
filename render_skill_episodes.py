"""Render N stage-0 episodes of the trained skill with head-cam capture -> mp4.
Slow-motion (10 fps out) since real presses land in 6-30 steps; 15-frame hold on the
final frame. Frames from the zed head cam via the env obs dict.

Run: PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 python -u
     render_skill_episodes.py --ckpt /root/skill_ckpts/skill_d20_s0_smoke.pt --episodes 3
"""

import argparse
import json
import os
import subprocess

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def head_rgb(obs):
    hits = []

    def walk(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}::{k}", v)
        else:
            hits.append((prefix, node))
    walk("", obs)
    for k, v in hits:
        if "zed" in k and "rgb" in k:
            a = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
            return a[..., :3].astype(np.uint8)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--demo-id", type=int, default=20)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--out", default="/root/skill_render")
    a = ap.parse_args()

    import omnigibson as og  # noqa: F401
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
    from rlpd_sac import RLPD, load_meta
    from skill_env_wrapper import SkillCommitEnv

    bank = json.load(open("/root/skill_start_bank.json"))
    entries = [e for e in bank["entries"]
               if e["demo"] == a.demo_id and e["stage"] == a.stage]
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo_id:08d}.hdf5",
        output_path="/root/render_tmp.hdf5",
        robot_obs_modalities=("rgb", "proprio"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS,
    )
    env = SkillCommitEnv(wrapper, entries, seed=7)
    meta = load_meta()
    agent = RLPD(meta["obs_layout"]["l2_geometry"][1], 12)
    agent.load(a.ckpt)

    os.makedirs(a.out, exist_ok=True)
    fidx = 0

    def emit(n=1):
        nonlocal fidx
        fr = head_rgb(env.env.get_obs()[0])
        if fr is None:
            return
        from PIL import Image
        for _ in range(n):
            Image.fromarray(fr).save(f"{a.out}/f{fidx:05d}.png")
            fidx += 1

    for ep in range(a.episodes):
        obs = env.reset()
        emit(8)  # hold the start state so the cut is readable
        done = False
        while not done:
            act = agent.act(obs, deterministic=True)
            obs, r, done, info = env.step(act)
            emit(1)
        emit(15)  # hold the outcome
        print(f"RENDER ep{ep}: success={info['success']} steps={info['steps']} "
              f"dist={info['dist']:.3f}", flush=True)

    subprocess.run(["ffmpeg", "-y", "-framerate", "10", "-i", f"{a.out}/f%05d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
                    f"{a.out}/skill_stage0.mp4"], check=True,
                   capture_output=True)
    print("RENDER_DONE", f"{a.out}/skill_stage0.mp4", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
