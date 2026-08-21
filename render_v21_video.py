"""Render V2.1 skill videos: grasp episode, press attempts, and the first grasp->press
composition attempt. Head-camera RGB at stride 3 -> mp4 (480p, ~10 fps)."""
import json, os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

CKPT = "/root/v21j_run/ckpt.pt"
OUT = "/root/videos"


def find_rgb_keys(node, path=""):
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}/{k}"
            if isinstance(v, dict):
                out += find_rgb_keys(v, p)
            elif "rgb" in str(k).lower():
                out.append((p, v))
    return out


def main():
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    import imageio
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
    from rlpd_sac import RLPD
    from skill_env_wrapper_v2 import SkillCommitEnvV2, load_snapshot_bank

    os.makedirs(OUT, exist_ok=True)
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000020.hdf5",
        output_path="/root/render_tmp.hdf5",
        robot_obs_modalities=("proprio", "rgb"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS)
    entries = [e for e in load_snapshot_bank("/root/snapshot_bank_v21") if e["demo"] == 20]
    env = SkillCommitEnvV2(wrapper, entries, l2_on=True, seed=0)
    agent = RLPD(env.obs_dim, 12, seed=0)
    agent.load(CKPT, reset_alpha=True)

    def grab():
        o = wrapper.env.get_obs()[0]
        keys = find_rgb_keys(o)
        # prefer a head/eyes camera; else first rgb
        keys.sort(key=lambda kv: ("zed" not in kv[0], kv[0]))  # head cam = zed_link
        arr = np.asarray(keys[0][1])
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            return arr[..., :3].astype(np.uint8)
        return None

    def run(entry, tag, assist, max_steps=300, compose_press=None):
        env.ag_assist_range = assist if entry["family"] == "pick_up_from" else 0.0
        obs = env.reset(entry=entry)
        frames, done, succ, phase = [], False, False, 1
        info = {}
        for t in range(max_steps * (2 if compose_press else 1)):
            act = agent.act(obs, deterministic=True)
            obs, r, done, info = env.step(act)
            if t % 3 == 0:
                f = grab()
                if f is not None:
                    frames.append(f)
            if done:
                if info["success"] and compose_press and phase == 1:
                    # SWITCH GOAL: keep world state, become a press episode
                    print(f"RV {tag}: grasp SUCCESS at step {env.steps}; switching to press",
                          flush=True)
                    env.entry = dict(compose_press)
                    env.active_arm = compose_press["active_arm"]
                    env.steps = 0
                    env.dwell = 0; env.dwell_bonus = 0.0; env.contact_made = False
                    env._lift_dwell = 0
                    env.target_base = env._live_target_base()
                    done, phase = False, 2
                    continue
                succ = bool(info["success"])
                break
        print(f"RV {tag}: success={succ} phase={phase} steps={env.steps} "
              f"dist={info.get('dist'):.3f} frames={len(frames)}", flush=True)
        if frames:
            h, w = frames[0].shape[:2]
            scale = 480 / w
            small = [f[::max(1, int(1/scale)), ::max(1, int(1/scale))] if scale < 1 else f
                     for f in frames]
            imageio.mimwrite(f"{OUT}/{tag}.mp4", small, fps=10, quality=7)
            print(f"RV wrote {OUT}/{tag}.mp4", flush=True)
        return succ

    gE = [e for e in entries if e["stage"] == "G"][0]
    pE = [e for e in entries if str(e["stage"]) == "0"][0]

    run(gE, "grasp_d20", assist=0.30)
    for i in range(3):
        if run(pE, f"press_d20_try{i+1}", assist=0.0):
            break
    run(gE, "compose_grasp_then_press_d20", assist=0.30, compose_press=pE)
    os._exit(0)


main()
