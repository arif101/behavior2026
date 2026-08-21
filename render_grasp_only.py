"""Render ONE grasp episode with the finger-cage assist (v21m geometry)."""
import os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

def main():
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    import imageio
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
    from rlpd_sac import RLPD
    from skill_env_wrapper_v2 import SkillCommitEnvV2, load_snapshot_bank

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000020.hdf5",
        output_path="/root/render_tmp2.hdf5",
        robot_obs_modalities=("proprio", "rgb"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS)
    entries = [e for e in load_snapshot_bank("/root/snapshot_bank_v21") if e["demo"] == 20]
    env = SkillCommitEnvV2(wrapper, entries, l2_on=True, seed=0)
    agent = RLPD(env.obs_dim, 12, seed=0)
    agent.load("/root/v21j_run/ckpt.pt", reset_alpha=True)

    def find_rgb(node, path=""):
        out = []
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, dict):
                    out += find_rgb(v, f"{path}/{k}")
                elif "rgb" in str(k).lower():
                    out.append((f"{path}/{k}", v))
        return out

    def grab():
        keys = find_rgb(wrapper.env.get_obs()[0])
        keys.sort(key=lambda kv: ("right_realsense" not in kv[0], kv[0]))  # grasping-arm wrist cam
        a = np.asarray(keys[0][1])
        return a[..., :3].astype(np.uint8) if a.ndim == 3 else None

    gE = [e for e in entries if e["stage"] == "G"][0]
    env.ag_assist_range = 0.30
    obs = env.reset(entry=gE)
    frames, done, info = [], False, {}
    while not done:
        act = agent.act(obs, deterministic=True)
        obs, r, done, info = env.step(act)
        f = grab()
        if f is not None:
            frames.append(f)
    print(f"RG success={info['success']} steps={info['steps']} "
          f"lift={info.get('lift_prog')} frames={len(frames)}", flush=True)
    if frames:
        frames = frames + [frames[-1]] * 10
        imageio.mimwrite("/root/videos/grasp_wrist_d20.mp4", frames, fps=5, quality=7)
        print("RG wrote /root/videos/grasp_fingercage_d20.mp4", flush=True)
    os._exit(0)

main()
