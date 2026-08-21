"""Third-person films: external camera looking at the table. Grasp + press."""
import os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def lookat_quat(pos, target):
    """USD camera looks down -Z, +Y up. Return xyzw quat for pos->target."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f = f / np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.stack([r, u, -f], axis=1)   # columns: +X right, +Y up, +Z back
    w = np.sqrt(max(1e-9, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = (m[2, 1] - m[1, 2]) / (4 * w)
    y = (m[0, 2] - m[2, 0]) / (4 * w)
    z = (m[1, 0] - m[0, 1]) / (4 * w)
    return np.array([x, y, z, w])


def main():
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    import imageio
    import torch as th
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
    from rlpd_sac import RLPD
    from skill_env_wrapper_v2 import SkillCommitEnvV2, load_snapshot_bank
    from skill_env_wrapper import _np

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000020.hdf5",
        output_path="/root/render_tmp3.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    entries = [e for e in load_snapshot_bank("/root/snapshot_bank_v21") if e["demo"] == 20]
    env = SkillCommitEnvV2(wrapper, entries, l2_on=True, seed=0)
    agent = RLPD(env.obs_dim, 12, seed=0)
    agent.load("/root/v21j_run/ckpt.pt", reset_alpha=True)

    cam = og.sim.viewer_camera
    print("TP viewer_camera:", type(cam).__name__, flush=True)

    def aim(target, offset=(0.9, -0.9, 0.6)):
        pos = np.asarray(target) + np.asarray(offset)
        q = lookat_quat(pos, target)
        cam.set_position_orientation(th.as_tensor(pos, dtype=th.float32),
                                     th.as_tensor(q, dtype=th.float32))

    def grab():
        obs, _ = cam.get_obs()
        f = np.asarray(obs["rgb"])
        return f[..., :3].astype(np.uint8)

    def film(entry, tag, assist, stride=2):
        env.ag_assist_range = assist if entry["family"] == "pick_up_from" else 0.0
        obs = env.reset(entry=entry)
        aim(_np(env.target.get_position_orientation()[0]))
        frames, done, info = [], False, {}
        while not done:
            act = agent.act(obs, deterministic=True)
            obs, r, done, info = env.step(act)
            if env.steps % stride == 0:
                aim(_np(env.target.get_position_orientation()[0]))
                frames.append(grab())
        frames += [frames[-1]] * 8
        imageio.mimwrite(f"/root/videos/{tag}.mp4", frames, fps=8, quality=7)
        print(f"TP {tag}: success={info['success']} steps={info['steps']} "
              f"frames={len(frames)}", flush=True)

    gE = [e for e in entries if e["stage"] == "G"][0]
    pE = [e for e in entries if str(e["stage"]) == "0"][0]
    film(gE, "tp_grasp_d20", assist=0.30, stride=1)
    film(pE, "tp_press_d20", assist=0.0, stride=3)
    os._exit(0)


main()
