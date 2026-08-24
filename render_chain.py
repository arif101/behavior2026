"""Film the grasp->press chain, third-person camera, up to 3 takes."""
import os
import numpy as np
os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


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
        input_path="/root/rawdemos/task-0000/episode_00000030.hdf5",
        output_path="/root/render_tmp4.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    entries = [e for e in load_snapshot_bank("/root/snapshot_bank_v21") if e["demo"] == 30]
    env = SkillCommitEnvV2(wrapper, entries, l2_on=True, seed=0)
    agent = RLPD(env.obs_dim, 12, seed=0)
    import glob as _g
    _ck = max(_g.glob("/root/v2*_run/ckpt.pt"), key=__import__("os").path.getmtime)
    print("CF using ckpt:", _ck, flush=True)
    agent.load(_ck, reset_alpha=True)
    cam = og.sim.viewer_camera

    def aim(target):
        pos = np.asarray(target) + np.array([0.9, -0.9, 0.6])
        cam.set_position_orientation(
            th.as_tensor(pos, dtype=th.float32),
            th.as_tensor(lookat_quat(pos, target), dtype=th.float32))

    def grab():
        obs, _ = cam.get_obs()
        return np.asarray(obs["rgb"])[..., :3].astype(np.uint8)

    gE = [e for e in entries if e["stage"] == "G"][0]
    _press = [e for e in entries if e["family"] == "press"]
    pE = next(iter(_press), None) or {
        "demo": gE["demo"], "stage": "0", "frame": -1, "family": "press",
        "active_arm": "left", "holding_arm": gE["active_arm"],
        "lift_z": gE.get("lift_z"), "grasp_closure_frame": None,
        "snapshot": gE["snapshot"], "target_name_sub": "radio"}

    def liveness():
        res = {}
        for cond in ("l2_on", "l2_off"):
            env.l2_on = (cond == "l2_on")
            out = []
            for k in range(6):
                env.ag_assist_range = 0.20
                obs = env.reset(entry=gE)
                done, info = False, {}
                while not done:
                    act = agent.act(obs, deterministic=True)
                    obs, r, done, info = env.step(act)
                out.append((int(info["success"]), round(info["dist"], 3),
                            info.get("lift_prog", 0)))
            res[cond] = out
            print(f"LV {cond}: succ={sum(o[0] for o in out)}/6 "
                  f"dists={[o[1] for o in out]}", flush=True)
        env.l2_on = True
        return res

    for take in range(1, 4):
        env.ag_assist_range = 0.20
        obs = env.reset(entry=gE)
        frames, done, info, phase = [], False, {}, 1
        for t in range(600):
            # mild stochasticity helps reproduce training-time behavior
            act = agent.act(obs, deterministic=(take == 1))
            obs, r, done, info = env.step(act)
            if t % 2 == 0:
                aim(_np(env.target.get_position_orientation()[0]))
                frames.append(grab())
            if done:
                if phase == 1 and info["success"]:
                    phase = 2
                    env.entry = dict(pE)
                    env.active_arm = pE["active_arm"]
                    env.steps = 0; env.dwell = 0; env.dwell_bonus = 0.0
                    env.contact_made = False; env._lift_dwell = 0
                    env.target_base = env._live_target_base()
                    env.hold_act = env._stationary_act(env._proprio61())
                    env.steps = -150   # phase-2 budget 450, matching training
                    done = False
                    obs = env._obs()
                    continue
                break
        succ = bool(info.get("success") and phase == 2)
        frames += [frames[-1]] * 10
        imageio.mimwrite(f"/root/videos/chain_take{take}.mp4", frames, fps=10, quality=7)
        print(f"CF take{take}: phase={phase} success={succ} steps={env.steps} "
              f"frames={len(frames)}", flush=True)
        if succ:
            break
    os._exit(0)


main()
