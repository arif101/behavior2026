"""Flywheel export, stage 1: re-render recorded success clips with cameras.

For each {run}/clips/clip_*.npz (acts (N,23) + meta json): reconstruct the bank entry,
reset the render env from the snapshot (full wrapper reset: hold repair, latch state),
replay the recorded commands 1:1, and capture per step:
  proprio (61), head/left/right rgb + depth_linear, robot base pose, radio pose,
  robot2cam pose per camera.
Emits {run}/render_clips/rclip_*.npz in the convert_clips_to_parquet schema
(action, proprio, head_rgb..., objpose_radio_89, base_pose, start_frame, robot2cam_*).

Gating (playbook): strict-physics only — clips with assist_r > 0 are SKIPPED unless
--include-assisted (curriculum-stage data is for the skill, not the VLA).
Recovery-inclusive: the clip already contains the full arc including fumbles.

Run (pauses training! single sim process):
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
    python -u export_render_clips.py --run /root/v21x_run [--include-assisted] [--limit N]
"""
import argparse
import glob
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def find_streams(node, path=""):
    out = {}
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, dict):
                out.update(find_streams(v, f"{path}/{k}"))
            else:
                kl = str(k).lower()
                if "rgb" in kl or "depth" in kl:
                    out[f"{path}/{k}"] = v
    return out


CAMS = {"head": "zed_link", "left": "left_realsense", "right": "right_realsense"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/root/v21x_run")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-assisted", action="store_true")
    a = ap.parse_args()

    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    import torch as th  # noqa: F401
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
    from skill_env_wrapper_v2 import SkillCommitEnvV2, load_snapshot_bank
    from skill_env_wrapper import _np

    clips = sorted(glob.glob(f"{a.run}/clips/clip_*.npz"))
    if a.limit:
        clips = clips[:a.limit]
    outdir = f"{a.run}/render_clips"
    os.makedirs(outdir, exist_ok=True)

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path="/root/rawdemos/task-0000/episode_00000020.hdf5",
        output_path="/root/export_tmp.hdf5",
        robot_obs_modalities=("proprio", "rgb", "depth_linear"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS)
    bank = load_snapshot_bank("/root/snapshot_bank_v21")
    by_key = {}
    for e in bank:
        by_key.setdefault((e["demo"], str(e["stage"])), e)
    env = SkillCommitEnvV2(wrapper, bank, l2_on=True, seed=0, randomize=False)
    rob = wrapper.scene.robots[0]
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]

    def cam_prims():
        out = {}
        for name, sub in CAMS.items():
            for s in rob.sensors.values() if hasattr(rob, "sensors") else []:
                if sub in getattr(s, "prim_path", ""):
                    out[name] = s
        return out

    n_done = n_skip = 0
    for cp in clips:
        z = np.load(cp, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        if (not a.include_assisted) and (meta.get("assist_r") or 0) > 0 \
                and meta["family"] == "pick_up_from":
            n_skip += 1
            continue
        entry = dict(by_key.get((meta["demo"], meta["stage"]),
                                by_key.get((meta["demo"], "0"))))
        entry["snapshot"] = meta["snapshot"]
        acts = np.asarray(z["acts"], np.float32)
        env.reset(entry=entry)
        cams = cam_prims()
        rec = {k: [] for k in ["proprio", "base_pose", "objpose_radio_89"]}
        vids = {f"{c}_{m}": [] for c in CAMS for m in ("rgb", "depth")}
        r2c = {c: [] for c in CAMS}
        for act in acts:
            env.env.step(np.asarray(act, np.float32))
            obs = wrapper.env.get_obs()[0]
            streams = find_streams(obs)
            for c in CAMS:
                for m in ("rgb", "depth"):
                    key = next((k for k in streams
                                if CAMS[c] in k and m in k.lower()), None)
                    if key is not None:
                        arr = np.asarray(streams[key])
                        if m == "rgb":
                            arr = arr[..., :3].astype(np.uint8)
                        vids[f"{c}_{m}"].append(arr)
            rec["proprio"].append(env._proprio61().astype(np.float32))
            bp, bq = rob.get_position_orientation()
            rec["base_pose"].append(np.concatenate([_np(bp), _np(bq)]).astype(np.float32))
            rp, rq = radio.get_position_orientation()
            rec["objpose_radio_89"].append(
                np.concatenate([_np(rp), _np(rq)]).astype(np.float32))
            for c, s in cams.items():
                cpz, cq = s.get_position_orientation()
                # robot2cam: cam pose in base frame
                from skill_env_wrapper import q2r
                Rb = q2r(_np(bq))
                rel_p = Rb.T @ (_np(cpz) - _np(bp))
                r2c[c].append(np.concatenate([rel_p, _np(cq)]).astype(np.float32))
        out = {"action": acts,
               "proprio": np.stack(rec["proprio"]),
               "base_pose": np.stack(rec["base_pose"]),
               "objpose_radio_89": np.stack(rec["objpose_radio_89"]),
               "start_frame": np.int64(entry.get("frame", 0)),
               "meta": z["meta"]}
        for k, v in vids.items():
            if v:
                key = {"head_rgb": "head_rgb", "head_depth": "head_depth",
                       "left_rgb": "left_rgb", "left_depth": "left_depth",
                       "right_rgb": "right_rgb", "right_depth": "right_depth"}[k]
                out[key] = np.stack(v)
        for c, v in r2c.items():
            out[f"robot2cam_{c}"] = np.stack(v)
        op = f"{outdir}/r{os.path.basename(cp)}"
        np.savez_compressed(op, **out)
        n_done += 1
        print(f"EXPORT {os.path.basename(op)} {meta['stage']} d{meta['demo']} "
              f"{len(acts)} steps", flush=True)
    print(f"EXPORT_DONE rendered={n_done} skipped_assisted={n_skip}", flush=True)
    os._exit(0)


main()
