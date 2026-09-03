"""Obs-render v2: converter-complete capture (head+left+right rgb/depth, base_pose, objpose, rac_-compatible keys).

Replays the clip's executed command stream (deterministic; weld re-established
at the recorded step) with the robot's HEAD camera + depth enabled, capturing at
every distinct-command boundary (= control steps), obs BEFORE stepping:
  head_rgb (H,W,3 u8), gt_depth (H,W f32), proprio (61 f32), action (23 f32)
plus radio pose per step and clip meta. Output: /root/factory_obs/dNNN_obs.npz
(matched to convert_clips_to_parquet.py expectations: success=True, head_rgb
present, obs->action alignment identical to reverse_curriculum_collect).

Run (one process per clip):
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
      python -u /root/obs_render_factory.py --demo 20
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

OUT = "/root/factory_obs2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, required=True)
    ap.add_argument("--clipdir", default="/root/factory_clips")
    ap.add_argument("--suffix", default="grasp_transport")
    ap.add_argument("--tag", default="0", help="output rac_<demo>_<tag>.npz")
    a2 = ap.parse_args()

    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np

    os.makedirs(OUT, exist_ok=True)
    z = np.load(f"{a2.clipdir}/d{a2.demo:03d}_{a2.suffix}.npz",
                allow_pickle=True)
    cmds = z["cmds"]
    meta = json.loads(str(z["meta"]))

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a2.demo:08d}.hdf5",
        output_path=f"/root/or2_tmp_{a2.demo}.hdf5",
        robot_obs_modalities=("proprio", "rgb", "depth_linear"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]

    def get_obs():
        obs = wrapper.env.get_obs()[0]
        out = {"pro": None, "zed_rgb": None, "zed_dep": None,
               "l_rgb": None, "l_dep": None, "r_rgb": None, "r_dep": None}
        def walk(node, path=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    pl = (path + "/" + kl)
                    if "proprio" in kl:
                        out["pro"] = _np(v).reshape(-1)
                    elif isinstance(v, dict):
                        walk(v, path + "/" + str(k))
                    else:
                        cam = ("zed" if "zed" in pl else
                               "l" if "left_realsense" in pl or "left" in pl else
                               "r" if "right_realsense" in pl or "right" in pl
                               else None)
                        if cam:
                            key = ("zed" if cam == "zed" else cam)
                            if kl.endswith("rgb") or kl == "rgb":
                                out[f"{key}_rgb"] = _np(v)
                            elif "depth" in kl:
                                out[f"{key}_dep"] = _np(v)
        walk(obs)
        return out

    rob_base = rob
    restore_to_frame(wrapper, 0, meta["t0"])
    try:
        rob._refresh_rigid_contact_view()
    except Exception:  # noqa: BLE001
        pass
    welded = False
    R = {k: [] for k in ("p", "a", "hz", "hd", "lz", "ld", "rz", "rd",
                          "rp", "bp")}
    prev_cmd = None
    for i, cmd in enumerate(cmds):
        cmd = np.asarray(cmd, np.float32)
        is_new = prev_cmd is None or not np.array_equal(cmd, prev_cmd)
        if is_new:
            o = get_obs()
            if o["pro"] is not None and o["zed_rgb"] is not None:
                R["p"].append(o["pro"].astype(np.float32))
                R["a"].append(cmd.copy())
                R["hz"].append(o["zed_rgb"][..., :3].astype(np.uint8))
                R["hd"].append(None if o["zed_dep"] is None
                               else o["zed_dep"].astype(np.float16))
                for c, kz, kd in (("l", "lz", "ld"), ("r", "rz", "rd")):
                    R[kz].append(None if o[f"{c}_rgb"] is None
                                 else o[f"{c}_rgb"][..., :3].astype(np.uint8))
                    R[kd].append(None if o[f"{c}_dep"] is None
                                 else o[f"{c}_dep"].astype(np.float16))
                rp, rq = radio.get_position_orientation()
                R["rp"].append(np.concatenate([_np(rp), _np(rq)]))
                bpp, bqq = rob_base.get_position_orientation()
                R["bp"].append(np.concatenate([_np(bpp), _np(bqq)]))
            prev_cmd = cmd.copy()
        wrapper.env.step(cmd)
        if (not welded and i >= meta["weld_k"]
                and rob._ag_obj_constraint_params.get("right") is None):
            ft = _np(th.stack([l.get_position_orientation()[0]
                               for l in rob.finger_links["right"]]).mean(dim=0))
            rob._establish_grasp(radio, radio.root_link_name, "right",
                                 th.as_tensor(ft, dtype=th.float32),
                                 "FixedJoint")
            welded = True
        elif not welded and rob._ag_obj_constraint_params.get("right") is not None:
            welded = True
        if i % 500 == 0:
            print(f"OR i={i}/{len(cmds)} steps={len(R['p'])}", flush=True)
    zend = float(_np(radio.get_position_orientation()[0])[2])
    dz = abs(zend - meta["radio_z_end"])
    def stk(key):
        v = R[key]
        if not v or any(x is None for x in v):
            return np.zeros(0)
        return np.stack(v)
    np.savez_compressed(
        f"{OUT}/rac_{a2.demo}_{a2.tag}.npz",
        proprio=np.stack(R["p"]), actions=np.stack(R["a"]),
        head_rgb=np.stack(R["hz"]), head_depth=stk("hd"),
        left_rgb=stk("lz"), left_depth=stk("ld"),
        right_rgb=stk("rz"), right_depth=stk("rd"),
        objpose_radio_89=np.stack(R["rp"]), base_pose=np.stack(R["bp"]),
        radio_rest_z=np.float64(0.534), success=True,
        meta=json.dumps({**meta, "determinism_delta": dz,
                         "n_steps": len(R["p"])}))
    print(f"OBS2_DONE d{a2.demo}: {len(R['p'])} steps "
          f"head={R['hz'][0].shape if R['hz'] else None} "
          f"left_ok={all(x is not None for x in R['lz'])} "
          f"right_ok={all(x is not None for x in R['rz'])} "
          f"DELTA={dz:.4f}", flush=True)
    os._exit(0)


main()
