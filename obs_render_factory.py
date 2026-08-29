"""Obs-render one factory clip into converter-ready training arrays.

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

OUT = "/root/factory_obs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, required=True)
    a = ap.parse_args()

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

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path=f"/root/or_tmp_{a.demo}.hdf5",
        robot_obs_modalities=("proprio", "rgb", "depth_linear"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]

    def get_obs():
        obs = wrapper.env.get_obs()[0]
        pro, rgb, dep = None, None, None
        def walk(node, path=""):
            nonlocal pro, rgb, dep
            if isinstance(node, dict):
                for k, v in node.items():
                    kl = str(k).lower()
                    if "proprio" in kl:
                        pro = _np(v).reshape(-1)
                    elif "zed" in path.lower() or "zed" in kl:
                        if kl.endswith("rgb") or kl == "rgb":
                            rgb = _np(v)
                        elif "depth" in kl:
                            dep = _np(v)
                    if isinstance(v, dict):
                        walk(v, path + "/" + str(k))
        walk(obs)
        return pro, rgb, dep

    restore_to_frame(wrapper, 0, meta["t0"])
    try:
        rob._refresh_rigid_contact_view()
    except Exception:  # noqa: BLE001
        pass
    welded = False
    rows_p, rows_a, rows_rgb, rows_dep, rows_rp = [], [], [], [], []
    prev_cmd = None
    for i, cmd in enumerate(cmds):
        cmd = np.asarray(cmd, np.float32)
        is_new = prev_cmd is None or not np.array_equal(cmd, prev_cmd)
        if is_new:
            pro, rgb, dep = get_obs()
            if pro is not None and rgb is not None:
                rows_p.append(pro.astype(np.float32))
                rows_a.append(cmd.copy())
                rows_rgb.append(rgb[..., :3].astype(np.uint8))
                rows_dep.append(None if dep is None
                                else dep.astype(np.float16))
                rp, rq = radio.get_position_orientation()
                rows_rp.append(np.concatenate([_np(rp), _np(rq)]))
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
            print(f"OR i={i}/{len(cmds)} steps={len(rows_p)}", flush=True)
    zend = float(_np(radio.get_position_orientation()[0])[2])
    dz = abs(zend - meta["radio_z_end"])
    deps = ([d for d in rows_dep if d is not None])
    np.savez_compressed(
        f"{OUT}/d{a.demo:03d}_obs.npz",
        proprio=np.stack(rows_p), actions=np.stack(rows_a),
        head_rgb=np.stack(rows_rgb),
        gt_depth=(np.stack(deps) if len(deps) == len(rows_dep) else np.zeros(0)),
        objpose_radio=np.stack(rows_rp),
        meta=json.dumps({**meta, "determinism_delta": dz, "success": True,
                         "n_steps": len(rows_p)}))
    print(f"OBS_DONE d{a.demo}: {len(rows_p)} control steps, "
          f"rgb={rows_rgb[0].shape if rows_rgb else None}, "
          f"depth_ok={len(deps) == len(rows_dep)}, DELTA={dz:.4f}", flush=True)
    os._exit(0)


main()
