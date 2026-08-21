"""M1 label projector: per-frame target-object 3D position -> head-cam pixel + z-depth labels.

Replays a raw HDF5 demo through the official playback wrapper (same state stream that produced
the replayed RGB/depth videos, so labels align 1:1 with those frames) and, at each frame, reads
the target object's world position and the head camera's world pose + intrinsics from the live
scene, then projects to pixels.

Usage (behavior conda env, from OmniGibson dir):
  python label_projector.py <episode.hdf5> <target_category> <out_labels.jsonl>

Camera convention: USD/OpenGL — camera looks along -Z, +Y up in its local frame.
"""
import json
import sys

import numpy as np
import torch as th

from omnigibson.macros import gm

gm.ENABLE_TRANSITION_RULES = False

from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper

# Schema-drift shim (recordings pre-date controller_groups; harmless if fixed upstream)
from omnigibson.robots.robot import Robot as _Robot

_orig_load = _Robot._load_state


def _shimmed_load(self, state):
    if "controller_groups" not in state:
        state["controller_groups"] = {}
    return _orig_load(self, state)


_Robot._load_state = _shimmed_load


def quat_to_rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    h5_path, target_cat, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    env = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=h5_path,
        output_path="/tmp/pb_labels_scratch.hdf5",
        n_render_iterations=1,
        include_task=True,
        include_robot_control=True,
        include_contacts=False,
    )
    scene = env.scene
    robot = scene.robots[0]

    targets = [o for o in scene.objects if target_cat in getattr(o, "category", "")]
    assert targets, f"no object with category containing '{target_cat}'"
    print("TARGETS", [o.name for o in targets])

    # OG sensor wrappers are not initialized in playback mode (touching robot.sensors
    # segfaults natively) — locate the zed Camera prim on the USD stage and query USD directly.
    import omnigibson as og
    import omni.usd

    cam_prim = None
    for prim in og.sim.stage.Traverse():
        path = str(prim.GetPath())
        if prim.GetTypeName() == "Camera" and "zed" in path.lower():
            cam_prim = prim
            break
    assert cam_prim is not None, "no zed Camera prim found on stage"
    print("CAM_PRIM", str(cam_prim.GetPath()))

    # USD stage transforms are STALE during playback (states load into physics/Fabric, not USD)
    # — verified: camera USD pose frozen for 3914 frames while the radio moved 0.33m.
    # Physics-backed link pose (same API family that returned moving radio poses) + the camera's
    # static local offset gives the true per-frame camera pose.
    zed_link = None
    for lname, link in robot.links.items():
        if "zed" in lname.lower():
            zed_link = link
            break
    assert zed_link is not None, f"no zed link among {list(robot.links)[:20]}"
    print("CAM_LINK", zed_link.prim_path)

    from pxr import UsdGeom
    M_local_row = np.array(UsdGeom.Xformable(cam_prim).GetLocalTransformation(), dtype=np.float64).reshape(4, 4)
    T_local = M_local_row.T  # Gf row-vector convention -> column-vector homogeneous matrix
    print("CAM_LOCAL_OFFSET", json.dumps([round(float(x), 6) for x in T_local.flatten()]))

    # Replayed zed videos are 720x720 (verified on episode 10 output)
    W, H = 720, 720
    focal = float(cam_prim.GetAttribute("focalLength").Get())
    h_ap = float(cam_prim.GetAttribute("horizontalAperture").Get())
    fx = W * focal / h_ap
    fy, cx, cy = fx, W / 2.0, H / 2.0
    print("CAM_INTRINSICS", json.dumps({"W": W, "H": H, "focal": focal, "h_ap": h_ap, "fx": fx}))

    fh = open(out_path, "w", buffering=1)
    ct = {"i": 0}

    def cb():
        # Store the composed camera world transform (column-vector homogeneous) + object world
        # positions; projection (convention-sensitive) happens offline, iterable without replays.
        i = ct["i"]
        lp, lq = zed_link.get_position_orientation()
        lp = lp.numpy() if hasattr(lp, "numpy") else np.asarray(lp)
        lq = lq.numpy() if hasattr(lq, "numpy") else np.asarray(lq)
        T_link = np.eye(4)
        T_link[:3, :3] = quat_to_rot(lq)
        T_link[:3, 3] = lp
        T_cam = T_link @ T_local  # column-vector: p_world = T_cam @ p_cam
        rec = {"frame": i, "M": [round(float(x), 6) for x in T_cam.flatten()]}
        objs = {}
        for o in targets:
            # AABB center = visual center; frame origin sits at the object base (verified:
            # origin labels project ~40-80px low/right of the radio in overlays)
            try:
                p = o.aabb_center
            except Exception:
                p = o.get_position_orientation()[0]
            p = p.numpy() if hasattr(p, "numpy") else np.asarray(p)
            objs[o.name] = [round(float(x), 5) for x in p]
        rec["objs"] = objs
        fh.write(json.dumps(rec) + "\n")
        ct["i"] += 1

    env.playback_episode(episode_id=0, record_data=False, post_state_update_callback=cb)
    print("LABELS_DONE", ct["i"], "frames ->", out_path)


if __name__ == "__main__":
    main()
