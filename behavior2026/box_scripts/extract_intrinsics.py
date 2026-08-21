"""One-time: dump ALL robot camera intrinsics (zed head + both wrist realsense) from the env.

Same env-load pattern as offset_extract.py. Unblocks wrist-camera pixel labels (aux decode)
and later wrist-sourced L2 map writes. Reads native resolution + K per sensor, cross-checks
the zed against our video-calibrated values (fx 238.9 fy 315.8 cx 364.7 cy 356.2 @ 720).
"""

import json
import os

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

import omnigibson as og
from omnigibson.macros import gm

gm.HEADLESS = True
gm.ENABLE_TRANSITION_RULES = False

from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper  # noqa: E402

DEMO = "/root/rawdemos/task-0000/episode_00000010.hdf5"

wrapper = HDF5PlaybackWrapper.create_from_hdf5(
    input_path=DEMO, output_path="/root/tmp_intr_out.hdf5", robot_obs_modalities=("rgb",),
)
robot = wrapper.env.robots[0]
out = {}
for name, sensor in robot.sensors.items():
    rec = {"class": type(sensor).__name__}
    for attr in ("image_width", "image_height", "focal_length", "horizontal_aperture",
                 "vertical_aperture", "clipping_range"):
        try:
            v = getattr(sensor, attr)
            rec[attr] = v.tolist() if hasattr(v, "tolist") else (list(v) if isinstance(v, tuple) else v)
        except Exception as e:  # noqa: BLE001
            rec[attr] = f"ERR {str(e)[:40]}"
    try:
        K = sensor.intrinsic_matrix
        rec["K"] = K.tolist() if hasattr(K, "tolist") else K
    except Exception as e:  # noqa: BLE001
        # derive from pinhole params: fx = width * focal / horizontal_aperture
        try:
            w, h = rec["image_width"], rec["image_height"]
            f, ha = float(rec["focal_length"]), float(rec["horizontal_aperture"])
            fx = w * f / ha
            va = rec.get("vertical_aperture")
            fy = h * f / float(va) if isinstance(va, (int, float)) else fx
            rec["K"] = [[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]]
            rec["K_derived"] = True
        except Exception as e2:  # noqa: BLE001
            rec["K"] = f"ERR {str(e)[:40]} / {str(e2)[:40]}"
    out[name] = rec
    print(name, json.dumps(rec), flush=True)

with open("/root/camera_intrinsics.json", "w") as f:
    json.dump(out, f, indent=1)
print("WROTE /root/camera_intrinsics.json", flush=True)
og.shutdown()
