"""One-time: extract the togglebutton metalink offset in radio-root frame.

Builds the env from a raw demo hdf5 (HDF5PlaybackWrapper.create_from_hdf5 — identical scene/
objects/scale to the demos we're labeling; all 200 episodes use the single model radio_89), then
reads the ToggledOn metalink pose and computes:

    p_off = R_root^T (p_meta - p_root)          q_off = q_root^-1 * q_meta

so that per-frame labels become  metalink_world(t) = radio_pose(t) . offset  — pure composition
over the 200 rescue pose JSONs, no replay.

Built-in validation: the env's frame-0 radio world pose must match pose JSON ep10 frame 0
(same episode). If root_vs_poseJSON_m is ~0, the composition base is verified end-to-end.
"""

import json
import os

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

import torch as th

import omnigibson as og
from omnigibson.macros import gm

gm.HEADLESS = True
gm.ENABLE_TRANSITION_RULES = False  # DataPlaybackWrapper asserts this

from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper  # noqa: E402

DEMO = "/root/rawdemos/task-0000/episode_00000010.hdf5"

wrapper = HDF5PlaybackWrapper.create_from_hdf5(
    input_path=DEMO,
    output_path="/root/tmp_playback_out.hdf5",
    robot_obs_modalities=(),
)
env = wrapper.env
radio = env.scene.object_registry("name", "radio_89")
assert radio is not None, f"radio_89 not found; objects: {[o.name for o in env.scene.objects][:40]}"

import omnigibson.utils.transform_utils as T  # noqa: E402
from omnigibson.object_states import ToggledOn  # noqa: E402

st = radio.states[ToggledOn]
link = st.link
p_root, q_root = radio.get_position_orientation()
p_meta, q_meta = link.get_position_orientation()
R = T.quat2mat(q_root)
p_off = R.T @ (p_meta - p_root)
q_off = T.quat_multiply(T.quat_inverse(q_root), q_meta)

out = {
    "demo": DEMO,
    "radio_scale": radio.scale.tolist(),
    "radio_aabb_extent": radio.aabb_extent.tolist(),
    "root_pos_world": p_root.tolist(),
    "root_quat_world": q_root.tolist(),
    "meta_link_name": link.name,
    "meta_prim_path": str(link.prim_path),
    "meta_pos_world": p_meta.tolist(),
    "meta_quat_world": q_meta.tolist(),
    "offset_pos_root_frame": p_off.tolist(),
    "offset_quat_root_frame": q_off.tolist(),
}

# every toggle-ish link, in case the asset has several
for name, l in radio.links.items():
    if "toggle" in name.lower():
        lp, lq = l.get_position_orientation()
        out.setdefault("all_toggle_links", {})[name] = {"pos": lp.tolist(), "quat": lq.tolist()}

try:
    out["marker_extent"] = st.visual_marker.aabb_extent.tolist()
except Exception as e:  # noqa: BLE001
    out["marker_extent_err"] = str(e)[:100]

# cross-check: same episode's rescue pose JSON, frame 0
pj = json.load(open("/root/poses_x/turning_on_radio/ep10.json"))
r0 = pj["frames"][0]["objects"]["radio_89"]
out["poseJSON_frame0"] = r0
out["root_vs_poseJSON_m"] = float(th.norm(p_root.cpu() - th.tensor(r0["pos"])))

print(json.dumps(out, indent=2), flush=True)
with open("/root/metalink_offset.json", "w") as f:
    json.dump(out, f, indent=2)
print("WROTE /root/metalink_offset.json", flush=True)
og.shutdown()
