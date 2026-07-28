"""Bridge replayed poses -> the labels_*.jsonl format that scripts/b1k/add_target_points.py eats.

Why go through the official converter instead of appending columns ourselves: grafting
target_points onto the source shards leaves meta/ describing the FULL 20,000-episode challenge set
while data/ holds only our task's shards. LeRobot builds its index from that meta and dies with a
DatasetGenerationError (which then falls back to the Hub and surfaces as a misleading 401). The
official `add_target_points.py` writes a fresh, self-consistent LeRobot dataset — same path
convert_phaseA.py used for Phase A — so the meta always matches the data.

Label contract (from add_target_points.py's docstring):
    one JSON per line: {"frame": i, "M": [16 floats], "objs": {obj_name: [x, y, z]}}
    M = COLUMN-VECTOR camera->world transform, reshape(4,4), R=M[:3,:3], t=M[:3,3]

We hold base pose in world (replayed) and base->cam from the parquet's
`observation.robot2cam_pose.zed_link_camera_0` (pos + quat xyzw, FK-derived), so

    T_world_cam = T_world_base @ T_base_cam

which is exactly the M the converter wants. Object world positions come straight from the replay.

Usage:
  python poses_to_labels.py --poses_dir /root/poses/turning_on_radio \\
      --parquet_dir /root/b1k_src/data --out_dir /root/radio_labels --task turning_on_radio
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq


def quat_to_R(q) -> np.ndarray:
    """Rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def T(pos, quat) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = quat_to_R(quat)
    M[:3, 3] = np.asarray(pos, dtype=float)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--parquet_dir", required=True, help="dir of source LeRobot data shards")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cam_key", default="observation.robot2cam_pose.zed_link_camera_0")
    ap.add_argument("--first_episode_index", type=int, default=0,
                    help="episode_index of this task's first episode (radio is task 0 -> 0)")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)

    # episode_index -> robot2cam rows, read once per shard.
    cam_by_ep: dict[int, np.ndarray] = {}
    for f in sorted(glob.glob(os.path.join(a.parquet_dir, "chunk-*", "file-*.parquet"))):
        t = pq.read_table(f, columns=["episode_index", a.cam_key])
        ei = np.asarray(t["episode_index"])
        cam = np.array([np.asarray(x) for x in t[a.cam_key].to_pylist()])
        for e in np.unique(ei):
            cam_by_ep[int(e)] = cam[ei == e]
    print(f"robot2cam loaded for {len(cam_by_ep)} episodes")

    # Positional pairing: the k-th replayed episode is the k-th episode_index of this task.
    # demo_ids are SPARSE (10, 20, ... 3000 with gaps) while episode_index is dense, so any
    # arithmetic mapping is wrong — assuming demo_id = 10*(k+1) paired unrelated episodes and
    # produced frame skews from -2766 to +2544.
    pose_files = sorted(glob.glob(os.path.join(a.poses_dir, "ep*.json")),
                        key=lambda p: int(os.path.basename(p)[2:-5]))
    n_ok = 0
    skews = []
    for k, pf in enumerate(pose_files):
        d = json.load(open(pf))
        F = d["frames"]
        ep_idx = a.first_episode_index + k
        cams = cam_by_ep.get(ep_idx)
        if cams is None:
            print(f"  skip {os.path.basename(pf)}: no episode_index {ep_idx} in shards")
            continue

        n = min(len(F), len(cams))
        skews.append(len(F) - len(cams))
        out = os.path.join(a.out_dir, f"labels_{ep_idx}.jsonl")
        with open(out, "w") as fh:
            for i in range(n):
                fr = F[i]
                T_wb = T(fr["base_pos"], fr["base_quat"])
                c = cams[i]
                T_bc = T(c[:3], c[3:7])
                M = T_wb @ T_bc                      # camera -> world
                objs = {name: [float(x) for x in v["pos"]]
                        for name, v in fr["objects"].items() if v}
                fh.write(json.dumps({"frame": i, "M": [float(x) for x in M.flatten()],
                                     "objs": objs}) + "\n")
        # add_target_points.py judges success by the .done.json sidecar, not the exit code.
        json.dump({"frames": n}, open(out + ".done.json", "w"))
        n_ok += 1

    s = np.asarray(skews)
    print(f"\nwrote {n_ok} label files to {a.out_dir}")
    if s.size:
        print(f"frame skew (replay - parquet): median {np.median(s):+.0f}  "
              f"min {s.min():+d}  max {s.max():+d}")
        if np.abs(s).max() > 4:
            print("  !! skew beyond +/-4 on some episodes — investigate before converting")


if __name__ == "__main__":
    main()
