"""Merge the FK camera poses (fk_cam_poses.py: /root/fk/cam_pose.npy [N, 21] aligned with index.npy [N, 2] =
(episode_index, frame_index)) into a LeRobot root as the `cam_pose` column (3 cams x 7, base frame) and register it.
  python add_cam_pose_column.py --root /root/b1k_radio_mix_full --npy /root/fk/cam_pose.npy --index /root/fk/index.npy
"""
import argparse, glob, json, pathlib
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--npy", required=True); ap.add_argument("--index", required=True)
a = ap.parse_args(); root = pathlib.Path(a.root)
cp = np.load(a.npy).astype(np.float32); ix = np.load(a.index)
lut = {(int(e), int(f)): i for i, (e, f) in enumerate(ix)}
files = sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)); wrote = 0; missing = 0
for f in files:
    t = pq.read_table(f); ep = t.column("episode_index").to_pylist(); fr = t.column("frame_index").to_pylist()
    rows = np.zeros((len(t), 21), np.float32)
    for i, (e, r) in enumerate(zip(ep, fr)):
        j = lut.get((int(e), int(r)))
        if j is None: missing += 1
        else: rows[i] = cp[j]
    if "cam_pose" in t.schema.names: t = t.drop(["cam_pose"])
    t = t.append_column("cam_pose", pa.array(rows.tolist(), type=pa.list_(pa.float32(), 21))); pq.write_table(t, f); wrote += len(t)
info = json.loads((root / "meta/info.json").read_text()); info["features"]["cam_pose"] = {"dtype": "float32", "shape": [21], "names": None}
(root / "meta/info.json").write_text(json.dumps(info, indent=4))
print(f"CAM_POSE_OK rows={wrote} missing={missing}")
