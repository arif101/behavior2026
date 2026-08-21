"""Append STAGE + per-camera PIXEL labels to /root/b1k_radio_map (aux-loss round 2).

Pure math over data already on the box — no video decode, no sim:

  stage       int32   0 APPROACH (base > 1.2 m from radio)
                      1 ACQUIRE  (base near, closest EE > 12 cm from metalink)
                      2 MANIPULATE (EE <= 12 cm OR radio lifted > 3 cm)
                      3 END      (last 4% of frames — proxy for post-success)
              Geometry-derived v1 (Larchenko lineage: stage structure ~2x in BEHAVIOR-25).
  aux_pixels  f32[9]  [head_u, head_v, head_vis, wl_u, wl_v, wl_vis, wr_u, wr_v, wr_vis]
              u,v NORMALIZED [0,1] by each camera's native resolution (resolution-independent:
              the aux head bins into its own patch grid); vis = in-frustum flag.
              Wrist projections need /root/camera_intrinsics.json (extract_intrinsics.py).

Also prints the wrist-visibility fraction — an independent measurement of the wrist-blindness
diagnosis (expected LOW; head expected high).
"""

import glob
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path("/root/b1k_radio_map")
EE_L, EE_R = slice(17, 20), slice(42, 45)
HEAD_K = (238.9, 315.8, 364.7, 356.2, 720, 720)  # fx fy cx cy W H (calibrated)
CAMS = {"head": "observation.robot2cam_pose.zed_link_camera_0",
        "left": "observation.robot2cam_pose.left_realsense_link_camera_0",
        "right": "observation.robot2cam_pose.right_realsense_link_camera_0"}


def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def wrist_K():
    p = pathlib.Path("/root/camera_intrinsics.json")
    if not p.exists():
        print("WARN: no camera_intrinsics.json — wrist pixels will be vis=0")
        return {}
    intr = json.loads(p.read_text())
    out = {}
    for side in ("left", "right"):
        for name, rec in intr.items():
            if f"{side}_realsense" in name and isinstance(rec.get("K"), list):
                K = rec["K"]
                out[side] = (K[0][0], K[1][1], K[0][2], K[1][2],
                             rec["image_width"], rec["image_height"])
    print("wrist intrinsics:", out)
    return out


def project(meta_b, cam7, K):
    fx, fy, cx, cy, W, H = K
    qv = q2r(cam7[3:7]).T @ (meta_b - cam7[:3])
    xc, yc, d = qv[0], -qv[1], -qv[2]
    if d <= 0.05:
        return 0.0, 0.0, 0.0
    u = xc / d * fx + cx
    v = yc / d * fy + cy
    if 0 <= u < W and 0 <= v < H:
        return u / W, v / H, 1.0
    return 0.0, 0.0, 0.0


def main():
    emap = json.load(open("/root/episode_map.json"))
    ep2demo = {int(i): int(d) for i, d in emap["mapping"].items()}
    wk = wrist_K()

    cache = {}

    def arrays(demo):
        if demo not in cache:
            ml = np.load(f"/root/metalink_labels/ep{demo}.npz")
            pj = json.load(open(f"/root/poses_x/turning_on_radio/ep{demo}.json"))
            base = np.array([f["base_pos"] for f in pj["frames"]])
            cache[demo] = (ml["meta_base"], ml["meta_world"], base)
        return cache[demo]

    st_counts = np.zeros(4, int)
    vis_sum = np.zeros(3)
    n_tot = 0
    for fp in sorted(glob.glob(str(ROOT / "data" / "**" / "*.parquet"), recursive=True)):
        t = pq.read_table(fp)
        eps = t["episode_index"].to_numpy()
        fis = t["frame_index"].to_numpy()
        state = np.stack(t["observation.state"].to_numpy())
        cams = {k: np.stack(t[c].to_numpy()) for k, c in CAMS.items()}
        N = len(eps)
        stage = np.zeros(N, np.int32)
        pix = np.zeros((N, 9), np.float32)
        ep_len = {}
        for e in np.unique(eps):
            ep_len[int(e)] = int((eps == e).sum())
        for r in range(N):
            demo = ep2demo[int(eps[r])]
            meta_b, meta_w, base = arrays(demo)
            fi = int(fis[r])
            d_l = np.linalg.norm(meta_b[fi] - state[r, EE_L])
            d_r = np.linalg.norm(meta_b[fi] - state[r, EE_R])
            d_ee = min(d_l, d_r)
            base_dist = np.linalg.norm(base[fi][:2] - meta_w[fi][:2])
            lifted = (meta_w[fi][2] - meta_w[0][2]) > 0.03
            if fi >= 0.96 * ep_len[int(eps[r])]:
                stage[r] = 3
            elif d_ee <= 0.12 or lifted:
                stage[r] = 2
            elif base_dist <= 1.2:
                stage[r] = 1
            pix[r, 0:3] = project(meta_b[fi], cams["head"][r], HEAD_K)
            for j, side in ((3, "left"), (6, "right")):
                if side in wk:
                    pix[r, j:j + 3] = project(meta_b[fi], cams[side][r], wk[side])
        table = t
        for name, arr in (("stage", pa.array(stage, pa.int32())),
                          ("aux_pixels", pa.array(list(pix), pa.list_(pa.float32(), 9)))):
            if name in table.column_names:
                table = table.drop_columns([name])
            table = table.append_column(name, arr)
        pq.write_table(table, fp)
        st_counts += np.bincount(stage, minlength=4)
        vis_sum += pix[:, [2, 5, 8]].sum(0)
        n_tot += N
        print(f"  {fp.split('/')[-1]}: {N} rows", flush=True)

    info_p = ROOT / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    info["features"]["stage"] = {"dtype": "int32", "shape": [1], "names": None}
    info["features"]["aux_pixels"] = {"dtype": "float32", "shape": [9], "names": None}
    info_p.write_text(json.dumps(info, indent=4))

    print(f"stages: {dict(enumerate((st_counts / n_tot).round(3)))}")
    print(f"visibility fractions head/wristL/wristR: {(vis_sum / n_tot).round(3)}")
    print(f"DONE: {n_tot} rows")


if __name__ == "__main__":
    main()
