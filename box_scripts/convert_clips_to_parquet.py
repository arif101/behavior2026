"""Convert reverse-curriculum splice clips (rc_clips/*.npz) into a LeRobot-v3 training dataset.

Produces a NEW dataset directory (default /root/b1k_radio_corrective) with the SAME layout and
schema as /root/b1k_radio_map (meta/ + data/ + videos/) so it can be merged or mixed at training
time. Only clips with success=True AND captured head_rgb become episodes (obs_t -> action_t,
already aligned by the collector: reverse_curriculum_collect.py captures obs BEFORE stepping).

Field provenance (exact / approximated / zeros -- see also the printed conversion report):
  action, observation.state   VERBATIM from clip npz (actions (N,23), proprio (N,61) f32; the
                              61-d eval layout matches the original observation.state -- both
                              come from the same r1pro.yaml proprio key list).
  target_points               EXACT REPLICATION of box_scripts/add_map_labels.py:
                              tp[:3] = meta_base - state[17:20]; tp[3:] = meta_base - state[42:45]
                              meta_world = radio_pos + R(radio_quat) @ P_OFF   (build_metalink_labels.py)
                              meta_base  = W2B(meta_world), where W2B (world->base, base static
                              during the splice) is Kabsch-fit from the banked per-demo label
                              pairs /root/metalink_labels/ep{demo}.npz (meta_world, meta_base)
                              over the demo frames >= start_frame. Fit residual + conditioning
                              printed; EE-proximity physical check printed (spot_check_metalink
                              convention: closest-arm min over last 40% expected ~0.05-0.15 m).
                              Clips WITHOUT objpose_radio_89 (early collector version) fall back
                              to the DEMO-TIMELINE meta_base[start_frame+t] (clamped) -- radio
                              pose diverges from the demo after the policy lifts it: APPROX+WARN.
  target_points_mask          [True, True] every frame (add_map_labels.py convention).
  stage                       1 (pick) until radio z rises > 0.03 m above the clip's first frame,
                              then 2 (press) to the last frame. z source: clip objpose_radio_89
                              (own-world, exact) or demo-timeline metalink z (fallback, WARN).
                              Value set matches add_stage_pixel_labels.py (1=ACQUIRE 2=MANIPULATE;
                              0/3 never apply to splices that start near the grasp).
  robot2cam_pose.{3 cams}     APPROXIMATION: per-camera MEDIAN over the reference rows with
                              stage >= 2 (posture-matched to the splice regime). Cameras are
                              link-rigid, not base-rigid: the zed rides the flexing torso and the
                              wrist cams ride the arms, so per-frame truth is NOT derivable from
                              clip data (no FK model here). Spread of the reference distribution
                              is printed so the error scale is on record.
  aux_pixels                  head (u,v,vis) projected from meta_base via the median zed pose and
                              the calibrated HEAD_K from add_stage_pixel_labels.py -- APPROX.
                              wrist entries = (0,0,0) vis=0: arm-mounted cam poses are pose-
                              dependent; a median would fabricate plausible-looking wrong labels.
                              (0,0,0) is the original invalid convention -- project() in
                              add_stage_pixel_labels.py returns exactly that out-of-frustum, and
                              the aux head masks on the vis flag.
  map_tokens_full/blind       ZEROS + TODO: builder is mapper/offline_driver.py + FoveatedMap
                              (decodes the original videos, needs pose JSONs + map stack -- not
                              runnable from clip data on CPU here). Zeros are SAFE: the policy
                              embeds all-zero tokens as the learned null token
                              (fork_snapshot/b1k_policy.py lines 64-65 / 124-126), and arm-A
                              (all-zero tokens) is the serving configuration that won.
  next.reward/terminated      0 / False everywhere except each episode's final row: 1.0 / True.
  next.truncated              False everywhere.
  timestamp                   frame_index / fps (f32).
  videos                      RGB: clip JPEG 1080 -> LANCZOS resize to native res (zed 720,
                              realsense 480) -> hevc yuv420p crf30 g8 x265 bframes=0 (specs from
                              the reference info.json). Depth: clip uint16-mm PNG -> NEAREST
                              resize -> lerobot.datasets.depth_utils.quantize_depth (12-bit log,
                              min 0.01 max 10.0 shift 3.5) -> hevc gray12le. One file per video
                              key, episodes concatenated (from/to timestamps in episodes meta).
                              If the box libx265 rejects gray12le, depth videos are SKIPPED with
                              a TODO (features stay registered in info.json -- flagged loudly).

Usage (box):
  /root/miniconda3/envs/openpi/bin/python convert_clips_to_parquet.py \
      --clips "/root/rc_clips/rc_60_0.npz" --out /root/b1k_radio_corrective_test --overwrite
  /root/miniconda3/envs/openpi/bin/python convert_clips_to_parquet.py \
      --clips "/root/rc_clips/*.npz" --out /root/b1k_radio_corrective
"""

import argparse
import glob
import io
import json
import os
import pathlib
import re
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

# ---------------------------------------------------------------------------------------------
# Banked constants
# ---------------------------------------------------------------------------------------------
# Togglebutton metalink offset in radio-root frame. Source of record:
# behavior2026/labels/metalink_offset.json (one-time sim read by box_scripts/offset_extract.py;
# link radio_89:meta__base_link_togglebutton_0_0_link). Falls back to /root/metalink_offset.json
# if present so a box-side re-extract wins automatically.
P_OFF = np.array([0.04468477889895439, 0.04208241030573845, -0.012461928650736809])

# Calibrated zed head intrinsics @ 720x720 -- fx fy cx cy W H, verbatim from
# box_scripts/add_stage_pixel_labels.py (HEAD_K).
HEAD_K = (238.9, 315.8, 364.7, 356.2, 720, 720)

EE_L, EE_R = slice(17, 20), slice(42, 45)
FPS = 30

CAM_COLS = {
    "head": "observation.robot2cam_pose.zed_link_camera_0",
    "left": "observation.robot2cam_pose.left_realsense_link_camera_0",
    "right": "observation.robot2cam_pose.right_realsense_link_camera_0",
}
# clip npz stream -> (video_key, target square resolution, is_depth)
VIDEO_STREAMS = [
    ("head_rgb", "observation.rgb.zed_link_camera_0", 720, False),
    ("left_rgb", "observation.rgb.left_realsense_link_camera_0", 480, False),
    ("right_rgb", "observation.rgb.right_realsense_link_camera_0", 480, False),
    ("head_depth", "observation.depth_linear.zed_link_camera_0", 720, True),
    ("left_depth", "observation.depth_linear.left_realsense_link_camera_0", 480, True),
    ("right_depth", "observation.depth_linear.right_realsense_link_camera_0", 480, True),
]

WARNINGS = []


def warn(msg):
    WARNINGS.append(msg)
    print(f"WARNING: {msg}", flush=True)


def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def project(meta_b, cam7, K):
    """Verbatim port of add_stage_pixel_labels.py project(): normalized (u, v, vis)."""
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


def fit_world_to_base(meta_w, meta_b):
    """Kabsch fit of the rigid world->base map from banked per-demo label pairs.

    Valid iff the base is static over the fitted window (true post-approach; the splice regime).
    Returns R, t, max residual (m), singular values of the centered world set (conditioning:
    sv[1] ~ 0 means near-collinear metalink motion -> rotation ill-pinned)."""
    cw, cb = meta_w.mean(0), meta_b.mean(0)
    A, B = meta_w - cw, meta_b - cb
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ cw
    resid = np.linalg.norm((meta_w @ R.T + t) - meta_b, axis=1)
    sv = np.linalg.svd(A, compute_uv=False)
    return R, t, float(resid.max()), sv


# ---------------------------------------------------------------------------------------------
# Per-clip label derivation
# ---------------------------------------------------------------------------------------------
def derive_labels(clip, demo, n, p_off, labels_dir):
    """Returns (meta_base (n,3), radio_z (n,), notes list)."""
    notes = []
    lab_p = pathlib.Path(labels_dir) / f"ep{demo}.npz"
    if not lab_p.exists():
        raise FileNotFoundError(f"{lab_p} missing -- metalink labels are required")
    lab = np.load(lab_p)
    mw_demo, mb_demo = lab["meta_world"].astype(np.float64), lab["meta_base"].astype(np.float64)
    start = int(clip["start_frame"])

    if "objpose_radio_89" in clip.files:
        # W2B fit is only needed on this path. The base can drift mid-demo (demo 60: 16 cm
        # residual from start_frame), so retry the fit window sliding right and keep the best.
        T = len(mw_demo)
        best = None
        for s0 in range(min(start, T - 30), min(start + 301, T - 30) + 1, 50):
            R, t, resid, sv = fit_world_to_base(mw_demo[s0:], mb_demo[s0:])
            if best is None or resid < best[2]:
                best = (R, t, resid, sv, s0)
            if resid < 0.01:
                break
        R, t, resid, sv, s0 = best
        notes.append(f"W2B Kabsch over demo frames [{s0}:{T}]: max resid {resid * 1000:.1f} mm, "
                     f"centered-world SVs {np.round(sv, 4)}")
        if resid > 0.01:
            warn(f"demo {demo}: W2B fit residual {resid * 1000:.1f} mm > 10 mm even after "
                 "window sliding -- base moves during manipulation; target_points frame suspect")
        if sv[1] < 1e-3:
            warn(f"demo {demo}: metalink motion near-collinear (sv={sv}) -- W2B rotation "
                 "ill-pinned")
        op = np.asarray(clip["objpose_radio_89"], np.float64)  # (n,7) pos+quat, clip's own world
        mw_clip = np.stack([op[i, :3] + q2r(op[i, 3:7]) @ p_off for i in range(n)])
        meta_base = mw_clip @ R.T + t
        d0 = np.linalg.norm(mw_clip[0] - mw_demo[min(start, len(mw_demo) - 1)])
        notes.append(f"objpose_radio_89 PRESENT (exact path); restore check "
                     f"|meta_clip(0)-meta_demo(start)| = {d0 * 1000:.1f} mm")
        if d0 > 0.02:
            warn(f"demo {demo}: clip frame-0 metalink is {d0 * 1000:.0f} mm from the demo's "
                 f"start_frame metalink -- restore/offset mismatch?")
        radio_z = op[:, 2]
    else:
        idx = np.minimum(start + np.arange(n), len(mb_demo) - 1)
        meta_base = mb_demo[idx]
        radio_z = mw_demo[idx, 2]  # metalink z as radio-z proxy (rigid attachment)
        moved = float(np.linalg.norm(mw_demo[idx] - mw_demo[idx[0]], axis=1).max())
        clamped = int((start + np.arange(n) >= len(mb_demo)).sum())
        warn(f"demo {demo}: TODO clip has NO objpose_radio_89 (early collector) -- using "
             f"DEMO-TIMELINE metalink labels; radio pose diverges from the demo once the policy "
             f"lifts it (demo-label motion in window: {moved * 100:.1f} cm; {clamped} frames "
             "clamped past demo end). Recollect with RC_TRACK_OBJECTS for exact labels.")
        notes.append("objpose ABSENT -> demo-timeline fallback (APPROXIMATE)")
    return meta_base.astype(np.float64), np.asarray(radio_z, np.float64), notes


# ---------------------------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------------------------
def encode_stream(out_path, frames_per_ep, size, is_depth):
    """Encode a list (per episode) of lists of PNG/JPEG bytes into one mp4 (episodes
    concatenated, continuous pts). Matches the reference specs from info.json."""
    import av

    out_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(out_path), "w")
    from fractions import Fraction
    if is_depth:
        from lerobot.datasets.depth_utils import quantize_depth
        stream = container.add_stream("hevc", FPS)  # depth: no g/crf (reference has null)
        stream.pix_fmt = "gray12le"
    else:
        stream = container.add_stream(
            "hevc", FPS, options={"g": "8", "crf": "30", "x265-params": "log-level=0:bframes=0"})
        stream.pix_fmt = "yuv420p"
    stream.width = stream.height = size
    stream.time_base = Fraction(1, FPS)

    count = 0
    for ep_frames in frames_per_ep:
        for buf in ep_frames:
            img = Image.open(io.BytesIO(buf))
            if is_depth:
                arr = np.asarray(img)
                if arr.dtype != np.uint16:
                    arr = arr.astype(np.uint16)  # PIL 'I' mode decodes 16-bit PNG as int32
                if arr.shape[0] != size:
                    arr = np.asarray(
                        Image.fromarray(arr).resize((size, size), Image.NEAREST))
                frame = quantize_depth(arr, input_unit="mm", video_backend="pyav")
            else:
                if img.size != (size, size):
                    img = img.resize((size, size), Image.LANCZOS)
                frame = av.VideoFrame.from_image(img)
            frame.pts = count
            frame.time_base = Fraction(1, FPS)
            for pkt in stream.encode(frame):
                container.mux(pkt)
            count += 1
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()
    return count


# ---------------------------------------------------------------------------------------------
def median_cam_poses(ref_root):
    """Per-camera median robot2cam over reference rows with stage >= 2 (posture-matched)."""
    fp = sorted(glob.glob(str(ref_root / "data" / "**" / "*.parquet"), recursive=True))[0]
    t = pq.read_table(fp, columns=list(CAM_COLS.values()) + ["stage"])
    stage = t["stage"].to_numpy()
    sel = stage >= 2
    out = {}
    for short, col in CAM_COLS.items():
        arr = np.stack(t[col].to_numpy())[sel]
        med, spread = np.median(arr, 0), arr.std(0)
        out[short] = med.astype(np.float32)
        print(f"  robot2cam[{short}] median (stage>=2, n={sel.sum()}): {np.round(med, 3)}\n"
              f"    per-dim std: {np.round(spread, 3)}  (error scale of the static-pose approx)")
    return out


def load_ref_episode_row(ref_root, demo):
    """Original episodes-meta row for this raw demo id (task_instance_id etc.)."""
    fp = sorted(glob.glob(str(ref_root / "meta" / "episodes" / "**" / "*.parquet"),
                          recursive=True))[0]
    t = pq.read_table(fp, columns=["raw_episode_id", "task_instance_id"])
    raws = t["raw_episode_id"].to_numpy()
    hit = np.where(raws == demo)[0]
    return int(t["task_instance_id"][int(hit[0])].as_py()) if len(hit) else -1


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="/root/rc_clips/*.npz", help="glob of clip npz files")
    ap.add_argument("--ref", default="/root/b1k_radio_map", help="reference dataset (read-only)")
    ap.add_argument("--out", default="/root/b1k_radio_corrective")
    ap.add_argument("--labels", default="/root/metalink_labels")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-videos", action="store_true")
    a = ap.parse_args()
    ref, out = pathlib.Path(a.ref), pathlib.Path(a.out)
    assert ref.resolve() != out.resolve(), "refusing to write into the reference dataset"
    if out.exists():
        if not a.overwrite:
            sys.exit(f"{out} exists (pass --overwrite)")
        shutil.rmtree(out)

    p_off = P_OFF
    if os.path.exists("/root/metalink_offset.json"):
        p_off = np.array(json.load(open("/root/metalink_offset.json"))["offset_pos_root_frame"])
        print("using /root/metalink_offset.json offset", np.round(p_off, 5))
    else:
        print("using repo-banked metalink offset (behavior2026/labels/metalink_offset.json):",
              np.round(p_off, 5))

    ref_info = json.loads((ref / "meta" / "info.json").read_text())
    ref_schema = pq.ParquetFile(sorted(glob.glob(str(ref / "data" / "**" / "*.parquet"),
                                                 recursive=True))[0]).schema_arrow
    print("computing reference median robot2cam poses...")
    cam_med = median_cam_poses(ref)

    # ---- select clips ----------------------------------------------------------------------
    clips = []
    for fp in sorted(glob.glob(a.clips)):
        d = np.load(fp, allow_pickle=True)
        ok = bool(d["success"]) and "head_rgb" in d.files and len(d["head_rgb"]) > 0
        print(f"{os.path.basename(fp)}: success={bool(d['success'])} "
              f"head_rgb={'head_rgb' in d.files} -> {'EPISODE' if ok else 'skip'}")
        if ok:
            clips.append((fp, d))
    if not clips:
        sys.exit("no eligible clips (success=True with head_rgb)")

    # ---- per-episode assembly --------------------------------------------------------------
    rows = {name: [] for name in ref_schema.names}
    ep_meta = []
    vid_frames = {key: [] for _, key, _, _ in VIDEO_STREAMS}
    global_idx = 0
    for ep_i, (fp, d) in enumerate(clips):
        demo = int(re.match(r"rc_(\d+)_\d+", os.path.basename(fp)).group(1))
        act = np.asarray(d["actions"], np.float32)
        prop = np.asarray(d["proprio"], np.float32)
        n = len(act)
        assert prop.shape == (n, 61), f"proprio shape {prop.shape} != ({n},61)"
        assert act.shape == (n, 23), f"actions shape {act.shape} != ({n},23)"
        for k in ("head_rgb", "left_rgb", "right_rgb"):
            assert len(d[k]) == n, f"{k} has {len(d[k])} frames, expected {n}"

        meta_base, radio_z, notes = derive_labels(d, demo, n, p_off, a.labels)
        for msg in notes:
            print(f"  ep{ep_i} (demo {demo}): {msg}")

        # stage: 1 until radio z rises > 0.03 above the clip's first frame, then 2 to the end
        lifted = np.where(radio_z - radio_z[0] > 0.03)[0]
        t_lift = int(lifted[0]) if len(lifted) else n  # never lifted -> all stage 1
        stage = np.ones(n, np.int32)
        stage[t_lift:] = 2
        if not len(lifted):
            warn(f"ep{ep_i} (demo {demo}): radio z never rose >0.03 -- whole clip stage=1")

        tp = np.zeros((n, 6), np.float32)
        tp[:, :3] = meta_base - prop[:, EE_L]
        tp[:, 3:] = meta_base - prop[:, EE_R]
        # physical validation (spot_check_metalink convention)
        dmin = np.minimum(np.linalg.norm(tp[:, :3], axis=1), np.linalg.norm(tp[:, 3:], axis=1))
        tail = dmin[int(0.6 * n):]
        print(f"  ep{ep_i} (demo {demo}): closest-arm |EE-metalink| min over last 40% = "
              f"{tail.min():.3f} m (expect ~0.05-0.15); stage split 1:{t_lift} 2:{n - t_lift}")
        if tail.min() > 0.30:
            warn(f"ep{ep_i} (demo {demo}): EE never approached the metalink "
                 f"({tail.min():.2f} m) -- label frame suspect")

        aux = np.zeros((n, 9), np.float32)  # wrists stay (0,0,0)=invalid by design
        for r in range(n):
            aux[r, 0:3] = project(meta_base[r], cam_med["head"].astype(np.float64), HEAD_K)

        rows["action"] += list(act)
        rows["observation.state"] += list(prop)
        for short, col in CAM_COLS.items():
            rows[col] += [cam_med[short]] * n
        rows["next.reward"] += [0.0] * (n - 1) + [1.0]
        rows["next.terminated"] += [False] * (n - 1) + [True]
        rows["next.truncated"] += [False] * n
        rows["timestamp"] += (np.arange(n) / FPS).astype(np.float32).tolist()
        rows["frame_index"] += list(range(n))
        rows["episode_index"] += [ep_i] * n
        rows["index"] += list(range(global_idx, global_idx + n))
        rows["task_index"] += [0] * n
        rows["target_points"] += list(tp)
        rows["target_points_mask"] += [np.array([True, True])] * n
        rows["map_tokens_full"] += [np.zeros(576, np.float32)] * n
        rows["map_tokens_blind"] += [np.zeros(576, np.float32)] * n
        rows["stage"] += stage.tolist()
        rows["aux_pixels"] += list(aux)

        for npz_key, vkey, _, is_dep in VIDEO_STREAMS:
            vid_frames[vkey].append(list(d[npz_key]) if npz_key in d.files else [])
        ep_meta.append({"ep": ep_i, "demo": demo, "n": n, "start": global_idx})
        global_idx += n

    warn("TODO map_tokens_full/blind are ZEROS (FoveatedMap offline_driver not runnable from "
         "clip data) -- zeros embed as the model's learned null token (b1k_policy.py), the "
         "arm-A serving convention")
    warn("robot2cam_pose columns + aux_pixels head use the stage>=2 MEDIAN reference pose "
         "(cameras are link-mounted, per-frame truth not derivable from clips); wrist "
         "aux_pixels left (0,0,0)=invalid")

    # ---- data parquet ----------------------------------------------------------------------
    arrays = []
    for field in ref_schema:
        col = rows[field.name]
        if pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type):
            npt = np.bool_ if pa.types.is_boolean(field.type.value_type) else np.float32
            arrays.append(pa.array([np.asarray(v, npt) for v in col], type=field.type))
        else:
            arrays.append(pa.array(col, type=field.type))
    table = pa.Table.from_arrays(arrays, schema=pa.schema(list(ref_schema)))
    dpath = out / "data" / "chunk-000" / "file-000.parquet"
    dpath.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dpath)
    print(f"wrote {dpath} ({table.num_rows} rows, {len(clips)} episodes)")

    # ---- videos ----------------------------------------------------------------------------
    depth_ok = True
    if a.skip_videos:
        warn("--skip-videos: no mp4s written")
    else:
        for npz_key, vkey, size, is_dep in VIDEO_STREAMS:
            vpath = out / "videos" / vkey / "chunk-000" / "file-000.mp4"
            if not any(vid_frames[vkey]):
                warn(f"TODO no {npz_key} frames in any clip -- {vkey} video skipped")
                continue
            try:
                nf = encode_stream(vpath, vid_frames[vkey], size, is_dep)
                print(f"wrote {vpath} ({nf} frames @ {size}x{size} "
                      f"{'gray12le' if is_dep else 'yuv420p'})")
            except Exception as e:  # noqa: BLE001
                if is_dep:
                    depth_ok = False
                    warn(f"TODO depth encode FAILED for {vkey} ({e}) -- depth videos skipped; "
                         "features remain registered in info.json (loader will not find files)")
                else:
                    raise

    # ---- meta ------------------------------------------------------------------------------
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    info = json.loads(json.dumps(ref_info))  # deep copy; feature blocks verbatim
    info["total_episodes"] = len(clips)
    info["total_frames"] = global_idx
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{len(clips)}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    import pandas as pd
    task = "turning_on_radio"
    tdf = pd.DataFrame({"task_index": np.array([0], np.int64)},
                       index=pd.Index([task], name="task"))
    tdf.to_parquet(out / "meta" / "tasks.parquet")

    erows = {}
    erows["episode_index"] = np.array([m["ep"] for m in ep_meta], np.int64)
    erows["tasks"] = [[task] for _ in ep_meta]
    erows["length"] = np.array([m["n"] for m in ep_meta], np.int64)
    erows["data/chunk_index"] = np.zeros(len(ep_meta), np.int64)
    erows["data/file_index"] = np.zeros(len(ep_meta), np.int64)
    erows["dataset_from_index"] = np.array([m["start"] for m in ep_meta], np.int64)
    erows["dataset_to_index"] = np.array([m["start"] + m["n"] for m in ep_meta], np.int64)
    bounds = np.cumsum([0] + [m["n"] for m in ep_meta]) / FPS
    for _, vkey, _, _ in VIDEO_STREAMS:
        erows[f"videos/{vkey}/chunk_index"] = np.zeros(len(ep_meta), np.int64)
        erows[f"videos/{vkey}/file_index"] = np.zeros(len(ep_meta), np.int64)
        erows[f"videos/{vkey}/from_timestamp"] = bounds[:-1].astype(np.float64)
        erows[f"videos/{vkey}/to_timestamp"] = bounds[1:].astype(np.float64)
    erows["meta/episodes/chunk_index"] = np.zeros(len(ep_meta), np.int64)
    erows["meta/episodes/file_index"] = np.zeros(len(ep_meta), np.int64)
    erows["task_index"] = np.zeros(len(ep_meta), np.int64)
    erows["demo_index_within_task"] = np.arange(len(ep_meta), dtype=np.int64)
    erows["raw_episode_id"] = np.array([m["demo"] for m in ep_meta], np.int64)
    erows["task_instance_id"] = np.array(
        [load_ref_episode_row(ref, m["demo"]) for m in ep_meta], np.int64)
    erows["annotation_path"] = [f"annotations/task-0000/episode_{m['demo']:08d}.json"
                                for m in ep_meta]
    # column order must match the reference episodes meta
    ref_ep_schema = pq.ParquetFile(sorted(glob.glob(
        str(ref / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))[0]).schema_arrow
    edf = pd.DataFrame({name: erows[name] for name in ref_ep_schema.names})
    edf.to_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)
    print(f"wrote meta/ (info.json, tasks.parquet, episodes) -> {out}")

    print("\n===== CONVERSION REPORT =====")
    print(f"episodes: {len(clips)}  frames: {global_idx}  out: {out}")
    for m in ep_meta:
        print(f"  ep{m['ep']}: demo {m['demo']}, {m['n']} frames")
    print(f"depth videos: {'OK' if depth_ok and not a.skip_videos else 'SKIPPED/FAILED'}")
    print(f"{len(WARNINGS)} warnings/TODOs:")
    for w in WARNINGS:
        print(f"  - {w}")


if __name__ == "__main__":
    main()
