#!/usr/bin/env python3
"""Add per-frame 3D target points to a B1K LeRobot dataset (G3 point-conditioning arms).

The BEHAVIOR-2026 sweep produces, per task:
  * a LeRobot dataset            sweep_out/<task>/b1k/<task>/   (meta/ + data/ parquet [+ videos/])
  * per-episode privileged labels sweep_labels/<task>/labels_*.jsonl

This script joins them: for every frame it computes, per arm, the 3D displacement from the
end-effector to the task's target object and writes two new columns into a COPY of the dataset:

  * ``target_points``      float32, flat list[6]  (row-major [left_xyz, right_xyz])
  * ``target_points_mask`` bool,    list[2]       (False = that arm has no target this frame)

matching the AdaLN point-conditioning contract in DESIGN.md / ``B1KInputs``
(``np.asarray(x).reshape(2, 3)`` — flat list[6] and [2,3] are both accepted; we store flat
because parquet degrades fixed-size lists to plain lists anyway).

Label format (one line per record, TWO records per video frame):
    {"frame": i, "M": [16 floats], "objs": {obj_name: [x, y, z]}}
  * ``M``: column-vector camera->world transform (reshape(4,4); R=M[:3,:3], t=M[:3,3]).
  * video frame k of an N-frame episode maps to record idx = int((k+1)*len(rows)/N) - 1
    ("last record per frame" — same mapping as behavior2026/stage/extract_stages.py, whose
    EE-world lifting math this script reuses).

Geometry (all reused from extract_stages.py):
    T_world_base = M_cam @ inv(T_base_cam)        # T_base_cam from the per-frame
                                                  # observation.robot2cam_pose.* column (pos+quat xyzw)
    delta_arm    = R_world_base^T @ (p_obj_world - p_ee_world)
                 = (p_obj in BASE frame) - (p_ee in BASE frame)
  ``p_ee`` in the BASE frame is read directly from observation.state
  (eef_left_pos = state[17:20], eef_right_pos = state[42:45]; R1Pro 61-dim proprio layout).

WHY the BASE frame (and not the gripper frame): the base frame is rotation-stable — the
gripper's orientation churns continuously during manipulation, which would make a
gripper-frame displacement vector spin even when robot and object are static. The base frame
also matches what the policy's proprio is expressed in (eef positions in state[17:20]/[42:45]
are base-frame), so the model relates the point to its own state without learning an extra
rotation.

Target selection (v1 heuristic — the stage-label pipeline will refine this later):
  * target categories for the task come from task_targets.json["<task>"]["targets"];
    a label object belongs to a category if its name starts with the category string.
  * per frame, per arm: the nearest NOT-YET-RELOCATED target instance to that arm's EE
    (world-frame distance). With a single target instance (e.g. turning_on_radio) both arms
    therefore get the same object — by construction.
  * "relocated" (v1): an instance is retired once it has been displaced > --disp-min from its
    initial position AND is stationary AND both EEs have moved away (> --release-dist), all
    sustained for --hold-frames consecutive frames. Instances that never move (radio, fridge)
    are simply never retired.
  * mask = False for an arm when no live target instance remains (task done) or the chosen
    instance's label is missing/NaN at that frame. (Visibility is NOT modeled in v1: the
    privileged labels are world positions regardless of occlusion.)

The source dataset is never modified. The copy preserves LeRobot metadata integrity: only the
data parquet files gain columns, plus a features entry in meta/info.json for the new keys.
Norm-stats are deliberately NOT touched — per DESIGN.md the points are raw meters and must
bypass normalization.

Usage:
    python scripts/b1k/add_target_points.py \
        --dataset  .../sweep_out/picking_up_trash/b1k/picking_up_trash \
        --labels-dir .../sweep_labels/picking_up_trash \
        --task picking_up_trash \
        --task-targets .../task_targets.json \
        --out .../picking_up_trash_pts \
        --validate
"""

import argparse
import dataclasses
import glob
import json
import os
import pathlib
import re
import shutil
import sys

import numpy as np

# ---- R1Pro observation.state slices (see extract_stages.py / OmniGibson PROPRIOCEPTION_INDICES) ----
STATE_SLICES = {
    "eef_left_pos": (17, 20),
    "eef_right_pos": (42, 45),
}
ARMS = ("left", "right")  # arm order in target_points: [left, right] (matches DESIGN.md)

DEFAULT_ROBOT2CAM_KEY = "observation.robot2cam_pose.zed_link_camera_0"


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion -> 3x3 rotation matrix (same as extract_stages.py)."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = np.array([x, y, z, w]) / np.sqrt(n)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def label_record_index(frame: int, n_records: int, n_frames: int) -> int:
    """Video frame k -> label record idx (last record per frame; extract_stages.py `lidx`)."""
    return int(np.clip(int((frame + 1) * n_records / n_frames) - 1, 0, n_records - 1))


@dataclasses.dataclass
class EpisodeArrays:
    """Per-frame arrays for one episode, on the video/parquet frame index."""

    n_frames: int
    obj_world: dict[str, np.ndarray]  # name -> (N,3) world positions (NaN where missing)
    t_world_base: np.ndarray  # (N,4,4)
    ee_base: dict[str, np.ndarray]  # arm -> (N,3) BASE-frame EE positions (from proprio)
    ee_world: dict[str, np.ndarray]  # arm -> (N,3) world-frame EE positions


def load_episode_arrays(labels_path: str, state: np.ndarray, robot2cam: np.ndarray) -> EpisodeArrays:
    """Lift labels + proprio onto a common frame index. Math reused from extract_stages.load_episode."""
    n_frames = len(state)
    with open(labels_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"empty labels file: {labels_path}")
    n_records = len(rows)
    obj_names = list(rows[0]["objs"].keys())

    obj_world = {o: np.full((n_frames, 3), np.nan) for o in obj_names}
    m_cam = np.zeros((n_frames, 4, 4))
    for k in range(n_frames):
        r = rows[label_record_index(k, n_records, n_frames)]
        for o in obj_names:
            p = r["objs"].get(o)
            if p is not None:
                obj_world[o][k] = p
        m_cam[k] = np.asarray(r["M"], dtype=float).reshape(4, 4)

    # T_world_base = T_world_cam @ inv(T_base_cam)
    t_world_base = np.zeros((n_frames, 4, 4))
    for k in range(n_frames):
        t_base_cam = np.eye(4)
        t_base_cam[:3, :3] = quat_to_rot(robot2cam[k, 3:7])
        t_base_cam[:3, 3] = robot2cam[k, :3]
        t_world_base[k] = m_cam[k] @ np.linalg.inv(t_base_cam)

    ee_base, ee_world = {}, {}
    for arm in ARMS:
        a, b = STATE_SLICES[f"eef_{arm}_pos"]
        eb = state[:, a:b].astype(float)
        ew = np.einsum("nij,nj->ni", t_world_base[:, :3, :3], eb) + t_world_base[:, :3, 3]
        ee_base[arm] = eb
        ee_world[arm] = ew
    return EpisodeArrays(
        n_frames=n_frames, obj_world=obj_world, t_world_base=t_world_base, ee_base=ee_base, ee_world=ee_world
    )


def target_instances(obj_names: list[str], target_categories: list[str]) -> list[str]:
    """Label objects belonging to the task's target categories (startswith match, as in extract_stages)."""
    return [o for o in obj_names if any(o.startswith(c) for c in target_categories)]


def relocated_from_frame(
    ep: EpisodeArrays,
    obj: str,
    *,
    disp_min: float,
    release_dist: float,
    stationary_v: float,
    speed_win: int,
    hold_frames: int,
) -> int:
    """First frame from which `obj` counts as relocated (retired), or n_frames if never.

    v1 heuristic: displaced-from-initial AND stationary AND released (both EEs far), sustained
    for `hold_frames` consecutive frames -> retired from the start of that run.
    """
    p = ep.obj_world[obj]
    n = ep.n_frames
    init = p[0]
    disp = np.linalg.norm(p - init, axis=1)
    speed = np.zeros(n)
    if n > speed_win:
        speed[speed_win:] = np.linalg.norm(p[speed_win:] - p[:-speed_win], axis=1) / speed_win
    dmin = np.minimum(
        np.linalg.norm(ep.ee_world["left"] - p, axis=1),
        np.linalg.norm(ep.ee_world["right"] - p, axis=1),
    )
    with np.errstate(invalid="ignore"):
        done = (disp > disp_min) & (dmin > release_dist) & (speed < stationary_v)
    done = np.nan_to_num(done.astype(float)).astype(bool)
    run = 0
    for k in range(n):
        run = run + 1 if done[k] else 0
        if run >= hold_frames:
            return k - hold_frames + 1
    return n


def compute_target_points(
    ep: EpisodeArrays,
    targets: list[str],
    *,
    disp_min: float,
    release_dist: float,
    stationary_v: float,
    speed_win: int,
    hold_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[list[str | None]]]:
    """Per-frame per-arm displacement (BASE frame) + validity mask.

    Returns (points [N,2,3] float32, mask [N,2] bool, retired_at {obj: frame}, chosen [N][2] names).
    """
    n = ep.n_frames
    points = np.zeros((n, 2, 3), dtype=np.float32)
    mask = np.zeros((n, 2), dtype=bool)
    chosen: list[list[str | None]] = [[None, None] for _ in range(n)]

    retired_at = {
        o: relocated_from_frame(
            ep,
            o,
            disp_min=disp_min,
            release_dist=release_dist,
            stationary_v=stationary_v,
            speed_win=speed_win,
            hold_frames=hold_frames,
        )
        for o in targets
    }

    r_bw = np.transpose(ep.t_world_base[:, :3, :3], (0, 2, 1))  # R_world_base^T per frame
    t_wb = ep.t_world_base[:, :3, 3]

    for k in range(n):
        live = [o for o in targets if k < retired_at[o] and np.isfinite(ep.obj_world[o][k]).all()]
        if not live:
            continue  # task done (or labels missing) -> mask stays False
        for a, arm in enumerate(ARMS):
            dists = [np.linalg.norm(ep.obj_world[o][k] - ep.ee_world[arm][k]) for o in live]
            obj = live[int(np.argmin(dists))]
            obj_base = r_bw[k] @ (ep.obj_world[obj][k] - t_wb[k])
            delta = obj_base - ep.ee_base[arm][k]
            points[k, a] = delta.astype(np.float32)
            mask[k, a] = True
            chosen[k][a] = obj
    return points, mask, retired_at, chosen


# --------------------------------------------------------------------------- dataset IO


def find_episode_parquets(dataset_root: pathlib.Path) -> dict[int, pathlib.Path]:
    """episode_index -> parquet path.

    Real LeRobot v3 layout (our sweep output): data files are named
    data/chunk-CCC/file-FFF.parquet and meta/episodes/**/*.parquet maps
    episode_index -> (data/chunk_index, data/file_index). With the sweep's
    flush_every_n_traj=1 each file holds exactly one episode; we assert that
    (multi-episode files would need per-row slicing, unimplemented).
    Fallback for synthetic fixtures: data/**/episode_N.parquet naming.
    """
    import pyarrow.parquet as pq

    meta_files = sorted(dataset_root.glob("meta/episodes/**/*.parquet"))
    if meta_files:
        out: dict[int, pathlib.Path] = {}
        for mf in meta_files:
            t = pq.read_table(mf, columns=["episode_index", "data/chunk_index", "data/file_index"]).to_pydict()
            for ei, ci, fi in zip(t["episode_index"], t["data/chunk_index"], t["data/file_index"]):
                p = dataset_root / f"data/chunk-{int(ci):03d}/file-{int(fi):03d}.parquet"
                if not p.exists():
                    continue  # partial mirror: only episodes matched to labels must exist
                out[int(ei)] = p
        if not out:
            raise FileNotFoundError(f"meta/episodes present but empty under {dataset_root}")
        return out

    out = {}
    for p in sorted(dataset_root.glob("data/**/*.parquet")):
        m = re.search(r"episode_(\d+)", p.stem)
        if m is None:
            raise ValueError(f"cannot parse episode index from {p}")
        out[int(m.group(1))] = p
    if not out:
        raise FileNotFoundError(f"no episode parquets under {dataset_root}/data")
    return out


def match_labels_to_episodes(labels_dir: pathlib.Path, episode_indices: list[int]) -> dict[int, pathlib.Path]:
    """Match labels_*.jsonl files to episodes.

    Preferred: the trailing integer in each filename matches the set of episode indices.
    Fallback: sorted filename order == sorted episode index order (warned).
    """
    files = sorted(pathlib.Path(f) for f in glob.glob(str(labels_dir / "labels_*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no labels_*.jsonl under {labels_dir}")
    if len(files) != len(episode_indices):
        # partial mirrors carry episodes beyond this task's labels: labels drive, episodes filter
        import re as _re
        lbl_ids = set()
        for f in files:
            m = _re.search(r"(\d+)(?!.*\d)", f.stem)
            if m:
                lbl_ids.add(int(m.group(1)))
        inter = [e for e in episode_indices if e in lbl_ids]
        missing = lbl_ids - set(episode_indices)
        if missing:
            raise ValueError(f"label files reference episodes absent from dataset: {sorted(missing)[:10]}")
        print(f"[labels] intersecting: {len(files)} labels x {len(episode_indices)} episodes -> {len(inter)}")
        episode_indices = inter
    by_int = {}
    for f in files:
        m = re.search(r"(\d+)(?!.*\d)", f.stem)
        if m is None:
            by_int = None
            break
        by_int[int(m.group(1))] = f
    if by_int is not None and set(by_int) == set(episode_indices):
        return by_int
    print("[warn] label filenames do not carry episode indices; matching by sorted order", file=sys.stderr)
    return dict(zip(sorted(episode_indices), files, strict=True))


_VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".webm"}


def copy_dataset(src: pathlib.Path, dst: pathlib.Path, *, overwrite: bool) -> None:
    """Copy the dataset tree. Videos are hardlinked when possible (large, read-only); everything
    else (metadata, parquet) is deep-copied because the copy gets rewritten and must never share
    inodes with the source (the source is never modified)."""
    if dst.resolve() == src.resolve():
        raise ValueError("--out must differ from --dataset (the source is never modified)")
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists (pass --overwrite to replace)")
        shutil.rmtree(dst)

    def _copy_fn(s, d, *, follow_symlinks=True):
        if pathlib.Path(s).suffix.lower() in _VIDEO_SUFFIXES:
            try:
                os.link(s, d)
                return d
            except OSError:
                pass
        return shutil.copy2(s, d, follow_symlinks=follow_symlinks)

    shutil.copytree(src, dst, copy_function=_copy_fn)


def update_info_features(dataset_root: pathlib.Path) -> None:
    """Register the new columns in meta/info.json (LeRobot feature schema)."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        print(f"[warn] {info_path} not found; skipping feature registration", file=sys.stderr)
        return
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    features["target_points"] = {
        "dtype": "float32",
        "shape": [6],
        "names": ["lx", "ly", "lz", "rx", "ry", "rz"],
    }
    features["target_points_mask"] = {"dtype": "bool", "shape": [2], "names": ["left", "right"]}
    info["features"] = features
    info_path.write_text(json.dumps(info, indent=4))


def rewrite_parquet(path: pathlib.Path, points: np.ndarray, mask: np.ndarray,
                    ep_rows: np.ndarray | None = None) -> None:
    """Write target_points/target_points_mask for one episode's rows. Files may hold
    several episodes (LeRobot v3): ep_rows targets this episode's rows; other rows keep
    existing values (zeros/False when first initialized). Idempotent per episode."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    n = len(table)
    if ep_rows is None:
        ep_rows = np.arange(n)
    if len(ep_rows) != len(points):
        raise ValueError(f"{path}: episode rows {len(ep_rows)} != computed frames {len(points)}")
    if "target_points" in table.column_names:
        pts = [list(x) for x in table.column("target_points").to_pylist()]
        msk = [list(x) for x in table.column("target_points_mask").to_pylist()]
        table = table.drop_columns(["target_points", "target_points_mask"])
    else:
        pts = [[0.0] * 6 for _ in range(n)]
        msk = [[False, False] for _ in range(n)]
    flat = points.reshape(len(points), 6).astype(np.float32)
    for j, r in enumerate(ep_rows):
        pts[int(r)] = flat[j].tolist()
        msk[int(r)] = [bool(mask[j, 0]), bool(mask[j, 1])]
    pts_arr = pa.array(pts, type=pa.list_(pa.float32()))
    mask_arr = pa.array(msk, type=pa.list_(pa.bool_()))
    table = table.append_column("target_points", pts_arr).append_column("target_points_mask", mask_arr)
    pq.write_table(table, path)


def read_state_and_robot2cam(path: pathlib.Path, robot2cam_key: str, ep_idx: int | None = None):
    import pyarrow.parquet as pq

    cols = ["observation.state", robot2cam_key]
    names = pq.read_schema(path).names
    has_ep = "episode_index" in names
    if has_ep:
        cols.append("episode_index")
    table = pq.read_table(path, columns=cols)
    rows = None
    if has_ep and ep_idx is not None:
        epcol = np.asarray(table.column("episode_index").to_pylist())
        rows = np.flatnonzero(epcol == ep_idx)
        table = table.take(rows)
    state = np.stack([np.asarray(x, dtype=float) for x in table.column("observation.state").to_pylist()])
    r2c = np.stack([np.asarray(x, dtype=float) for x in table.column(robot2cam_key).to_pylist()])
    if state.shape[1] != 61:
        raise ValueError(f"{path}: expected 61-dim R1Pro observation.state, got {state.shape[1]}")
    if r2c.shape[1] != 7:
        raise ValueError(f"{path}: expected 7-dim robot2cam pose (pos+quat xyzw), got {r2c.shape[1]}")
    return state, r2c, rows


# --------------------------------------------------------------------------- validation


def validate_episode(
    ep_idx: int,
    ep: EpisodeArrays,
    targets: list[str],
    points: np.ndarray,
    mask: np.ndarray,
    retired_at: dict[str, int],
    chosen: list[list[str | None]],
    *,
    n_spot_checks: int = 5,
) -> None:
    n = ep.n_frames
    print(f"\n[episode {ep_idx}] frames={n}  target instances={targets or 'NONE'}")
    for o, f in retired_at.items():
        if f < n:
            print(f"  relocated: {o} retired from frame {f}")
    norms = np.linalg.norm(points, axis=2)  # (N,2)
    for a, arm in enumerate(ARMS):
        cov = 100.0 * mask[:, a].mean()
        v = norms[mask[:, a], a]
        if len(v):
            print(
                f"  {arm:>5}: mask coverage {cov:5.1f}%  |d| m: min {v.min():.3f}  med {np.median(v):.3f}  "
                f"mean {v.mean():.3f}  p95 {np.percentile(v, 95):.3f}  max {v.max():.3f}  "
                f"[in 0.1-3m: {100.0 * ((v >= 0.1) & (v <= 3.0)).mean():.1f}%]"
            )
        else:
            print(f"  {arm:>5}: mask coverage {cov:5.1f}%  (no valid frames)")
    valid_frames = np.flatnonzero(mask.any(axis=1))
    if len(valid_frames):
        for k in valid_frames[np.linspace(0, len(valid_frames) - 1, min(n_spot_checks, len(valid_frames))).astype(int)]:
            a = 0 if mask[k, 0] else 1
            arm, obj = ARMS[a], chosen[k][a]
            ow, ew, d = ep.obj_world[obj][k], ep.ee_world[arm][k], points[k, a]
            print(
                f"    spot f={k:4d} arm={arm:<5} obj={obj:<24} obj_w=[{ow[0]:7.3f} {ow[1]:7.3f} {ow[2]:7.3f}] "
                f"ee_w=[{ew[0]:7.3f} {ew[1]:7.3f} {ew[2]:7.3f}] d_base=[{d[0]:7.3f} {d[1]:7.3f} {d[2]:7.3f}] "
                f"|d|={np.linalg.norm(d):.3f}"
            )


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="source LeRobot dataset root (meta/ + data/)")
    ap.add_argument("--labels-dir", required=True, help="directory with per-episode labels_*.jsonl")
    ap.add_argument("--task", required=True, help="task name (key into task_targets.json)")
    ap.add_argument("--task-targets", required=True, help="path to task_targets.json")
    ap.add_argument("--out", required=True, help="output dataset root (copy; source never modified)")
    ap.add_argument("--robot2cam-key", default=DEFAULT_ROBOT2CAM_KEY)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--validate", action="store_true", help="print per-episode stats + spot-check rows")
    # v1 relocation-heuristic knobs (defaults follow extract_stages.py CFG)
    ap.add_argument("--disp-min", type=float, default=0.12, help="m; displacement to count as relocated")
    ap.add_argument("--release-dist", type=float, default=0.20, help="m; EE farther than this = released")
    ap.add_argument("--stationary-v", type=float, default=0.002, help="m/frame; stationary threshold")
    ap.add_argument("--speed-win", type=int, default=5, help="frames; finite-difference window")
    ap.add_argument("--hold-frames", type=int, default=15, help="frames; sustain requirement to retire")
    args = ap.parse_args()

    src = pathlib.Path(args.dataset)
    out = pathlib.Path(args.out)
    tt_all = json.loads(pathlib.Path(args.task_targets).read_text())
    task_targets = tt_all if "targets" in tt_all else tt_all[args.task]
    target_categories = task_targets.get("targets", [])
    if not target_categories:
        raise ValueError(f"task {args.task!r} has no target categories in {args.task_targets}")
    print(f"[task] {args.task}: target categories = {target_categories}")

    episodes = find_episode_parquets(src)
    labels = match_labels_to_episodes(pathlib.Path(args.labels_dir), list(episodes))
    episodes = {e: p for e, p in episodes.items() if e in labels}
    print(f"[dataset] {len(episodes)} episodes at {src}")

    copy_dataset(src, out, overwrite=args.overwrite)
    update_info_features(out)

    heur = dict(
        disp_min=args.disp_min,
        release_dist=args.release_dist,
        stationary_v=args.stationary_v,
        speed_win=args.speed_win,
        hold_frames=args.hold_frames,
    )
    all_norms, total_frames, total_valid = [], 0, 0
    for ep_idx, pq_path in sorted(episodes.items()):
        state, r2c, ep_rows = read_state_and_robot2cam(pq_path, args.robot2cam_key, ep_idx)
        ep = load_episode_arrays(str(labels[ep_idx]), state, r2c)
        targets = target_instances(list(ep.obj_world), target_categories)
        if not targets:
            print(f"[warn] episode {ep_idx}: no target instances among {list(ep.obj_world)}", file=sys.stderr)
        points, mask, retired_at, chosen = compute_target_points(ep, targets, **heur)
        rewrite_parquet(out / pq_path.relative_to(src), points, mask, ep_rows=ep_rows)
        total_frames += ep.n_frames
        total_valid += int(mask.sum())
        all_norms.append(np.linalg.norm(points, axis=2)[mask])
        if args.validate:
            validate_episode(ep_idx, ep, targets, points, mask, retired_at, chosen)

    v = np.concatenate(all_norms) if all_norms else np.zeros(0)
    print(f"\n[summary] {len(episodes)} episodes, {total_frames} frames")
    print(f"[summary] arm-frame mask coverage: {100.0 * total_valid / max(1, 2 * total_frames):.1f}%")
    if len(v):
        print(
            f"[summary] |d| over all valid arm-frames (m): min {v.min():.3f}  med {np.median(v):.3f}  "
            f"mean {v.mean():.3f}  p95 {np.percentile(v, 95):.3f}  max {v.max():.3f}"
        )
        plaus = 100.0 * ((v >= 0.1) & (v <= 3.0)).mean()
        print(f"[summary] plausibility (|d| in 0.1-3.0 m): {plaus:.1f}%")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
