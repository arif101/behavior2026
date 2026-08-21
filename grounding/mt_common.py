"""v0.5 multi-task grounding: shared constants, global category vocabulary,
object-name -> category mapping, demo->file matching, and splits.

Conventions carried over unchanged from the validated single-task pilot
(grounding/dataset.py): all GT/eval in the ORIGINAL 720p pixel frame with the
calibrated intrinsics (never recomputed); depth mp4 = HEVC gray12le decoded
gray16le via ffmpeg; z_m = DEPTH_SCALE*raw16 + DEPTH_BIAS; label row for video
frame k is idx = int((k+1)*len(rows)/n_video) - 1 (last record per frame).
"""

import glob
import json
import os
import re

# Calibrated ZED intrinsics at 720x720 (fitted upstream -- DO NOT recompute).
FX, FY, CX, CY = 238.9, 315.8, 364.7, 356.2
W = H = 720
IMG = 518            # DINOv2 backbone input (37x37 patches @14px)
DGRID = 148          # cached depth grid (= 4 * 37)
HM = 180             # output heatmap resolution
FRAME_STRIDE = 5
DEPTH_SCALE = 8.928079e-05
DEPTH_BIAS = -0.0119

DATA = "/root/data/sweep_100"
CACHE = "/root/cache_v05"
TASK_TARGETS = "/root/task_targets.json"

# Occlusion rule v2 (2026-07-15 mid-run correction; tunable at train time
# because the cache stores raw margins = z_proj - depth(u,v) and per-instance
# world displacement). An in-frame instance is VISIBLE iff margin < thr_j:
#   STATIC instance (world displacement < MOVE_DISP; furniture/containers):
#       thr = clip(max(0.15, q75_j + 0.10), None, 0.80)
#     Containers are BIMODAL (fridge: closed door => margin ~ +0.4 because the
#     label is the CENTER; open door => margin ~ -0.2 via the interior). Both
#     states are visibly present -- q75 tolerates the closed state; the 0.80
#     cap still gates through-wall/other-room cases (margins >= 1m).
#   MOVING instance (task targets):
#       thr = clip(max(0.15, q10_j + 0.10), None, 0.50)
#     q10_j = its own surface offset when visible (0 for small opaque objects,
#     NEGATIVE for transparent ones, +0.2-0.3 for big boxes) -- so a box stays
#     supervised while visible, but the fridge/cabinet/trunk-INTERIOR frames
#     (margin jump >> q10) are gated out: no hallucination positives.
# Fixed 0.15 for everything (coordinator's first proposal) was verified to
# false-negative closed cabinets/fridges (visible furniture would have become
# 'never visible' negative supervision); the pure-q10 adaptive rule wrongly
# gated the CLOSED state of articulated containers. Verified mechanically on
# clearing_food_from_table_into_fridge + storing_food before restart.
# Invalid depth at the pixel (z<=0.05m) -> visibility falls back to in-frame.
OCC_MIN = 0.15        # absolute floor of the keep-threshold
OCC_DIFF = 0.10       # differential above the instance's own offset quantile
OCC_CAP_STATIC = 0.80
OCC_CAP_MOVING = 0.50
# Measured separation: true targets displace >= 1.8m, static furniture <= 0.05,
# ARTICULATED furniture (fridge/cabinet doors shift the label center) 0.25-0.33
# -> 0.5 classifies articulated furniture as static (the intent).
MOVE_DISP = 0.50      # meters of world displacement => "moving"

# 3-4 entire tasks held out from training (category/scene transfer split).
# Chosen for scene diversity + max category overlap with the 24 training tasks:
#   kitchen           : putting_dishes_away_after_cleaning (4/4 cats seen in train)
#   outdoor-ish       : loading_the_car (driveway/car; 3/6 cats seen)
#   articulated-heavy : putting_away_Halloween_decorations (cabinets; 5/7 seen)
#   storage/living    : boxing_books_up_for_storage (2/2 cats seen)
HELDOUT_TASKS = [
    "putting_dishes_away_after_cleaning",
    "loading_the_car",
    "putting_away_Halloween_decorations",
    "boxing_books_up_for_storage",
]


def all_tasks():
    return sorted(os.listdir(os.path.join(DATA, "sweep_labels")))


def train_tasks():
    return [t for t in all_tasks() if t not in HELDOUT_TASKS]


# ---------------------------------------------------------------- vocabulary
_MODEL_SUFFIX = re.compile(r"^[a-z]{6}$")


def base_vocab():
    """The benchmark vocabulary: union of targets+references over ALL 100 tasks
    in task_targets.json (515 categories), sorted. Stable across sweeps."""
    tt = json.load(open(TASK_TARGETS))
    cats = set()
    for v in tt.values():
        cats.update(v["targets"])
        cats.update(v["references"])
    return sorted(cats)


def strip_instance(name):
    """Object instance name -> category name: strip trailing _N indices and a
    6-char lowercase model code (OmniGibson naming: cat[_model6][_N])."""
    toks = name.split("_")
    while toks and toks[-1].isdigit():
        toks = toks[:-1]
    if len(toks) > 1 and _MODEL_SUFFIX.match(toks[-1]):
        return "_".join(toks), "_".join(toks[:-1])
    return "_".join(toks), None


def map_category(name, vocab_set):
    """Map an instance name to a category. Longest token-prefix match against
    the vocab (handles bottom_cabinet vs bottom_cabinet_no_top); falls back to
    stripping the model suffix. Returns category-name or None."""
    toks = name.split("_")
    while toks and toks[-1].isdigit():
        toks = toks[:-1]
    for j in range(len(toks), 0, -1):
        cand = "_".join(toks[:j])
        if cand in vocab_set:
            return cand
    full, nosuffix = strip_instance(name)
    return nosuffix if (nosuffix and nosuffix in vocab_set) else None


def build_vocab():
    """base vocab + EXTRA categories observed in sweep labels but absent from
    task_targets (e.g. scene objects like standing_mirror). Extras are trained
    but flagged; the benchmark vocab (first 515 ids) is stable."""
    base = base_vocab()
    vs = set(base)
    extras = set()
    for t in all_tasks():
        dones = glob.glob(os.path.join(DATA, "sweep_labels", t, "*.done.json"))
        for d in dones:
            for o in json.load(open(d))["label_objs"]:
                if map_category(o, vs) is None:
                    full, nosuffix = strip_instance(o)
                    extras.add(nosuffix or full)
    vocab = base + sorted(extras)
    return vocab, {c: i for i, c in enumerate(vocab)}, len(base)


# ------------------------------------------------------- demo->file matching
def episode_table(task):
    """Match labels_<demo>.jsonl to videos/file-XXX.mp4 via episode length:
    done.json num_samples == parquet length + 1 (verified on-box). Returns
    list of dicts sorted by file index. Raises on ambiguity."""
    lab_dir = os.path.join(DATA, "sweep_labels", task)
    meta_dir = os.path.join(DATA, "sweep_out", task, "b1k", task,
                            "meta", "episodes", "chunk-000")
    import pandas as pd
    lengths = {}
    for p in sorted(glob.glob(os.path.join(meta_dir, "file-*.parquet"))):
        fi = int(os.path.basename(p).split("-")[1].split(".")[0])
        lengths[fi] = int(pd.read_parquet(p, columns=["length"])["length"].iloc[0])
    if not lengths:  # meta missing on HF for some tasks -> ffprobe (equivalent:
        # parquet length == video nb_frames, verified on-box)
        import subprocess
        for fi in range(5):
            p = video_path(task, "rgb", fi)
            if not os.path.exists(p):
                continue
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames", "-of", "csv=p=0", p],
                capture_output=True, text=True, check=True).stdout.strip()
            lengths[fi] = int(out)
    demos = {}
    for d in sorted(glob.glob(os.path.join(lab_dir, "*.done.json"))):
        j = json.load(open(d))
        demos[j["demo_id"]] = j["num_samples"]
    assert len(lengths) == len(demos) == 5, (task, lengths, demos)
    # exact assignment demo -> file with num_samples == length+1
    table = []
    used = set()
    for demo, ns in sorted(demos.items()):
        cand = [fi for fi, L in lengths.items()
                if (ns == L + 1 or ns == L) and fi not in used]
        assert len(cand) >= 1, (task, demo, ns, lengths)
        if len(cand) > 1:
            raise AssertionError(f"ambiguous demo->file in {task}: {demo} {cand}")
        used.add(cand[0])
        table.append({"demo": demo, "file": cand[0], "n_video": lengths[cand[0]]})
    table.sort(key=lambda r: r["file"])
    return table


def heldout_ep_file(table):
    """Held-out episode per task = the file carrying the HIGHEST demo id
    (for turning_on_radio -> demo 50, the same episode the pilot held out)."""
    return max(table, key=lambda r: r["demo"])["file"]


# ------------------------------------------------------------- projection
def project(M_flat, p_world):
    import numpy as np
    M = np.asarray(M_flat, dtype=np.float64).reshape(4, 4)
    R, t = M[:3, :3], M[:3, 3]
    pc = R.T @ (np.asarray(p_world) - t)
    z = -pc[2]
    if z <= 1e-6:
        return -1.0, -1.0, float(z), False
    u = CX + FX * pc[0] / z
    v = CY - FY * pc[1] / z
    return float(u), float(v), float(z), bool(0 <= u < W and 0 <= v < H)


def video_path(task, kind, file_idx):
    key = {"rgb": "observation.rgb.zed_link_camera_0",
           "depth": "observation.depth_linear.zed_link_camera_0"}[kind]
    return os.path.join(DATA, "sweep_out", task, "b1k", task, "videos",
                        key, "chunk-000", f"file-{file_idx:03d}.mp4")


def labels_path(task, demo):
    return os.path.join(DATA, "sweep_labels", task, f"labels_{demo}.jsonl")


def instance_thresholds(margin, inframe, disp):
    """Per-instance visibility thresholds: [N,K] margins, [K] displacement ->
    [K] thresholds implementing the documented static/moving rule."""
    import numpy as np
    K = margin.shape[1]
    thr = np.full(K, OCC_MIN, dtype=np.float32)
    for j in range(K):
        s = inframe[:, j] & np.isfinite(margin[:, j])
        if s.sum() < 5:
            continue
        if disp[j] < MOVE_DISP:      # static: furniture / containers
            q75 = np.percentile(margin[s, j], 75)
            thr[j] = min(max(OCC_MIN, q75 + OCC_DIFF), OCC_CAP_STATIC)
        else:                        # moving: task targets
            q10 = np.percentile(margin[s, j], 10)
            thr[j] = min(max(OCC_MIN, q10 + OCC_DIFF), OCC_CAP_MOVING)
    return thr


def visible_mask(inframe, margin, thr):
    """The documented visibility rule, vectorized (numpy arrays).
    inframe [N,K] bool; margin [N,K] float (NaN = unknown depth); thr [K]."""
    import numpy as np
    unknown = ~np.isfinite(margin)
    return inframe & (unknown | (margin < thr[None, :]))
