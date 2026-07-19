"""Stage-head shared constants and conventions.

Carried over unchanged from the validated grounding pipeline (grounding/mt_common.py):
calibrated 720p intrinsics (never recomputed), 518 backbone input (37x37 patches),
148 depth grid, every-5th-frame stride, gray16le depth decode + affine calibration,
label row for video frame k is idx = int((k+1)*len(rows)/n_video) - 1.

The stage cache EXTENDS the v0.5 grounding cache dir (CACHE/<task>/ep{f:03d}/) with:
  glob.npy          float16 [N, 768]   mean DINO patch token per cached frame
  proprio.npy       float32 [N, 61]    R1Pro observation.state at cached frames
  stage_labels.npz  per-frame per-arm supervision (see build_cache.py docstring)
so rgb (frames/f_*.jpg) and depth.npy are read from the existing grounding cache.

Timebase: cache frames are every FRAME_STRIDE-th video frame at 30 fps -> 6 Hz.
The serve/train history window is T_HIST cache frames (24 -> 4.0 s, inside the
2-5 s spec window).
"""

import os

# Calibrated ZED intrinsics at 720x720 (fitted upstream -- DO NOT recompute).
FX, FY, CX, CY = 238.9, 315.8, 364.7, 356.2
W = H = 720
IMG = 518            # DINOv2/v3 backbone input (37x37 patches @14px)
GRID = 37
DGRID = 148          # cached depth grid (= 4 * 37)
FRAME_STRIDE = 5     # cache stride over 30fps video -> 6 Hz
FPS_CACHE = 30.0 / FRAME_STRIDE
DEPTH_SCALE = 8.928079e-05
DEPTH_BIAS = -0.0119

# ---- head hyperparameters (single source of truth) ----
D_MODEL = 384
N_HEAD = 6
T_HIST = 24          # history window in cache frames (4.0 s @ 6 Hz)
L_MAX = 32           # max goal literals per task (spec: ledger <= 32-dim)
BLEND_SEC = 0.3      # soft-boundary label blending half-width (cosine, v2 3.2)

# ---- v2 loss tricks (spec 5, Larchenko/LeHome-validated) ----
HEAD_WD = 1e-3       # weight decay on output heads only (aux heads overfit)
PSAT_TAIL_SEC = 0.7  # upweight p_sat frames within this of a satisfaction flip
PSAT_TAIL_BOOST = 20.0   # (LeHome: 20x on last 20 frames @30fps ~ 0.67s)
PSAT_SMOOTH = 0.05   # label smoothing toward per-task p_sat base rates

# Shared frozen backbones (same table as grounding/model_mt.BACKBONES).
BACKBONES = {
    "dinov2_vitb14": dict(img=518, grid=37, feat=768, kind="hub"),
    "dinov3_vitb16": dict(img=512, grid=32, feat=768, kind="hf",
                          hf_id="facebook/dinov3-vitb16-pretrain-lvd1689m"),
}

# Extractor phase set (stage/extract_stages.py) -- the *within-skill* axis.
PHASES = ["idle", "approach", "grasp", "transport", "place", "release"]
N_PHASES = len(PHASES)
PHASE_IDX = {p: i for i, p in enumerate(PHASES)}

ARMS = ["left", "right"]

# R1Pro observation.state slices (61-dim; layout validated in extract_stages.py).
PROPRIO_DIM = 61

# Box-default paths, overridable via env (same convention as mt_common).
DATA = os.environ.get("B2026_DATA", "/root/data/sweep_100")
CACHE = os.environ.get("B2026_CACHE", "/root/cache_v05")
STAGE_LABELS_DIR = os.environ.get("B2026_STAGE_LABELS", "/root/stage_labels")
ANNOT_DIR = os.environ.get("B2026_ANNOT", "/root/data/annotations")
TASK_TARGETS = os.environ.get("B2026_TASK_TARGETS", "/root/task_targets.json")


def cache_dir(task, file_idx):
    return os.path.join(CACHE, task, f"ep{file_idx:03d}")


def label_row_for_frame(k, n_rows, n_video):
    """Last-record-per-frame mapping (2 physics records per rendered frame)."""
    return min(int((k + 1) * n_rows / n_video) - 1, n_rows - 1)
