"""Build the stage-head training cache for a set of episodes.

Works over any LeRobot-style episode root (the box sweep_out, or a local
download of HF behavior-1k/2026-challenge-demos -- see runpod/prep_data.py),
plus the organizers' annotation JSONs. Extends/creates the per-episode cache
dir (common.cache_dir) with:

  frames/f_%05d.jpg  518x518 RGB every FRAME_STRIDE-th frame (skipped if the
                     grounding cache already built them)
  depth.npy          f16 [N,148,148] m (skipped if present)
  glob.npy           f16 [N,768] mean DINO patch token per cached frame (GPU)
  proprio.npy        f32 [N,61] observation.state at cached frames
  literals.json      encoded goal literals (taxonomy.encode_literals)
  stage_labels.npz   per-arm supervision on the cache timebase:
      stage_{arm} i64 [N]      taxonomy stage class (IDLE where none)
      seg_{arm} i64 [N]        official segment id (-1 none; boundary blending)
      progress_{arm} f32 [N]   within-segment progress
      phase_{arm} i64 [N]      extractor phase (-1 = masked; needs Phase-1 labels)
      active_lit_{arm} i64 [N] literal index (-1 = masked)
      ledger f32 [N,L]         per-literal P(satisfied) target
      ledger_valid u8 scalar   1 = trained, 0 = masked
  .stage_done marker

Ledger v1 proxy (official-only mode): a literal flips satisfied at the END of
a completed place/insert/close-family segment whose manipulating object
matches the literal's target category. Replaced by Phase-1's real BDDL
predicate evaluation when those labels land; ledger_valid distinguishes them.

v4: three label upgrades, each independently optional so partial data trains:
  * literals come from B2026_TASK_LITERALS (goal_literals.py: real predicates
    from the task's BDDL goal) when present, else the task_targets guess;
  * the ledger comes from B2026_PREDICATES/<task>/ep{fi:03d}.json
    (eval_predicates.py: per-frame sim-state predicate eval) when present,
    else the v1 proxy; per-literal validity lands in ledger_lit_valid [L]
    (old caches without it default to all-valid);
  * stage_labels_posbins.npz: the kill-ablation label set (spec eval 4) --
    stage/seg/progress relabeled as N_STAGES uniform temporal-position bins,
    ledger/literal fields copied -- so the identical head trains on
    Larchenko-style bins with --label_file stage_labels_posbins.npz.
"""

import argparse
import json
import os
import subprocess

import numpy as np

from common import (DEPTH_BIAS, DEPTH_SCALE, DGRID, FRAME_STRIDE, H, IMG,
                    L_MAX, TASK_TARGETS, W, cache_dir)
from official_annotations import merge_arm_labels, parse_episode
from taxonomy import (N_STAGES, SKILL_NAMES, SKILL_OF_STAGE, encode_literals,
                      literals_from_task_targets)

TASK_LITERALS = os.environ.get("B2026_TASK_LITERALS", "/root/task_literals.json")
PREDICATES_DIR = os.environ.get("B2026_PREDICATES", "/root/predicates")

# stages whose completion flips a literal (place/insert/close/toggle family)
COMPLETING_SKILLS = {3, 4, 6, 8, 11, 12, 14, 19, 61, 69, 70, 88, 90, 91, 92, 98}


def _select_expr(v0, v1):
    """LeRobot v3 packs episodes into shared video files: select frames
    [v0, v1) of the file, then every FRAME_STRIDE-th of those."""
    return (f"select=between(n\\,{v0}\\,{v1 - 1})*not(mod(n-{v0}\\,{FRAME_STRIDE}))"
            if v1 else f"select=not(mod(n\\,{FRAME_STRIDE}))")


def decode_rgb(video, out_dir, n_expected, v0=0, v1=None):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-threads", "4", "-i", video,
         "-vf", f"{_select_expr(v0, v1)},scale={IMG}:{IMG}:flags=area",
         "-vsync", "0", "-start_number", "0", "-q:v", "2",
         os.path.join(out_dir, "f_%05d.jpg")], check=True)
    n = len([f for f in os.listdir(out_dir) if f.endswith(".jpg")])
    assert n == n_expected, f"{video}: {n} jpgs != {n_expected}"


def decode_depth(video, n_expected, v0=0, v1=None):
    """Streamed gray16le decode -> f16 [N,148,148] meters (mt pattern)."""
    import torch
    import torch.nn.functional as Fnn
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-threads", "4", "-i", video,
         "-vf", _select_expr(v0, v1), "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fb = H * W * 2
    out = np.zeros((n_expected, DGRID, DGRID), dtype=np.float16)
    i0 = 0
    while True:
        buf = proc.stdout.read(fb * 256)
        if not buf:
            break
        n = len(buf) // fb
        raw = np.frombuffer(buf[:n * fb], dtype=np.uint16).reshape(n, H, W)
        z = raw.astype(np.float32) * DEPTH_SCALE + DEPTH_BIAS
        t = torch.from_numpy(z).unsqueeze(1)
        small = Fnn.adaptive_avg_pool2d(t, DGRID).squeeze(1).numpy()
        out[i0:i0 + n] = small.astype(np.float16)
        i0 += n
    proc.wait()
    assert i0 == n_expected, f"{video}: {i0} depth frames != {n_expected}"
    return out


def build_glob(frames_dir, n, device, backbone, batch=64):
    """Mean DINO patch token per cached frame -> f16 [N,feat].
    backbone: (tokens_fn, spec) from train.load_backbone, loaded ONCE."""
    import torch
    import torch.nn.functional as Fnn
    from PIL import Image
    tokens_fn, spec = backbone
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = np.zeros((n, spec["feat"]), dtype=np.float16)
    with torch.no_grad():
        for a in range(0, n, batch):
            b = min(n, a + batch)
            imgs = np.stack([
                np.asarray(Image.open(os.path.join(frames_dir, f"f_{i:05d}.jpg"))
                           .convert("RGB"), dtype=np.uint8) for i in range(a, b)])
            x = torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2).float() / 255
            if x.shape[-1] != spec["img"]:
                x = Fnn.interpolate(x, size=(spec["img"], spec["img"]),
                                    mode="bilinear", align_corners=False,
                                    antialias=True)
            x = (x - mean) / std
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16,
                                enabled=device != "cpu"):
                f = tokens_fn(x)
            out[a:b] = f.mean(1).float().cpu().numpy().astype(np.float16)
    return out


def match_literal(obj_names, literals):
    """First literal whose target category prefixes an object name, else -1."""
    for j, lit in enumerate(literals[:L_MAX]):
        t = (lit.get("target") or "").split(".")[0]
        if t and any(o.startswith(t) for o in obj_names):
            return j
    return -1


def _fit(a, n):
    """Edge-extend a per-frame array to n rows (annotation task_duration can
    run a few frames short of the LeRobot row count for the same episode)."""
    if len(a) >= n:
        return a
    if isinstance(a, list):
        return a + [a[-1]] * (n - len(a))
    return np.concatenate([a, np.repeat(a[-1:], n - len(a), axis=0)])


def real_ledger(pred_eval, ks, L):
    """eval_predicates.py output -> (ledger [len(ks), L], lit_valid [L]).
    sat rows are sampled every eval_every_frames video frames; cache frame k
    takes the last row at or before it. -1 (not evaluable this frame) forward-
    fills; leading unknowns are 0."""
    frames = np.asarray(pred_eval["frames"])
    sat = np.asarray(pred_eval["sat"], dtype=np.int8)      # [T, L], -1/0/1
    filled = sat.astype(np.float32)
    for j in range(sat.shape[1]):
        last = 0.0
        for t in range(sat.shape[0]):
            if sat[t, j] < 0:
                filled[t, j] = last
            else:
                last = filled[t, j]
    rows = np.clip(np.searchsorted(frames, ks, side="right") - 1, 0, len(frames) - 1)
    led = np.zeros((len(ks), L), dtype=np.float32)
    led[:, :sat.shape[1]] = filled[rows]
    lit_valid = np.zeros(L, dtype=np.uint8)
    lit_valid[:sat.shape[1]] = np.asarray(pred_eval["evaluable"], dtype=np.uint8)
    return led, lit_valid


def posbin_labels(lab, ks, n_video):
    """Kill-ablation label set (spec eval 4): identical head, Larchenko-style
    temporal-position bins instead of semantic skills. Same class count
    (N_STAGES) so capacity matches; both arms share the episode-time bin;
    phase/active-literal masked; ledger fields copied (comparable ledger F1)."""
    bins = np.minimum((ks.astype(np.float64) / max(1, n_video) * N_STAGES)
                      .astype(np.int64), N_STAGES - 1)
    within = (ks.astype(np.float64) / max(1, n_video) * N_STAGES) - bins
    out = {}
    for arm in ("left", "right"):
        out[f"stage_{arm}"] = bins
        out[f"seg_{arm}"] = bins                     # bin id = segment id
        out[f"progress_{arm}"] = within.astype(np.float32)
        out[f"phase_{arm}"] = np.full(len(ks), -1, dtype=np.int64)
        out[f"active_lit_{arm}"] = np.full(len(ks), -1, dtype=np.int64)
    for k in ("ledger", "ledger_valid", "ledger_lit_valid", "segments_json"):
        out[k] = lab[k]
    return out


def build_labels(annot_path, literals, n_video, ks, extractor_perframe=None,
                 pred_eval=None):
    off = parse_episode(annot_path)
    arms = merge_arm_labels(off, extractor_perframe)
    need = int(ks.max()) + 1

    L = L_MAX
    lab = {}
    for arm in ("left", "right"):
        A = arms[arm]
        for k in ("stage", "seg_id", "progress", "phase", "active_obj"):
            A[k] = _fit(A[k], need)
        lab[f"stage_{arm}"] = A["stage"][ks]
        lab[f"seg_{arm}"] = A["seg_id"][ks]
        lab[f"progress_{arm}"] = A["progress"][ks].astype(np.float32)
        lab[f"phase_{arm}"] = A["phase"][ks]
        al = np.full(len(ks), -1, dtype=np.int64)
        for i, k in enumerate(ks):
            s = A["seg_id"][k]
            if s >= 0:
                al[i] = match_literal(off["segments"][s]["manip"] or
                                      off["segments"][s]["objects"], literals)
        lab[f"active_lit_{arm}"] = al

    if pred_eval is not None:
        # v4: real per-frame predicate evaluation over recorded sim state
        lab["ledger"], lab["ledger_lit_valid"] = real_ledger(pred_eval, ks, L)
        lab["ledger_valid"] = np.uint8(1 if lab["ledger_lit_valid"].any() else 0)
    else:
        # ledger v1 proxy from completed completing-segments
        led = np.zeros((len(ks), L), dtype=np.float32)
        flips = []                                 # (video_frame, literal_idx)
        for s in off["segments"]:
            if s["skill_id"] in COMPLETING_SKILLS:
                j = match_literal(s["manip"] or s["objects"], literals)
                if j >= 0:
                    flips.append((s["end"], j))
        for f_end, j in flips:
            for i, k in enumerate(ks):
                if k >= f_end:
                    led[i, j] = 1.0
        lab["ledger"] = led
        lab["ledger_valid"] = np.uint8(1 if flips else 0)
        lab["ledger_lit_valid"] = np.ones(L, dtype=np.uint8)
    lab["segments_json"] = np.frombuffer(
        json.dumps([{k: v for k, v in s.items()} for s in off["segments"]])
        .encode(), dtype=np.uint8)
    return lab


def build_episode(task, file_idx, rgb_video, depth_video, parquet, annot_path,
                  task_targets, device="cuda", extractor_perframe_path=None,
                  vrange=None, prange=None, backbone=None, drange=None):
    """vrange/prange: [from, to) frame/row spans of this episode inside shared
    LeRobot-v3 chunk files (None = the file is a single episode)."""
    d = cache_dir(task, file_idx)
    if os.path.exists(os.path.join(d, ".stage_done")):
        return f"{task}/ep{file_idx:03d}: cached"
    os.makedirs(d, exist_ok=True)

    import pandas as pd
    import pyarrow.parquet as pq
    cols = ["observation.state"]
    if "episode_index" in pq.read_schema(parquet).names:
        df = pd.read_parquet(parquet, columns=cols + ["episode_index"])
        df = df[df["episode_index"] == file_idx]
        assert len(df), f"{parquet}: episode {file_idx} not in file"
        st = np.stack(df["observation.state"].values).astype(np.float32)
    else:
        df = pd.read_parquet(parquet, columns=cols)
        st = np.stack(df["observation.state"].values).astype(np.float32)
        if prange:
            st = st[prange[0]:prange[1]]
    n_video = len(st)
    v0, v1 = (vrange if vrange else (0, None))
    ks = list(range(0, n_video, FRAME_STRIDE))
    N = len(ks)

    fdir = os.path.join(d, "frames")
    if not os.path.exists(os.path.join(fdir, f"f_{N-1:05d}.jpg")):
        decode_rgb(rgb_video, fdir, N, v0, v1)
    dep_path = os.path.join(d, "depth.npy")
    if not os.path.exists(dep_path):
        d0, d1 = (drange if drange else (v0, v1))
        np.save(dep_path, decode_depth(depth_video, N, d0, d1))

    np.save(os.path.join(d, "proprio.npy"), st[ks])
    glob_path = os.path.join(d, "glob.npy")
    if not os.path.exists(glob_path):
        if backbone is None:
            from train import load_backbone
            backbone = load_backbone("dinov2_vitb14", device)
        np.save(glob_path, build_glob(fdir, N, device, backbone))

    # v4 literal/ledger sources, best available first (see docstring)
    pred_eval = None
    pe_path = os.path.join(PREDICATES_DIR, task, f"ep{file_idx:03d}.json")
    if os.path.exists(pe_path + ".done"):
        pred_eval = json.load(open(pe_path))
    if pred_eval is not None:
        literals = pred_eval["literals"]
        lit_src = "predicates"
    elif os.path.exists(TASK_LITERALS):
        tl = json.load(open(TASK_LITERALS))
        literals = (tl.get(task) or {}).get("literals")
        lit_src = "bddl"
        if not literals:
            literals = literals_from_task_targets(task_targets)
            lit_src = "guess"
    else:
        literals = literals_from_task_targets(task_targets)
        lit_src = "guess"
    pred, tgt, ref, mask = encode_literals(literals, L_MAX)
    json.dump(dict(pred=pred, tgt=tgt, ref=ref, mask=mask, literals=literals,
                   source=lit_src),
              open(os.path.join(d, "literals.json"), "w"))

    exf = None
    if extractor_perframe_path and os.path.exists(extractor_perframe_path):
        exf = [json.loads(l) for l in open(extractor_perframe_path)]
    lab = build_labels(annot_path, literals, n_video, np.array(ks), exf, pred_eval)
    np.savez(os.path.join(d, "stage_labels.npz"), **lab)
    np.savez(os.path.join(d, "stage_labels_posbins.npz"),
             **posbin_labels(lab, np.array(ks), n_video))
    open(os.path.join(d, ".stage_done"), "w").close()
    return (f"{task}/ep{file_idx:03d}: built N={N} lits={lit_src} "
            f"ledger={'real' if pred_eval else 'proxy'} "
            f"arms={'extractor' if exf else 'official-only'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="JSON list of episodes: [{task, file_idx, rgb, depth, "
                         "parquet, annot, extractor_perframe?}] "
                         "(runpod/prep_data.py writes this)")
    ap.add_argument("--task_targets", default=TASK_TARGETS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="dinov2_vitb14")
    args = ap.parse_args()

    from train import load_backbone
    backbone = load_backbone(args.backbone, args.device)
    tt_all = json.load(open(args.task_targets))
    eps = json.load(open(args.manifest))
    for e in eps:
        tt = tt_all.get(e["task"], {})
        msg = build_episode(e["task"], e["file_idx"], e["rgb"], e["depth"],
                            e["parquet"], e["annot"], tt, args.device,
                            e.get("extractor_perframe"),
                            e.get("vrange"), e.get("prange"), backbone,
                            e.get("drange"))
        print(msg, flush=True)


if __name__ == "__main__":
    main()
