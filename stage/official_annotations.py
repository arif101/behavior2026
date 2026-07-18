"""Parse the organizers' per-episode annotations into per-frame stage labels.

Input: one annotations/task-XXXX/episode_XXXXXXXX.json from HF
`behavior-1k/2026-challenge-demos`. Schema (verified on real files):
  skill_annotation: ordered segments, each with skill_id [list], object_id
  [nested lists, occasionally str -- be lenient], manipulating_object_id,
  frame_duration [start, end) in VIDEO frames, skill_type in
  {navigation, uncoordinated, coordinated}.

Output of parse_episode(): per-VIDEO-frame arrays
  stage      [Nv] int    stage class (taxonomy index; IDLE where no segment)
  progress   [Nv] float  within-segment progress in [0,1]
  seg_id     [Nv] int    segment index (-1 = none) -- for boundary blending
  and the segment table (for arm attribution + duration stats).

Arm attribution (merge_arm_labels): the official stream is single-track but
episodes are bimanual-parallel (validated gotcha #5). The Phase-1 extractor's
per-arm tracks are the arbiter:
  - navigation segments -> both arms (whole-base motion);
  - manipulation segments -> the arm whose extractor track owns the segment's
    manipulating object over that span; if the extractor has no opinion
    (no per-arm labels yet / object not tracked), both arms inherit the stage
    and the OTHER-arm supervision is left in place -- official-only mode.
The merge rule is gated by the >=4-task overlay validation before training
(VALIDATION.md discipline).
"""

import json

import numpy as np

from taxonomy import IDLE, NAV_SKILL_IDS, STAGE_OF_SKILL


def _first(x, default=None):
    """Lenient scalar from the annotations' list-wrapped fields."""
    while isinstance(x, (list, tuple)):
        if not x:
            return default
        x = x[0]
    return x if x is not None else default


def _flat_names(x):
    out = []
    def rec(v):
        if isinstance(v, (list, tuple)):
            for e in v:
                rec(e)
        elif isinstance(v, str):
            out.append(v)
    rec(x)
    return out


def load_segments(path):
    """Segment table from one annotation file, sorted by start frame."""
    a = json.load(open(path))
    n_video = int(a["meta_data"]["task_duration"])
    segs = []
    for s in a.get("skill_annotation", []):
        sid = _first(s.get("skill_id"))
        if sid is None or int(sid) not in STAGE_OF_SKILL:
            continue
        f0, f1 = s["frame_duration"][0], s["frame_duration"][-1]
        segs.append(dict(
            skill_id=int(sid),
            stage=STAGE_OF_SKILL[int(sid)],
            start=int(f0), end=int(f1),
            objects=_flat_names(s.get("object_id")),
            manip=_flat_names(s.get("manipulating_object_id")),
            skill_type=_first(s.get("skill_type"), "uncoordinated"),
        ))
    segs.sort(key=lambda s: s["start"])
    return segs, n_video, a.get("task_name", "?")


def parse_episode(path):
    segs, n_video, task_name = load_segments(path)
    stage = np.full(n_video, IDLE, dtype=np.int64)
    progress = np.zeros(n_video, dtype=np.float32)
    seg_id = np.full(n_video, -1, dtype=np.int64)
    for k, s in enumerate(segs):
        a, b = max(0, s["start"]), min(n_video, s["end"])
        if b <= a:
            continue
        stage[a:b] = s["stage"]
        seg_id[a:b] = k
        progress[a:b] = (np.arange(a, b) - s["start"]) / max(1, s["end"] - s["start"])
    return dict(stage=stage, progress=progress, seg_id=seg_id,
                segments=segs, n_video=n_video, task_name=task_name)


def merge_arm_labels(official, extractor_perframe=None):
    """Per-arm supervision on the VIDEO-frame timebase.

    extractor_perframe: list of per-frame dicts from extract_stages.py
    (--out_perframe), or None for official-only mode.

    Returns dict with, per arm in (left, right):
      stage [Nv] int, progress [Nv] float, seg_id [Nv] int,
      phase [Nv] int (-1 where unknown -> masked from loss),
      active_obj [Nv] object-name or '' (extractor-owned).
    """
    Nv = official["n_video"]
    out = {}
    # extractor arrays, resampled to Nv if lengths differ (extractor runs on
    # the parquet timebase which equals video frames for LeRobot exports)
    ex_obj = {"left": [""] * Nv, "right": [""] * Nv}
    ex_phase = {"left": np.full(Nv, -1, np.int64), "right": np.full(Nv, -1, np.int64)}
    if extractor_perframe:
        from common import PHASE_IDX
        Ne = len(extractor_perframe)
        for i in range(Nv):
            r = extractor_perframe[min(int((i + 1) * Ne / Nv) - 1, Ne - 1)]
            arms = r.get("arms")
            if arms:                       # per-arm extractor format (Phase-1 refactor)
                for arm in ("left", "right"):
                    t = arms.get(arm) or {}
                    ex_obj[arm][i] = t.get("active_object") or ""
                    ex_phase[arm][i] = PHASE_IDX.get(t.get("phase", ""), -1)
            else:                          # pilot single-track format
                obj = r.get("active_object") or ""
                ph = PHASE_IDX.get(r.get("phase", ""), -1)
                for arm in ("left", "right"):
                    ex_obj[arm][i] = obj
                    ex_phase[arm][i] = ph

    for arm in ("left", "right"):
        stage = np.full(Nv, IDLE, dtype=np.int64)
        progress = np.zeros(Nv, dtype=np.float32)
        seg_id = np.full(Nv, -1, dtype=np.int64)
        for k, s in enumerate(official["segments"]):
            a, b = max(0, s["start"]), min(Nv, s["end"])
            if b <= a:
                continue
            if s["skill_id"] in NAV_SKILL_IDS or not s["manip"]:
                owns = True                            # whole-body / no-hand skill
            elif extractor_perframe:
                manip = s["manip"][0]
                span = ex_obj[arm][a:b]
                other = ex_obj["right" if arm == "left" else "left"][a:b]
                mine = sum(1 for o in span if o and manip.startswith(o.split("_")[0]))
                theirs = sum(1 for o in other if o and manip.startswith(o.split("_")[0]))
                owns = mine >= theirs and (mine > 0 or theirs == 0)
            else:
                owns = True                            # official-only mode
            if owns:
                stage[a:b] = s["stage"]
                seg_id[a:b] = k
                progress[a:b] = (np.arange(a, b) - s["start"]) / max(1, s["end"] - s["start"])
        out[arm] = dict(stage=stage, progress=progress, seg_id=seg_id,
                        phase=ex_phase[arm], active_obj=ex_obj[arm])
    return out
