#!/usr/bin/env python3
"""Contact-window tagging for oversampling (DATA_ASSEMBLY_SPEC.md §4, AUGUST §3).

Contact/grasp frames are a sliver of each demo; uniform sampling drowns them in transit
frames, so BC under-learns the commit (a root cause of the 0-grasp gate result). This tags
the contact frame-ranges per episode and assigns a per-frame sample_weight, so a weighted
sampler shows the model the grasp/press moments far more often.

THREE contact signals (union) per DATA_ASSEMBLY_SPEC §4:
  1. ANNOTATION (this tool, no GPU): frames inside contact-category skill segments
     (grasp/press/place/open/close/pour/...), from skill_annotation frame_duration + verb.
  2. GRIPPER transition (box TODO — needs action arrays): frames where the gripper crosses
     open<->closed (+/- a window).
  3. PROXIMITY (box TODO — needs state arrays): EE within a threshold of the manip object.
The annotation signal is the strongest and is buildable now; (2)/(3) refine it on a box.

Output: per-episode contact frame-ranges + recommended weight, consumable by the assembler to
write a `sample_weight` column. Also reports the contact frame FRACTION (sets the oversample
ratio: to make contact ~p_target of sampled frames, weight ≈ p_target*(1-f)/((1-p_target)*f)).

Usage:
  python tag_contact_windows.py --glob "/path/real_ann/*.json" --weight 4.0 --pad 8 \
      --out contact_windows.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re

CONTACT_VERBS = {  # verbs whose segments are commit/contact (mirror build_reward_map categories)
    "pick up", "pick up from", "grasp", "put down", "place on", "place in", "place next",
    "place next to", "press", "turn on", "toggle on", "turn off", "pour", "pour into",
    "wipe", "spray", "sweep", "cut", "slice", "hang", "open door", "open", "close door", "close",
}


def _flat(x):
    out = []
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for e in x:
            out.extend(_flat(e))
    return out


def _flat_num(x):
    """Numeric flatten — _flat drops numbers (it keeps only strings), so frame_duration needs this."""
    out = []
    if isinstance(x, bool):
        return out
    if isinstance(x, (int, float)):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for e in x:
            out.extend(_flat_num(e))
    return out


def norm_task(name):
    return re.sub(r"\s+", "_", (name or "").strip())


def verb_of(desc: str):
    toks = desc.lower().split()
    for n in (3, 2, 1):
        cand = " ".join(toks[:n])
        if cand in CONTACT_VERBS:
            return cand
    return toks[0] if toks else ""


def episode_contact_ranges(ann: dict, pad: int, n_frames: int | None):
    """Return (contact frame-ranges [(s,e)], episode_length_est)."""
    segs = ann.get("skill_annotation", [])
    ranges = []
    max_end = 0
    for s in segs:
        nums = [int(x) for x in _flat_num(s.get("frame_duration")) if isinstance(x, (int, float))]
        if len(nums) < 2:
            continue
        start, end = min(nums), max(nums)  # segment span (robust to flat [s,e] or nested)
        max_end = max(max_end, end)
        descs = _flat(s.get("skill_description"))
        v = verb_of(descs[0]) if descs else ""
        if v in CONTACT_VERBS:
            ranges.append((max(0, start - pad), end + pad))
    N = n_frames or ann.get("meta_data", {}).get("task_duration") or max_end
    # merge overlapping ranges
    ranges.sort()
    merged = []
    for s, e in ranges:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged, int(N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--weight", type=float, default=4.0, help="sample_weight for contact frames")
    ap.add_argument("--pad", type=int, default=8, help="frames of context around each segment")
    ap.add_argument("--out", default="contact_windows.json")
    args = ap.parse_args()
    if not args.annotations and not args.glob:
        ap.error("need --annotations or --glob")

    paths = (sorted(glob.glob(args.glob)) if args.glob
             else sorted(glob.glob(f"{args.annotations}/task-*/episode_*.json")))
    out = {}
    tot_contact = tot_frames = 0
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        t = norm_task(d.get("task_name"))
        ranges, N = episode_contact_ranges(d, args.pad, None)
        cframes = sum(min(e, N) - s for s, e in ranges if s < N)
        tot_contact += max(0, cframes)
        tot_frames += N
        out.setdefault(t, []).append({
            "n_frames": N, "contact_ranges": ranges,
            "contact_frac": round(cframes / N, 3) if N else 0.0,
        })

    frac = tot_contact / tot_frames if tot_frames else 0.0
    # weight to reach a target contact sampling fraction
    def weight_for(p_target):
        if frac <= 0 or frac >= 1:
            return None
        return round(p_target * (1 - frac) / ((1 - p_target) * frac), 2)

    meta = {
        "sample_weight_contact": args.weight, "sample_weight_other": 1.0, "pad": args.pad,
        "overall_contact_frac": round(frac, 4),
        "episodes_tagged": sum(len(v) for v in out.values()),
        "weight_to_reach_25pct_contact": weight_for(0.25),
        "weight_to_reach_40pct_contact": weight_for(0.40),
        "note": "annotation signal only; add gripper-transition + EE-proximity on a box (§4).",
    }
    json.dump({"meta": meta, "by_task": out}, open(args.out, "w"), indent=2)
    print(f"episodes tagged: {meta['episodes_tagged']}")
    print(f"overall contact-frame fraction: {frac*100:.1f}%")
    print(f"  -> to make contact ~25% of sampled frames, weight ≈ {meta['weight_to_reach_25pct_contact']}")
    print(f"  -> to make contact ~40% of sampled frames, weight ≈ {meta['weight_to_reach_40pct_contact']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
