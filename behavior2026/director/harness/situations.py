"""Situation extractor for the offline Opus harness.

Builds the episode-start SITUATION REPORT (the deliberative tier's briefing) from a
recorded episode, paired with the human demonstrator's ACTUAL opening strategy as
ground truth. Output feeds two consumers:
  1. prompt construction for OpusBrain (offline prompt development — no robot, no sim)
  2. strategy-agreement scoring: does the model independently propose what the expert did?

A situation is deliberately what the live system will know at t=0: task literals,
object positions (belief-seeded), room-scale distances — never future frames.

Usage:
  python -m director.harness.situations --labels labels_10010.jsonl \
      --annotation trash_ann.json --task-literals auto --out situations.jsonl
"""
from __future__ import annotations

import argparse
import json
import math


def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def extract(labels_path: str, ann_path: str) -> dict:
    ann = json.load(open(ann_path))
    first = json.loads(open(labels_path).readline())
    objs = first["objs"]

    # ---- ground-truth opening strategy from the human (first quarter of segments)
    segs = ann["skill_annotation"]
    k = max(2, len(segs) // 4)
    opening = [{
        "skill": s["skill_description"][0],
        "objects": s["object_id"][0],
        "manipulates": (s.get("manipulating_object_id") or [None])[0],
    } for s in segs[:k]]

    # ---- pairwise geometry the briefing exposes (what makes strategy non-obvious)
    names = list(objs)
    dists = {f"{a}<->{b}": round(euclid(objs[a], objs[b]), 2)
             for i, a in enumerate(names) for b in names[i + 1:]}

    return {
        "task": ann["task_name"],
        "duration_frames": ann["meta_data"]["task_duration"],
        "objects_at_start": {k: [round(x, 2) for x in v] for k, v in objs.items()},
        "pairwise_distances_m": dists,
        "human_opening": opening,
        "human_full_skill_sequence": [s["skill_description"][0] for s in segs],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True,
                    help="pairs labels.jsonl:annotation.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.out, "w") as f:
        for pair in args.episodes:
            lab, ann = pair.split(":")
            s = extract(lab, ann)
            f.write(json.dumps(s) + "\n")
            print(f"[situation] {s['task']}: {len(s['objects_at_start'])} objects, "
                  f"opening={[o['skill'] for o in s['human_opening']]}")


if __name__ == "__main__":
    main()
