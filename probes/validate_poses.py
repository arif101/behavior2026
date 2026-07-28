"""Post-hoc validation of replay pose outputs against the published instance JSONs.

Exists because `replay_poses_batch.py` shipped with an instance-index bug: it computed
`instance = demo_id // 10`, which is correct ONLY when task_id == 0 (turning_on_radio). demo_id
encodes both (task = demo_id // 10000), so the instance is `(demo_id % 10000) // 10`. For every
other task the lookup glob-missed and the episode was written with `validation_error_m: null` —
silently unverified rather than loudly failed.

This re-checks any already-produced episodes without re-running the 6-minute replay. Compares the
first-frame pose of each tracked object against
  2026-challenge-task-instances/scenes/*/json/*<task>_instances/*_0_<instance>_template-tro_state.json

Radio reference: agreement was 0.0005–0.0012 m, so anything above ~0.01 m is a real problem.

Usage:  python validate_poses.py --poses_root /root/poses [--tol 0.01]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

DATA_PATH = os.environ.get("OG_DATA_PATH", "/root/bw/BEHAVIOR-1K/datasets")


def published(task: str, instance: int) -> dict[str, np.ndarray]:
    """All object poses published for this task instance, keyed by their JSON name."""
    pat = os.path.join(DATA_PATH, "2026-challenge-task-instances", "scenes", "*", "json",
                       f"*{task}_instances", f"*_0_{instance}_template-tro_state.json")
    hits = glob.glob(pat)
    if not hits:
        return {}
    out = {}
    for k, v in json.load(open(hits[0])).items():
        if isinstance(v, dict) and "root_link" in v:
            out[k] = np.asarray(v["root_link"]["pos"], dtype=float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_root", required=True)
    ap.add_argument("--tol", type=float, default=0.01)
    a = ap.parse_args()

    n_ok = n_bad = n_skip = 0
    per_task: dict[str, list[float]] = {}

    for f in sorted(glob.glob(os.path.join(a.poses_root, "*", "ep*.json"))):
        d = json.load(open(f))
        task, demo_id = d["task"], d["demo_id"]
        inst = (demo_id % 10000) // 10
        gt = published(task, inst)
        if not gt or not d.get("frames"):
            n_skip += 1
            continue

        # Scene names ('radio_89') differ from published JSON keys ('radio_receiver.n.01_1');
        # match on the leading category token, which is shared.
        best = None
        for name, rec in d["frames"][0]["objects"].items():
            stem = name.split("_")[0]
            for gname, gpos in gt.items():
                if stem and stem in gname:
                    e = float(np.linalg.norm(np.asarray(rec["pos"]) - gpos))
                    best = e if best is None else min(best, e)
        if best is None:
            n_skip += 1
            continue

        per_task.setdefault(task, []).append(best)
        if best <= a.tol:
            n_ok += 1
        else:
            n_bad += 1
            print(f"  BAD {task} ep{demo_id} inst{inst}: {best:.4f} m", flush=True)

    print(f"\nvalidated ok={n_ok} bad={n_bad} unmatched={n_skip}")
    print(f"{'task':44s} {'n':>5} {'median':>9} {'max':>9}")
    for t in sorted(per_task):
        e = np.array(per_task[t])
        print(f"{t:44s} {len(e):5d} {np.median(e):9.5f} {e.max():9.5f}")


if __name__ == "__main__":
    main()
