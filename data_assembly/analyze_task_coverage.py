#!/usr/bin/env python3
"""Diversity-first task selection for August Phase A (DATA_ASSEMBLY_SPEC.md §2).

Builds a (task x primitive-verb x object-category x skill_type) coverage matrix from the
challenge demo annotations, then greedily samples a task set that MAXIMIZES taxonomy coverage
under a budget — because generalization scales with diversity, not depth. Gate tasks are
forced in (Tier 0); tasks sharing gate primitives are prioritized (Tier 1); the rest fill
uncovered taxonomy cells (Tier 2).

No GPU. Reads annotation JSONs from a local mirror of
  behavior-1k/2026-challenge-demos :: annotations/task-XXXX/episode_*.json
(one representative episode per task is enough for the coverage signature).

Annotation schema (per episode JSON), consistent with harness/situations.py:
  { "task_name": str,
    "skill_annotation": [ { "skill_description": [str, ...],
                            "object_id": [str, ...],
                            "manipulating_object_id": [str|null, ...],
                            "skill_type": str }, ... ],
    "meta_data": {...} }

Usage:
  python analyze_task_coverage.py --annotations /path/to/2026-challenge-demos/annotations \
      --budget 20 --out phaseA_taskset.json
  # or point at a flat dir of *.json episode annotations with --glob
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

GATE_TASKS = {  # Tier 0 — always included (deep labels + eval baselines anchor the A/Bs)
    "turning_on_radio", "picking_up_trash",
    "attach_a_camera_to_a_tripod", "thawing_frozen_food",
}


def strip_instance(obj_id: str) -> str:
    """radio_89 -> radio ; can_of_soda_113 -> can_of_soda ; bar_byvbuc_0 -> bar."""
    if not obj_id:
        return ""
    return re.sub(r"_[a-z0-9]+$", "", re.sub(r"_\d+$", "", obj_id))


def primitive_verb(desc: str) -> str:
    """First 1-2 tokens of a skill description ('pick up from' -> 'pick up')."""
    toks = desc.lower().split()
    if not toks:
        return ""
    two = " ".join(toks[:2])
    return two if two in _KNOWN_TWO else toks[0]


_KNOWN_TWO = {"pick up", "put down", "open door", "close door", "move to",
              "place in", "place on", "place next", "turn on", "turn off"}


def task_signature(ann_episodes: list[dict]) -> dict:
    """Aggregate a task's coverage signature from >=1 episode annotation(s)."""
    verbs, objs, types = set(), set(), set()
    for ep in ann_episodes:
        for seg in ep.get("skill_annotation", []):
            for d in (seg.get("skill_description") or []):
                v = primitive_verb(d)
                if v:
                    verbs.add(v)
            for o in (seg.get("object_id") or []):
                c = strip_instance(o)
                if c:
                    objs.add(c)
            for m in (seg.get("manipulating_object_id") or []):
                c = strip_instance(m)
                if c:
                    objs.add(c)
            st = seg.get("skill_type")
            if st:
                types.add(st)
    return {"verbs": verbs, "objects": objs, "skill_types": types}


def cell_set(sig: dict) -> set:
    """Taxonomy cells this task covers (used for greedy set-cover)."""
    cells = set()
    for v in sig["verbs"]:
        cells.add(("verb", v))
    for o in sig["objects"]:
        cells.add(("obj", o))
    for t in sig["skill_types"]:
        cells.add(("type", t))
    return cells


def load_task_annotations(anndir: str, flat_glob: str | None) -> dict[str, list[dict]]:
    """task_name -> list of episode annotation dicts (>=1 per task)."""
    paths = (sorted(glob.glob(flat_glob)) if flat_glob
             else sorted(glob.glob(os.path.join(anndir, "task-*", "episode_*.json"))))
    by_task: dict[str, list[dict]] = defaultdict(list)
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        t = d.get("task_name") or os.path.basename(os.path.dirname(p))
        if len(by_task[t]) < 3:  # 1-3 episodes is enough for a coverage signature
            by_task[t].append(d)
    return dict(by_task)


def gate_primitive_cells(sigs: dict[str, dict]) -> set:
    """Cells covered by the Tier-0 gate tasks (Tier-1 = tasks that share these)."""
    cells = set()
    for t in GATE_TASKS:
        if t in sigs:
            cells |= cell_set(sigs[t])
    return cells


def greedy_cover(sigs: dict[str, dict], budget: int) -> list[dict]:
    """Tier 0 forced; then greedy max-marginal-coverage to the budget."""
    chosen, covered = [], set()

    def add(task, tier):
        cells = cell_set(sigs[task])
        gain = len(cells - covered)
        chosen.append({"task": task, "tier": tier,
                       "adds_cells": gain, "n_verbs": len(sigs[task]["verbs"]),
                       "n_objs": len(sigs[task]["objects"])})
        covered.update(cells)

    for t in sorted(GATE_TASKS):
        if t in sigs:
            add(t, 0)
    gate_cells = gate_primitive_cells(sigs)
    remaining = {t for t in sigs if t not in {c["task"] for c in chosen}}

    # Tier 1: tasks sharing gate primitives, by marginal gain
    while len(chosen) < budget and remaining:
        cand = [(t, len(cell_set(sigs[t]) - covered)) for t in remaining
                if cell_set(sigs[t]) & gate_cells]
        cand = [c for c in cand if c[1] > 0]
        if not cand:
            break
        t, _ = max(cand, key=lambda x: x[1])
        add(t, 1); remaining.discard(t)

    # Tier 2: pure taxonomy fillers, by marginal gain
    while len(chosen) < budget and remaining:
        t, gain = max(((t, len(cell_set(sigs[t]) - covered)) for t in remaining),
                      key=lambda x: x[1])
        if gain == 0:
            break
        add(t, 2); remaining.discard(t)

    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=None, help="dir with task-*/episode_*.json")
    ap.add_argument("--glob", default=None, help="flat glob of episode annotation *.json")
    ap.add_argument("--budget", type=int, default=20, help="task count for Phase A1")
    ap.add_argument("--out", default="phaseA_taskset.json")
    args = ap.parse_args()
    if not args.annotations and not args.glob:
        ap.error("need --annotations DIR or --glob PATTERN")

    by_task = load_task_annotations(args.annotations, args.glob)
    if not by_task:
        raise SystemExit("no annotations found — check --annotations/--glob path")
    sigs = {t: task_signature(eps) for t, eps in by_task.items()}

    all_verbs = set().union(*(s["verbs"] for s in sigs.values()))
    all_objs = set().union(*(s["objects"] for s in sigs.values()))
    all_types = set().union(*(s["skill_types"] for s in sigs.values()))
    total_cells = len(all_verbs) + len(all_objs) + len(all_types)

    chosen = greedy_cover(sigs, args.budget)
    covered = set()
    for c in chosen:
        covered |= cell_set(sigs[c["task"]])

    out = {
        "n_tasks_total": len(sigs),
        "taxonomy": {"verbs": len(all_verbs), "objects": len(all_objs),
                     "skill_types": sorted(all_types)},
        "budget": args.budget,
        "selected": chosen,
        "coverage": {"cells_covered": len(covered), "cells_total": total_cells,
                     "pct": round(100.0 * len(covered) / max(1, total_cells), 1)},
    }
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"tasks={len(sigs)}  taxonomy cells={total_cells} "
          f"(verbs {len(all_verbs)} / objs {len(all_objs)} / types {len(all_types)})")
    print(f"selected {len(chosen)} tasks covering "
          f"{out['coverage']['pct']}% of the taxonomy:")
    for c in chosen:
        print(f"  T{c['tier']}  +{c['adds_cells']:2d} cells  {c['task']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
