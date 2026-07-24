#!/usr/bin/env python3
"""Staged reward map from demo skill annotations (DATA_ASSEMBLY_SPEC.md §5, AUGUST §5.4).

RL from a 0% floor needs a DENSE, MONOTONE, per-subtask reward (sparse 0/1 gives no gradient).
This builds that schedule per task from the ordered skill sequence in the annotations:
each skill segment -> a checkable predicate (holding/on/open/toggled/near) + a cumulative
reward weight, with manipulation/contact subgoals weighted above navigation, and a terminal
bonus. Output feeds (a) the RL reward, (b) the cofounder's stage-head target (same schedule),
(c) the contact-oversampling cross-reference (is_contact subgoals).

Annotation-derived (no GPU). TODO on a sim box: refine the predicate triggers against the
task BDDL goal conditions (BEHAVIOR-1K/bddl3) for exact sim-checkable predicates.

Usage:
  python build_reward_map.py --glob "/path/real_ann/*.json" --out reward_shaping.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re

# verb -> (predicate template, category, base weight). category: nav | manip | contact.
VERB_MAP = {
    "move to": ("near({obj})", "nav", 0.5),
    "navigate": ("near({obj})", "nav", 0.5),
    "walk to": ("near({obj})", "nav", 0.5),
    "pick up": ("holding({obj})", "contact", 1.5),
    "pick up from": ("holding({obj})", "contact", 1.5),
    "grasp": ("holding({obj})", "contact", 1.5),
    "put down": ("placed({obj})", "contact", 1.3),
    "place on": ("ontop({obj},{tgt})", "contact", 1.5),
    "place in": ("inside({obj},{tgt})", "contact", 1.5),
    "place next": ("nextto({obj},{tgt})", "contact", 1.3),
    "place next to": ("nextto({obj},{tgt})", "contact", 1.3),
    "open door": ("open({obj})", "manip", 1.2),
    "open": ("open({obj})", "manip", 1.2),
    "close door": ("closed({obj})", "manip", 1.0),
    "close": ("closed({obj})", "manip", 1.0),
    "press": ("toggled_on({obj})", "contact", 1.5),
    "turn on": ("toggled_on({obj})", "contact", 1.5),
    "toggle on": ("toggled_on({obj})", "contact", 1.5),
    "turn off": ("toggled_off({obj})", "contact", 1.3),
    "pour": ("filled({tgt})", "contact", 1.5),
    "pour into": ("filled({tgt})", "contact", 1.5),
    "wipe": ("clean({obj})", "contact", 1.4),
    "spray": ("sprayed({obj})", "contact", 1.4),
    "sweep": ("clean({obj})", "contact", 1.4),
    "cut": ("cut({obj})", "contact", 1.5),
    "slice": ("cut({obj})", "contact", 1.5),
    "hang": ("hung({obj})", "contact", 1.4),
}
DEFAULT = ("done({obj})", "manip", 1.0)


def _flat(x) -> list:
    out = []
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for e in x:
            out.extend(_flat(e))
    return out


def norm_task(name):
    return re.sub(r"\s+", "_", (name or "").strip())


def strip_inst(o):
    if not isinstance(o, str):
        return ""
    return re.sub(r"_[a-z0-9]+$", "", re.sub(r"_\d+$", "", o))


def verb_of(desc: str):
    toks = desc.lower().split()
    two = " ".join(toks[:2]) if len(toks) >= 2 else ""
    three = " ".join(toks[:3]) if len(toks) >= 3 else ""
    if three in VERB_MAP:
        return three
    if two in VERB_MAP:
        return two
    return toks[0] if toks else ""


def subgoals_for_task(ann_episodes: list[dict]) -> list[dict]:
    """Ordered, de-duplicated subgoal list from the (most complete) demo's skill sequence."""
    # pick the episode with the most segments as the reference sequence
    ep = max(ann_episodes, key=lambda e: len(e.get("skill_annotation", [])))
    segs = sorted(ep.get("skill_annotation", []), key=lambda s: s.get("skill_idx", 0))
    subs, seen = [], set()
    for s in segs:
        descs = _flat(s.get("skill_description"))
        manip = _flat(s.get("manipulating_object_id"))
        objs = _flat(s.get("object_id"))
        if not descs:
            continue
        v = verb_of(descs[0])
        tmpl, cat, w = VERB_MAP.get(v, DEFAULT)
        obj = strip_inst((manip or objs or [""])[0])
        tgt = strip_inst(objs[-1]) if len(objs) > 1 else ""
        pred = tmpl.format(obj=obj or "obj", tgt=tgt or "target")
        key = (v, obj, tgt)
        if key in seen:  # collapse repeats (e.g., 3 cans -> one "holding(can)" rung is enough)
            continue
        seen.add(key)
        subs.append({"verb": v, "predicate": pred, "category": cat,
                     "is_contact": cat == "contact", "base_weight": w})
    return subs


def assign_rewards(subs: list[dict], terminal_bonus: float = 0.3) -> list[dict]:
    """Monotone cumulative schedule: normalize base weights to sum (1 - bonus), terminal += bonus."""
    tot = sum(s["base_weight"] for s in subs) or 1.0
    scale = (1.0 - terminal_bonus)
    cum = 0.0
    out = []
    for i, s in enumerate(subs):
        r = s["base_weight"] / tot * scale
        if i == len(subs) - 1:
            r += terminal_bonus
        cum += r
        out.append({**s, "reward": round(r, 4), "cum_reward": round(cum, 4),
                    "subgoal_idx": i})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--glob", default=None)
    ap.add_argument("--out", default="reward_shaping.json")
    args = ap.parse_args()
    if not args.annotations and not args.glob:
        ap.error("need --annotations DIR or --glob PATTERN")

    paths = (sorted(glob.glob(args.glob)) if args.glob
             else sorted(glob.glob(f"{args.annotations}/task-*/episode_*.json")))
    by_task = {}
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        t = norm_task(d.get("task_name"))
        by_task.setdefault(t, []).append(d)

    reward_map, warnings = {}, []
    for t, eps in sorted(by_task.items()):
        subs = subgoals_for_task(eps)
        if not subs:
            warnings.append(f"{t}: no subgoals")
            continue
        sched = assign_rewards(subs)
        # monotonicity audit
        cums = [s["cum_reward"] for s in sched]
        if any(cums[i] < cums[i - 1] for i in range(1, len(cums))):
            warnings.append(f"{t}: NON-MONOTONE (bug)")
        reward_map[t] = {
            "n_subgoals": len(sched), "n_contact": sum(s["is_contact"] for s in sched),
            "schedule": sched,
        }
    json.dump({"reward_map": reward_map, "warnings": warnings}, open(args.out, "w"), indent=2)

    print(f"tasks with reward schedules: {len(reward_map)}")
    ex = "turning_on_radio" if "turning_on_radio" in reward_map else next(iter(reward_map))
    print(f"\nexample — {ex}:")
    for s in reward_map[ex]["schedule"]:
        flag = " [contact]" if s["is_contact"] else ""
        print(f"  {s['subgoal_idx']}. {s['predicate']:28s} +{s['reward']:.2f} "
              f"(cum {s['cum_reward']:.2f}){flag}")
    n_contact = sum(v["n_contact"] for v in reward_map.values())
    print(f"\ntotal contact subgoals across corpus: {n_contact}")
    if warnings:
        print("warnings:", warnings[:8])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
