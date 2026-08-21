"""Trace-replay harness: drive the reflex tier from a REAL recorded episode.

Oracle stubs answer from the recording: belief = true object positions (labels JSONL),
stage phases = official annotation segments, p_sat = real containment geometry computed
from recorded positions, grounding = true position of the bound instance + noise.
EE is approximated from the manipulated-object trajectory (parquet proprio integration
is the planned upgrade); decisions, not millimeters, are under test here.

Output: a trace JSONL in the ONE schema shared with live rollouts (trace.py).

Usage:
  python -m director.harness.replay --labels labels_10010.jsonl \
      --annotation trash_ann.json --out trace_10010.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from director.core import Config, Literal  # noqa: E402
from director.reflex import Reflex  # noqa: E402
from director.trace import TraceWriter  # noqa: E402

INSIDE_R_XY = 0.30   # containment test: horizontal distance to container (m)
INSIDE_DZ = 0.55     # and vertical proximity (m)


def load_episode(labels_path: str, ann_path: str):
    rows = [json.loads(l) for l in open(labels_path)]
    ann = json.load(open(ann_path))
    n = min(ann["meta_data"]["task_duration"], len(rows) // 2)
    objs_at = lambda k: rows[min(int(k * len(rows) / n), len(rows) - 1)]["objs"]
    return n, objs_at, ann["skill_annotation"]


def build_tracks(n, segs, container):
    """Per-frame: phase per arm, R-arm EE approximation, attachment windows."""
    phase_map = {"move to": "nav", "pick up from": "contact", "place in": "contact",
                 "place on": "contact"}
    # attachment: container -> L from mid of its first manip seg to end of its last;
    # other objects -> R from mid of pick seg to end of the following place seg
    attach = {}   # frame -> {instance: arm}
    manip_of = [None] * n
    phases = [{"L": "idle", "R": "transition"} for _ in range(n)]
    for s in segs:
        a, b = s["frame_duration"]
        b = min(b, n)
        who = "L" if (s.get("manipulating_object_id") or [None])[0] == container else "R"
        ph = phase_map.get(s["skill_description"][0], "transition")
        for f in range(a, b):
            phases[f][who] = ph
            if s.get("manipulating_object_id"):
                manip_of[f] = (s["manipulating_object_id"][0], who)
    cont_manips = [s for s in segs if (s.get("manipulating_object_id") or [None])[0] == container]
    if cont_manips:
        c0 = (cont_manips[0]["frame_duration"][0] + cont_manips[0]["frame_duration"][1]) // 2
        c1 = min(cont_manips[-1]["frame_duration"][1] - 30, n)
        for f in range(c0, c1):
            attach.setdefault(f, {})[container] = "L"
    cur = None
    for s in segs:
        obj = (s.get("manipulating_object_id") or [None])[0]
        if obj is None or obj == container:
            continue
        a, b = s["frame_duration"]
        if s["skill_description"][0] == "pick up from":
            cur = (obj, (a + b) // 2)
        elif s["skill_description"][0] in ("place in", "place on") and cur and cur[0] == obj:
            for f in range(cur[1], min(b, n)):
                attach.setdefault(f, {})[obj] = "R"
            cur = None
    return phases, manip_of, attach


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(0)

    n, objs_at, segs = load_episode(args.labels, args.annotation)
    container = "trash_can_116"
    cans = ["can_of_soda_114", "can_of_soda_115", "can_of_soda_113"]  # human order
    literals = [Literal(i, "inside", "can_of_soda", c, container,
                        est_ticks=2000, skill_type="uncoordinated")
                for i, c in enumerate(cans)]
    phases, manip_of, attach = build_tracks(n, segs, container)

    rf = Reflex(literals, Config())
    tw = TraceWriter(args.out)
    ee_r = list(objs_at(0)[cans[0]])
    latches, q_final = [], 0.0

    for k in range(0, n, args.stride):
        objs = objs_at(k)
        att = attach.get(k, {})
        # --- EE approximation: manip object if any, else glide toward bound target
        m = manip_of[k]
        if m and m[1] == "R":
            ee_r = list(objs[m[0]])
        else:
            tgt = None
            for arm in rf.st.arms.values():
                if arm.assigned is not None:
                    inst = rf._bound.get(arm.assigned)
                    if inst:
                        tgt = objs[inst]
                        break
            if tgt:
                ee_r = [a + (b - a) * 0.02 for a, b in zip(ee_r, tgt)]
        ee_l = list(objs[container]) if att.get(container) == "L" else [3.0, 4.0, 0.6]
        # --- real containment geometry -> p_sat
        p_sat, visible = {}, {}
        for lit in literals:
            c, t = objs[lit.target_instance], objs[container]
            dxy = math.hypot(c[0] - t[0], c[1] - t[1])
            inside = dxy < INSIDE_R_XY and abs(c[2] - t[2]) < INSIDE_DZ
            p_sat[lit.lid] = 0.97 if inside else 0.03
            visible[lit.lid] = True
        belief = {name: {"mu": tuple(p), "sigma_tr": 0.05, "p_exists": 0.99,
                         "age": 0.05, "attached_to": att.get(name)}
                  for name, p in objs.items()}
        grounding = {}
        for aname, arm in rf.st.arms.items():
            inst = rf._bound.get(arm.assigned) if arm.assigned is not None else None
            if inst:
                p = objs[inst]
                grounding[aname] = {"point": tuple(x + rng.gauss(0, 0.01) for x in p),
                                    "conf": 0.92, "instance": inst}
        obs = {
            "time_remaining": n - k,
            "stage": {a: {"phase": phases[k][a],
                          "progress": 0.5, "entropy": 0.1} for a in ("L", "R")},
            "p_sat": p_sat, "visible": visible,
            "grounding": grounding, "belief": belief,
            "pfail": {"L": 0.04 + rng.random() * 0.02, "R": 0.04 + rng.random() * 0.02},
            "ee": {"L": tuple(ee_l), "R": tuple(ee_r)},
        }
        out = rf.tick(obs)
        tw.rec(obs, out)
        for ev in out["events"]:
            if ev[0] == "latch":
                latches.append((k, ev[1]))
        q_final = out["q"]
    tw.close()
    print(f"frames={n} q_final={q_final:.2f} latches={latches}")
    print(f"ledger={ {l: s.value for l, s in rf.st.lstate.items()} }")
    print(f"events_ok={'terminal' if rf.st.terminal else 'ran-to-end'}")


if __name__ == "__main__":
    main()
