"""Use OmniGibson as its OWN state parser — the correct answer to the 623-float `state` array.

Background
----------
`probes/state_decoder.py` hardcodes a uniform 15-wide object block. That is wrong: per the
OmniGibson source, a serialized object is

    7 (pos+ori)  +  3N (joint pos/vel/eff)  +  S (per-object stateful states)

so widths are object-dependent (a radio with ToggledOn differs from a rigid box). Blind
segmentation is underdetermined — one radio episode admits >10^7 exact tilings — so parsing it
ourselves is a dead end.

We don't have to. `scene.load_state(state, serialized=True)` is OmniGibson's own deserializer and
it already knows the registry. This probe dumps the ground truth we need:

  * the object registry ORDER (what fixes block order in the flat array)
  * each object's serialized WIDTH (validating 7 + 3N + S)
  * the scene's total serialized length, which must equal the rawdata episode's state dim

CRITICAL: the env must be built the way the DEMOS were recorded, or the registry — and therefore
every offset — differs. So we call the challenge's own `generate_basic_environment_config`
(omnigibson/eval/utils/eval_utils.py, used by evaluator.py:173) rather than hand-rolling a config.
Hand-rolled configs failed here repeatedly.
"""

from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="turning_on_radio")
    ap.add_argument("--expect-dim", type=int, default=623,
                    help="state dim of the matching rawdata episode (623 for turning_on_radio)")
    ap.add_argument("--out", default="/root/registry_probe.json")
    ap.add_argument("--partial", action="store_true", default=True,
                    help="load only task-relevant rooms, as the demo recorder does")
    ap.add_argument("--full-scene", dest="partial", action="store_false")
    ap.add_argument("--augment", action="store_true", default=True,
                    help="augment_rooms() adds neighbouring rooms; the recorder may not have")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    a = ap.parse_args()

    import omnigibson as og
    # NB: these live in two different packages — evaluator.py pulls the task list from `gello`
    # and the config builder from `eval_utils`. Importing both from eval_utils fails.
    from gello.utils.og_teleop_utils import (
        augment_rooms,
        get_task_relevant_room_types,
        load_available_tasks,
    )
    from omnigibson.eval.utils.eval_utils import generate_basic_environment_config

    tasks = load_available_tasks()
    assert a.task in tasks, f"unknown task {a.task!r}; have {list(tasks)[:5]}…"
    task_cfg = tasks[a.task][0]
    cfg = generate_basic_environment_config(task_name=a.task, task_cfg=task_cfg)
    cfg["task"]["include_obs"] = False

    # PARTIAL SCENE LOAD — the demos are recorded this way (evaluator.py:156). Loading the full
    # scene gives 262 objects / dump_state 3171, while a rawdata episode is 623: the registry (and
    # therefore every byte offset) only matches when the same rooms are loaded.
    if a.partial:
        rooms = get_task_relevant_room_types(activity_name=a.task)
        if a.augment:
            rooms = augment_rooms(rooms, task_cfg["scene_model"], a.task)
        cfg["scene"]["load_room_types"] = rooms
        print(f"partial scene load (augment={a.augment}), rooms = {rooms}")

    # generate_basic_environment_config() returns NO robots — evaluator.py adds them in a separate
    # step (_build_robot_config, evaluator.py:203). Skipping it makes BDDLSampler die on
    # `self._env.robots[0]` with IndexError, then segfault. Mirror that step here, using the same
    # canonical r1pro.yaml the challenge evaluates with.
    import os

    from omegaconf import OmegaConf

    import omnigibson.eval as _eval_pkg

    robot_cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(os.path.dirname(_eval_pkg.__file__), "r1pro.yaml")),
        resolve=True,
    )
    robot_cfg.pop("eval", None)                       # eval-only block, not an env kwarg
    robot_cfg["model"] = robot_cfg["model"].lower()
    robot_cfg["position"] = task_cfg["robot_start_position"]
    robot_cfg["orientation"] = task_cfg["robot_start_orientation"]
    cfg["robots"] = [robot_cfg]

    env = og.Environment(configs=cfg)
    scene = env.scene

    rows, total = [], 0
    for obj in scene.objects:
        try:
            w = int(obj.serialize(obj.dump_state(serialized=False)).shape[0])
        except Exception as e:  # noqa: BLE001 — probe: record and continue
            w = -1
            print(f"  ! serialize failed for {obj.name}: {type(e).__name__}: {e}")
        stateful = []
        try:
            stateful = [getattr(k, "__name__", str(k))
                        for k, v in obj.states.items() if getattr(v, "stateful", False)]
        except Exception:  # noqa: BLE001
            pass
        rows.append({"name": obj.name, "width": w,
                     "n_joints": int(getattr(obj, "n_joints", 0) or 0),
                     "stateful": sorted(stateful)})
        total += max(w, 0)

    print(f"\nregistry: {len(rows)} objects, widths sum = {total}")
    print(f"{'idx':>4} {'off':>6} {'width':>6} {'njnt':>5}  name")
    off = 0
    for i, r in enumerate(rows):
        print(f"{i:>4} {off:>6} {r['width']:>6} {r['n_joints']:>5}  {r['name']}  {r['stateful'] or ''}")
        off += max(r["width"], 0)

    n = int(len(scene.dump_state(serialized=True)))
    print(f"\nscene.dump_state(serialized=True) length = {n}")

    # THE GATE: match the rawdata episode's state dim and every recorded offset becomes
    # interpretable. Mismatch means this env does not reproduce how the demos were recorded.
    verdict = "MATCH" if n == a.expect_dim else f"MISMATCH (expected {a.expect_dim}, got {n})"
    print(f"GATE vs rawdata state dim: {verdict}")

    with open(a.out, "w") as f:
        json.dump({"task": a.task, "objects": rows, "widths_sum": total,
                   "scene_state_len": n, "expect_dim": a.expect_dim, "gate": verdict}, f, indent=1)
    print(f"wrote {a.out}")

    og.shutdown()


if __name__ == "__main__":
    main()
