"""Run-3 arm eval: official OmniGibson Evaluator on instance 301 + grasp-completion probe.
Connects to an already-running policy websocket (serve_a0.py). Adds per-rollout:
  grasp_completed : did the eval's assisted-grasp weld fire on the radio (either arm)?
  fingertip_min_m : min fingertip->radio distance over the rollout (approach-stall probe).
Writes json/<task>_<inst>_<r>.json (superset of the official schema) + grasp_summary.json.
Run (behavior env):  OMNIGIBSON_HEADLESS=1 python -u /root/eval_run3_arm.py \
  --arm a0 --num-rollouts 25 --port 8000 --output-dir /root/a0_eval [--write-video]
"""
import argparse, json, os
import numpy as np

def _dump(obj, path):
    """Write JSON durably: flush+fsync before returning, so os._exit(0) can't drop the buffer."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
        f.flush(); os.fsync(f.fileno())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--task-name", default="turning_on_radio")
    ap.add_argument("--num-rollouts", type=int, default=25)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = default timeout (EVAL_TIMEOUT_MULTIPLIER x human avg)")
    ap.add_argument("--output-dir", default="/root/a0_eval")
    ap.add_argument("--write-video", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip rollouts whose json already exists")
    a = ap.parse_args()

    from omegaconf import OmegaConf
    from omnigibson.macros import gm
    from omnigibson.eval.evaluator import Evaluator, resolve_instance_ids
    from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
    gm.HEADLESS = True
    seed = seed_everything(DEFAULT_EVAL_SEED)
    inst = resolve_instance_ids(a.task_name, [0], mode="public_test")[0]
    print(f"EVAL arm={a.arm} task={a.task_name} instance={inst} n={a.num_rollouts} seed={seed}", flush=True)
    rc = OmegaConf.load("/root/bw/BEHAVIOR-1K/OmniGibson/omnigibson/eval/r1pro.yaml")
    cfg = OmegaConf.create({
        # VALIDATED scoring path: AffordanceMapFullRes injects head-RGB affordance target_points
        # + live map tokens (requires MAP_ARM, B1K_TASK_TARGETS env + the 3 passthrough patches).
        # The default EnvironmentWrapper leaves the policy blind (null-sentinel ~0.5-0.78m). See RUN2ref validation.
        "env_wrapper": {"_target_": "behavior2026_eval.affordance_map_fullres.AffordanceMapFullRes"},
        "policy_name": "websocket",
        "model": {"_target_": "omnigibson.eval.policies.WebsocketPolicy", "host": a.host, "port": a.port},
        "headless": True, "partial_scene_load": True, "max_steps": (a.max_steps or None),
        "write_video": a.write_video, "mode": "public_test", "seed": seed,
        "task": {"name": a.task_name}, "robot": rc,
    })
    jdir = os.path.join(a.output_dir, "json"); os.makedirs(jdir, exist_ok=True)
    vdir = os.path.join(a.output_dir, "videos"); os.makedirs(vdir, exist_ok=True)

    rows = []
    with Evaluator(cfg) as ev:
        radio = next(o for o in ev.env.scene.objects if "radio" in o.name.lower())
        rob = ev.robot
        def grasped():
            try:
                p = rob._ag_obj_constraint_params
                return (p.get("right") is not None) or (p.get("left") is not None)
            except Exception:
                return False
        from omnigibson.object_states import ToggledOn as _TO
        def ee_button_min():
            try:
                btn = np.asarray(radio.states[_TO].link.get_position_orientation()[0]).reshape(-1)[:3]
                d = 9.9
                for arm in ("left","right"):
                    ep = np.asarray(rob.eef_links[arm].get_position_orientation()[0]).reshape(-1)[:3]
                    d = min(d, float(np.linalg.norm(ep - btn)))
                return d
            except Exception:
                return None
        def fingertip_min():
            try:
                import torch as th
                rp = np.asarray(radio.get_position_orientation()[0]).reshape(-1)[:3]
                d = 9.9
                for arm in ("left", "right"):
                    for l in rob.finger_links[arm]:
                        fp = np.asarray(l.get_position_orientation()[0]).reshape(-1)[:3]
                        d = min(d, float(np.linalg.norm(fp - rp)))
                return d
            except Exception:
                return None
        ev.reset(); ev.load_task_instance(int(inst))
        orig_step = ev.step
        state = {"grasp": False, "fmin": 9.9, "ebmin": 9.9}
        def probed_step():
            t, tr = orig_step()
            if grasped(): state["grasp"] = True
            fm = fingertip_min()
            if fm is not None: state["fmin"] = min(state["fmin"], fm)
            eb = ee_button_min()
            if eb is not None: state["ebmin"] = min(state["ebmin"], eb)
            return t, tr
        for r in range(a.num_rollouts):
            _rp = os.path.join(jdir, f"{a.task_name}_{inst}_{r}.json")
            if a.resume and os.path.exists(_rp):
                rows.append(json.load(open(_rp))); print(f"RESUME skip rollout {r} (exists)", flush=True); continue
            ev.reset(); state["grasp"] = False; state["fmin"] = 9.9; state["ebmin"] = 9.9
            if a.write_video:
                ev.start_recording(os.path.join(vdir, f"{a.task_name}_{inst}_{r}.mp4"), rate=30)
            term = trunc = False; steps = 0
            while not (term or trunc):
                term, trunc = probed_step(); steps += 1
            succ = bool(ev.env.task.success)
            m = {}
            for metric in ev.metrics: m.update(metric.aggregate(ev.env))
            row = {"task": a.task_name, "instance_id": int(inst), "rollout_id": r, "steps": steps,
                   "success": succ, "grasp_completed": state["grasp"],
                   "fingertip_min_m": round(state["fmin"], 4), "ee_button_min_m": round(state["ebmin"], 4), "q_score": m.get("q_score", {})}
            _dump(row, os.path.join(jdir, f"{a.task_name}_{inst}_{r}.json"))
            if a.write_video: ev.stop_recording()
            gc = sum(x["grasp_completed"] for x in rows) + row["grasp_completed"]
            sc = sum(x["success"] for x in rows) + row["success"]
            print(f"ROLLOUT {r}: success={succ} grasp={state['grasp']} fmin={state['fmin']:.3f} "
                  f"eb={state['ebmin']:.3f} steps={steps} | running grasp={gc}/{r+1} success={sc}/{r+1}", flush=True)
            rows.append(row)
        # summary computed INSIDE the `with Evaluator` block: its __exit__ (Omnigibson
        # simulator teardown) hard-terminates the process, so anything after the block never runs.
        summ = {"arm": a.arm, "instance": int(inst), "n": len(rows),
                "grasp_completed": sum(x["grasp_completed"] for x in rows),
                "success": sum(x["success"] for x in rows),
                "fingertip_min_median": round(float(np.median([x["fingertip_min_m"] for x in rows])), 4)}
        _dump({"summary": summ, "rows": rows}, os.path.join(a.output_dir, f"grasp_summary_{a.arm}.json"))
        print("EVAL_DONE", json.dumps(summ), flush=True)
    os._exit(0)

main()
