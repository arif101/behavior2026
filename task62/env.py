"""Eval-exact task-62 (halve_an_egg) environment + instance loader.

Mirrors omnigibson/eval/evaluator.py::Evaluator.load_env + _build_robot_config +
_apply_robot_eval_settings (BEHAVIOR-1K v3.9.0) so rollouts here ARE eval physics. Boot once
on the template instance; train/test instances are then applied as TRO state overlays (~1 s),
exactly as the eval harness does.

Facts this file encodes (measured on task-62; see STEP0_GAP_REPORT_62.md):
  * robot_name defaults to "robot" — the name used when the demos were RECORDED (verified in
    the hdf5 config attr). The eval harness names it "robot_r1"; the name is a pure identifier
    (uuid = hash(name)) but MUST match the demos for og.sim.load_state to deserialize them.
  * Demo config: action 30 Hz / physics 120 Hz, grasping_mode "assisted", action_normalize
    False — identical to eval/r1pro.yaml, so this env is eval-exact AND demo-exact.
  * gm.USE_GPU_DYNAMICS stays False (eval default); enabling it was rejected against Gate 0.
"""
import os
import json
import yaml

import omnigibson as og
from omnigibson.macros import gm

TASK = "halve_an_egg"
N_GOAL_LITERALS = 5           # each conjunct is worth 0.2 of q_score

if os.environ.get("B26_GPU_DYNAMICS") == "1":
    gm.USE_GPU_DYNAMICS = True


def build_env(partial_rooms=True, max_steps=None, headless=True, robot_name="robot",
              apply_eval_settings=True):
    """Boot the eval-exact env on the template instance (activity_instance_id=0)."""
    gm.HEADLESS = headless
    from gello.utils.og_teleop_cfg import DISABLED_TRANSITION_RULES
    from gello.utils.og_teleop_utils import augment_rooms, get_task_relevant_room_types, load_available_tasks
    from omnigibson.eval.utils.eval_utils import EVAL_TIMEOUT_MULTIPLIER, generate_basic_environment_config
    from omnigibson.eval.utils.score_utils import load_human_stats
    from omnigibson.eval import evaluator as EV

    # evaluator.load_env: rig-disabled transition rules off; SlicingRule (chop) stays ON
    for rule in DISABLED_TRANSITION_RULES:
        rule.ENABLED = False

    available = load_available_tasks()
    assert TASK in available, f"{TASK} not in available tasks"
    task_cfg = available[TASK][0]
    cfg = generate_basic_environment_config(task_name=TASK, task_cfg=task_cfg)
    if partial_rooms:
        rooms = get_task_relevant_room_types(activity_name=TASK)
        rooms = augment_rooms(rooms, task_cfg["scene_model"], TASK)
        cfg["scene"]["load_room_types"] = rooms

    # evaluator._build_robot_config with the default r1pro.yaml, name overridden
    robot_cfg = dict(yaml.safe_load(open(EV.DEFAULT_ROBOT_CONFIG_PATH)))
    robot_cfg["model"] = robot_cfg["model"].lower()
    eval_cfg = robot_cfg.pop("eval", None) or {}
    robot_cfg["name"] = robot_name
    robot_cfg["position"] = task_cfg["robot_start_position"]
    robot_cfg["orientation"] = task_cfg["robot_start_orientation"]
    cfg["robots"] = [robot_cfg]

    hs = load_human_stats(TASK)
    cfg["task"]["termination_config"]["max_steps"] = max_steps or int(hs["length"] * EVAL_TIMEOUT_MULTIPLIER)
    cfg["task"]["include_obs"] = False

    env = og.Environment(configs=cfg)
    env._eval_robot_config = eval_cfg
    og.sim.update_handles()

    if apply_eval_settings:
        # evaluator._apply_robot_eval_settings
        robot = env.robots[0]
        og.sim.stop()
        robot.base_footprint_link.mass = EV.EVAL_BASE_LINK_MASS
        og.sim.play()
        head = eval_cfg.get("camera_sensor_names", {}).get("head")
        if head is not None:
            sensor_name = head.split("::")[-1].replace("robot_r1", robot_name, 1)
            if sensor_name in robot.sensors:
                robot.sensors[sensor_name].horizontal_aperture = EV.EVAL_HEAD_HORIZONTAL_APERTURE
    return env, hs


def load_instance(env, instance_id, mode="train", settle_steps=25):
    """Apply a task instance's TRO state (objects + robot pose) onto the booted env."""
    from omnigibson.utils.asset_utils import get_task_instance_path
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    task = env.task
    scene_model = task.scene_name
    tro_filename = task.get_cached_activity_scene_filename(
        scene_model=scene_model, activity_name=task.activity_name,
        activity_definition_id=task.activity_definition_id, activity_instance_id=instance_id,
    )
    path = get_task_instance_path(
        scene_model, f"{scene_model}_task_{task.activity_name}_instances/{tro_filename}-tro_state", mode=mode)
    if path is None:
        raise FileNotFoundError(f"no {mode} instance {instance_id} for {task.activity_name}/{scene_model}")

    robot = env.robots[0]
    state = recursively_convert_to_torch(json.load(open(path)))
    for key, val in state.items():
        if key == "robot_poses":
            poses = {k.lower(): v for k, v in val.items()}
            avail = poses.get("robot", poses.get(robot.model))
            if avail is None:
                raise KeyError(f"no robot pose in instance {instance_id}")
            robot.set_position_orientation(avail[0]["position"], avail[0]["orientation"])
            env.scene.write_task_metadata(key=key, data=val)
        else:
            task.object_scope[key].load_state(val, serialized=False)

    og.sim.update_handles()
    for _ in range(settle_steps):
        og.sim.step_physics()
    return path


def goal_status(env):
    """(q_score, satisfied, unsatisfied) using the challenge's own predicate check."""
    task = env.task
    _, sat = task.compiled_task.check_goal(task._evaluate_predicate)
    s, u = sat["satisfied"], sat["unsatisfied"]
    return len(s) / max(1, len(s) + len(u)), s, u
