"""Per-frame goal-predicate evaluation over recorded sim state (v4).

Replaces the ledger v1 proxy (build_cache.py COMPLETING_SKILLS flips) with the
real thing: replay each rawdata HDF5 through OmniGibson playback and evaluate
every goal literal (stage/goal_literals.py specs) against live object states --
the same state machinery leaderboard scoring reads. One pass per episode also
emits the extractor labels JSONL (camera M + object world positions, now with
AABB half-extents for nearest-point distance, gotcha #3), so the per-arm
attribution pipeline needs no second replay.

Playback conventions inherited from probes/replay_labeled.py (validated):
env construction, boot mutex, zed prim/link camera pose, the 2-records-per-
video-frame callback cadence, benign teardown segfaults after DONE lines.

Usage (behavior conda env, cwd=/root/BEHAVIOR-1K/OmniGibson):
  python /root/stage/eval_predicates.py --manifest /root/data/manifest.json \
      --rawdata_root /root/rawdata --task_literals /root/task_literals.json \
      --out_dir /root/predicates --labels_dir /root/sweep_labels_v4 \
      [--only task/file_idx ...]

Success contract: judge by the PRED_DONE line + <out>.done marker, never the
exit code.
"""

import argparse
import json
import contextlib
import os
import re
import sys

import numpy as np

import aabb_predicates

REPLAY_OBS_DIR = os.environ.get("REPLAY_OBS_DIR",
                                "/root/BEHAVIOR-1K/OmniGibson/scripts/learning")
sys.path.insert(0, REPLAY_OBS_DIR)
import replay_obs as R  # noqa: E402

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from omnigibson.envs import LeRobotPlaybackWrapper  # noqa: E402
from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES  # noqa: E402
from omnigibson.eval.utils.light_utils import LightToggleSynchronizer  # noqa: E402
from omnigibson import object_states as OS  # noqa: E402

# Schema-drift shim (recordings pre-date controller_groups).
from omnigibson.robots.robot import Robot as _Robot  # noqa: E402

_orig_load = _Robot._load_state


def _shimmed_load(self, state):
    if "controller_groups" not in state:
        state["controller_groups"] = {}
    return _orig_load(self, state)


_Robot._load_state = _shimmed_load

EVAL_EVERY_FRAMES = 5      # video frames between predicate evals (= cache stride, 6 Hz)
MAX_INST = 10              # instance cap per literal side (loading_the_car guard)

# taxonomy predicate -> (object_states class, negate, binary)
STATE_MAP = {
    "ontop": ("OnTop", False, True),
    "inside": ("Inside", False, True),
    "nextto": ("NextTo", False, True),
    "under": ("Under", False, True),
    "touching": ("Touching", False, True),
    "attached": ("AttachedTo", False, True),
    "draped": ("Draped", False, True),
    "overlaid": ("Overlaid", False, True),
    "open": ("Open", False, False),
    "closed": ("Open", True, False),
    "toggled_on": ("ToggledOn", False, False),
    "toggled_off": ("ToggledOn", True, False),
    "folded": ("Folded", False, False),
    "unfolded": ("Unfolded", False, False),
    "cooked": ("Cooked", False, False),
    "frozen": ("Frozen", False, False),
    "hot": ("Heated", False, False),
    "on_fire": ("OnFire", False, False),
    "covered": ("Covered", False, True),   # reference is a particle system
    "filled": ("Filled", False, True),
    "saturated": ("Saturated", False, True),
}


def demo_id_from_annot(annot_path):
    """annotations/task-XXXX/episode_XXXXXXXX.json -> (task_id, demo_id)."""
    m = re.search(r"task-(\d{4})/episode_(\d+)\.json", annot_path)
    assert m, f"cannot parse demo id from {annot_path}"
    return int(m.group(1)), int(m.group(2))


def bind_instances(scene, cats):
    """Scene objects whose category matches any cat (pilot substring convention)."""
    out = [o for o in scene.objects
           if any(c and c in (getattr(o, "category", "") or "") for c in cats)]
    return sorted(out, key=lambda o: o.name)[:MAX_INST]


def state_value(obj, state_name, other=None):
    """None = not evaluable for this object (missing state / API mismatch)."""
    cls = getattr(OS, state_name, None)
    if cls is None or cls not in obj.states:
        return None
    try:
        return bool(obj.states[cls].get_value(other) if other is not None
                    else obj.states[cls].get_value())
    except Exception:
        return None


def obj_aabb(o):
    """(center[3], half_extent[3]) world AABB as numpy, or None if unavailable."""
    try:
        c = o.aabb_center
        c = c.numpy() if hasattr(c, "numpy") else np.asarray(c)
        e = o.aabb_extent
        e = e.numpy() if hasattr(e, "numpy") else np.asarray(e)
        return np.asarray(c, float), np.asarray(e, float) / 2.0
    except Exception:
        try:
            lo, hi = o.aabb
            lo = lo.numpy() if hasattr(lo, "numpy") else np.asarray(lo)
            hi = hi.numpy() if hasattr(hi, "numpy") else np.asarray(hi)
            return (lo + hi) / 2.0, (hi - lo) / 2.0
        except Exception:
            return None


class LiteralEvaluator:
    """One goal literal bound to scene instances; sat() per call."""

    def __init__(self, spec, scene):
        self.spec = spec
        self.pred = spec["predicate"]
        # v4 fix: kinematic relations (ontop/inside/nextto/under/touching) via
        # AABB geometry -- object_states returns all-zero for these under
        # playback (they lean on contact data we don't populate). Logical/joint
        # states (toggled_on/open/cooked/...) keep the object_states path, which
        # works (radio ledger F1 0.95).
        self.kinematic = aabb_predicates.is_kinematic(self.pred)
        self.state_name, self.negate, self.binary = STATE_MAP.get(
            self.pred, (None, False, True))
        self.tgts = bind_instances(scene, spec.get("target_cats") or [])
        self.refs = (bind_instances(scene, spec.get("reference_cats") or [])
                     if (self.binary or self.kinematic) else [])
        if self.kinematic:
            self.evaluable = bool(self.tgts and self.refs)
        else:
            # particle-system references (covered dust / filled water) are not
            # scene objects -- resolve them through the target's state args
            self.evaluable = bool(self.state_name and self.tgts
                                  and (not self.binary or self.refs
                                       or self.pred in ("covered", "filled",
                                                        "saturated")))

    def _one(self, t):
        if self.kinematic:
            ta = obj_aabb(t)
            if ta is None:
                return None
            vals = []
            for r in self.refs:
                ra = obj_aabb(r)
                if ra is not None:
                    vals.append(aabb_predicates.evaluate(
                        self.pred, ta[0], ta[1], ra[0], ra[1]))
            v = (any(vals) if vals else None)
        elif not self.binary:
            v = state_value(t, self.state_name)
        elif self.refs:
            vals = [state_value(t, self.state_name, r) for r in self.refs]
            vals = [v for v in vals if v is not None]
            v = any(vals) if vals else None       # any reference instance
        else:
            v = None                               # particle systems: below
        if v is None:
            return None
        return (not v) if self.negate else v

    def sat(self):
        """1/0, or -1 if not evaluable this frame."""
        if not self.evaluable:
            return -1
        vals = [self._one(t) for t in self.tgts]
        vals = [v for v in vals if v is not None]
        if not vals:
            return -1
        q = self.spec.get("quantifier", "")
        if q in ("forall", "forpairs"):
            return int(all(vals))
        if q == "forn":
            return int(sum(vals) >= max(1, self.spec.get("forn_n", 1)))
        return int(any(vals))                      # exists / plain / or-branch


def run_episode(task, file_idx, task_id, demo_id, lits_spec, args):
    out_path = os.path.join(args.out_dir, task, f"ep{file_idx:03d}.json")
    lab_path = os.path.join(args.labels_dir, task, f"labels_{file_idx:03d}.jsonl")
    if os.path.exists(out_path + ".done"):
        print(f"SKIP {task}/ep{file_idx:03d}: done", flush=True)
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    os.makedirs(os.path.dirname(lab_path), exist_ok=True)

    input_path = os.path.join(args.rawdata_root,
                              f"task-{task_id:04d}/episode_{demo_id:08d}.hdf5")
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    gm.ENABLE_TRANSITION_RULES = False
    task_name = R._get_task_name_from_task_id(task_id)
    assert task_name == task, f"manifest task {task} != rawdata task {task_name}"
    available_tasks = R._load_challenge_available_tasks()
    scene_model = available_tasks[task_name][0]["scene_model"]
    full_scene_file = R._find_full_scene_file(task_name=task_name, scene_model=scene_model)
    load_room_instances = R._load_room_instances(task_name=task_name)

    # Isaac boot mutex (replay_labeled convention: concurrent boots corrupt caches)
    import fcntl
    boot_lock = open("/root/isaac_boot.lock", "w")
    fcntl.flock(boot_lock, fcntl.LOCK_EX)
    print("BOOT_LOCK_ACQUIRED", flush=True)

    scratch = os.path.join(args.scratch, f"{task}_{file_idx:03d}")
    env = LeRobotPlaybackWrapper.create_from_hdf5(
        input_path=input_path,
        full_scene_file=full_scene_file,
        load_room_instances=load_room_instances,
        robot_sensor_config={"VisionSensor": {"sensor_kwargs": {
            "image_height": 128, "image_width": 128}}},   # minimal render load
        n_render_iterations=(0 if args.no_render else 1),
        flush_every_n_steps=100000,
        flush_every_n_traj=1,
        include_robot_control=False,
        robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        robot_obs_modalities=["proprio"],                 # no video recording
        include_contacts=False,
        output_path=f"scratch/{task}",
        root_dir=scratch,
        overwrite=True,
        robot_type="R1Pro",
        task_name=task_name,
        include_task_obs=False,
    )
    env.load_observation_space()
    fcntl.flock(boot_lock, fcntl.LOCK_UN)
    print("BOOT_LOCK_RELEASED", flush=True)

    if args.no_render:
        # Predicate eval reads physics + joint state, never pixels, so rendering
        # is pure waste. playback_episode calls og.sim.render() every step when
        # a post_state_update_callback is set (data_wrapper ~L747), on top of
        # the render inside env.step(); killing both is ~5-10x on long episodes.
        # render_on_step(False) routes env.step() through the physics-only branch
        # (simulator ~L1388); the no-op render() neutralizes the explicit calls.
        og.sim.render = lambda *a, **k: None
        print("NO_RENDER enabled", flush=True)

    scene = env.scene
    robot = scene.robots[0]

    evals = [LiteralEvaluator(s, scene) for s in lits_spec["literals"]]
    for j, ev in enumerate(evals):
        print(f"LIT {j} {ev.spec['predicate']}({ev.spec['target']}"
              f"{',' + ev.spec['reference'] if ev.spec.get('reference') else ''}) "
              f"tgts={len(ev.tgts)} refs={len(ev.refs)} evaluable={ev.evaluable} "
              f"method={'aabb' if ev.kinematic else 'object_states'}",
              flush=True)

    # label objects for the extractor pass: union of literal instances,
    # deduped, capped (pilot convention)
    label_objs = []
    seen = set()
    for ev in evals:
        for o in ev.tgts + ev.refs:
            if o.name not in seen:
                seen.add(o.name)
                label_objs.append(o)
    label_objs = label_objs[:args.max_objs]
    print("LABEL_OBJS", len(label_objs), [o.name for o in label_objs], flush=True)

    # camera machinery (pilot-validated: USD prim for static offset, physics
    # link for per-frame pose; stage transforms are stale during playback)
    cam_prim = None
    for prim in og.sim.stage.Traverse():
        if prim.GetTypeName() == "Camera" and "zed" in str(prim.GetPath()).lower():
            cam_prim = prim
            break
    assert cam_prim is not None, "no zed Camera prim found on stage"
    zed_link = next((l for n, l in robot.links.items() if "zed" in n.lower()), None)
    assert zed_link is not None, f"no zed link among {list(robot.links)[:20]}"
    from pxr import UsdGeom
    T_local = np.array(UsdGeom.Xformable(cam_prim).GetLocalTransformation(),
                       dtype=np.float64).reshape(4, 4).T

    def quat_to_rot(q):
        x, y, z, w = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def world_pos(o):
        try:
            p = o.aabb_center
        except Exception:
            p = o.get_position_orientation()[0]
        return p.numpy() if hasattr(p, "numpy") else np.asarray(p)

    def half_extent(o):
        try:
            e = o.aabb_extent
            e = e.numpy() if hasattr(e, "numpy") else np.asarray(e)
            return e / 2.0
        except Exception:
            try:
                lo, hi = o.aabb
                lo = lo.numpy() if hasattr(lo, "numpy") else np.asarray(lo)
                hi = hi.numpy() if hasattr(hi, "numpy") else np.asarray(hi)
                return (hi - lo) / 2.0
            except Exception:
                return np.zeros(3)

    fh = open(lab_path, "w", buffering=1)
    ct = {"i": 0}
    frames, sat_rows = [], []
    eval_every_records = 2 * EVAL_EVERY_FRAMES     # 2 records per video frame

    def post_cb():
        i = ct["i"]
        # extractor labels: every record (2x/frame convention preserved)
        lp, lq = zed_link.get_position_orientation()
        lp = lp.numpy() if hasattr(lp, "numpy") else np.asarray(lp)
        lq = lq.numpy() if hasattr(lq, "numpy") else np.asarray(lq)
        T_link = np.eye(4)
        T_link[:3, :3] = quat_to_rot(lq)
        T_link[:3, 3] = lp
        T_cam = T_link @ T_local
        rec = {"frame": i, "M": [round(float(x), 6) for x in T_cam.flatten()]}
        rec["objs"] = {o.name: [round(float(x), 5) for x in world_pos(o)]
                       for o in label_objs}
        rec["ext"] = {o.name: [round(float(x), 5) for x in half_extent(o)]
                      for o in label_objs}
        fh.write(json.dumps(rec) + "\n")
        # predicate eval at the cache cadence
        if i % eval_every_records == 0:
            frames.append(i // 2)
            sat_rows.append([ev.sat() for ev in evals])
        ct["i"] += 1

    callbacks = [post_cb]
    if task_name in R.LIGHT_REPLAY_TASKS:
        callbacks.insert(0, LightToggleSynchronizer(env.scene).sync_from_current_state)
        print("LIGHT_SYNC enabled", flush=True)
    cb = callbacks[0] if len(callbacks) == 1 else (lambda: [c() for c in callbacks])

    demo_ids = sorted(int(k.split("_", 1)[1]) for k in env.input_hdf5["data"]
                      if k.startswith("demo_"))
    episode_id = demo_ids[-1]
    print(f"REPLAYING demo_{episode_id}", flush=True)
    render_ctx = (og.sim.render_on_step(False) if args.no_render
                  else contextlib.nullcontext())
    with render_ctx:
        try:
            env.playback_episode(episode_id=episode_id, record_data=False,
                                 post_state_update_callback=cb)
        except TypeError:
            # older wrapper signature: recording not optional -- scratch absorbs it
            env.playback_episode(episode_id=episode_id, record_data=True,
                                 post_state_update_callback=cb)
    fh.close()

    evaluable = [bool(ev.evaluable) and any(r[j] >= 0 for r in sat_rows)
                 for j, ev in enumerate(evals)]
    out = dict(task=task, file_idx=file_idx, task_id=task_id, demo_id=demo_id,
               eval_every_frames=EVAL_EVERY_FRAMES,
               n_records=ct["i"], n_video_est=ct["i"] // 2,
               literals=lits_spec["literals"], evaluable=evaluable,
               frames=frames, sat=sat_rows)
    with open(out_path, "w") as f:
        json.dump(out, f)
    open(out_path + ".done", "w").close()
    n_ev = sum(evaluable)
    print(f"PRED_DONE {task}/ep{file_idx:03d} {len(frames)} eval frames, "
          f"{n_ev}/{len(evals)} literals evaluable -> {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--rawdata_root", default="/root/rawdata")
    ap.add_argument("--task_literals", default="/root/task_literals.json")
    ap.add_argument("--out_dir", default="/root/predicates")
    ap.add_argument("--labels_dir", default="/root/sweep_labels_v4")
    ap.add_argument("--scratch", default="/root/pred_scratch")
    ap.add_argument("--max_objs", type=int, default=20)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to task/file_idx entries, e.g. turning_on_radio/3")
    ap.add_argument("--render", dest="no_render", action="store_false",
                    help="render every step (old, slow path); default is no-render")
    ap.add_argument("--no_render", dest="no_render", action="store_true", default=True,
                    help="physics-only playback (default): ~5-10x faster, pixels unused")
    args = ap.parse_args()

    lits_all = json.load(open(args.task_literals))
    eps = json.load(open(args.manifest))
    only = set(args.only) if args.only else None
    for e in eps:
        key = f"{e['task']}/{e['file_idx']}"
        if only and key not in only:
            continue
        task_id, demo_id = demo_id_from_annot(e["annot"])
        run_episode(e["task"], e["file_idx"], task_id, demo_id,
                    lits_all[e["task"]], args)

    print("ALL_PRED_DONE", flush=True)
    og.shutdown()   # teardown segfault after this line is benign


if __name__ == "__main__":
    main()
