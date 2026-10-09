"""REVERSE-CURRICULUM collector (RFCL-adapted, pilot v1): start episodes AT the hard part.

Uses HDF5PlaybackWrapper to build the env from a demo and STATE-PLAYBACK to a frame near the
demonstrator's grasp-descent initiation, then HANDS CONTROL TO THE POLICY (websocket, same
serving stack) for a short horizon. Scores by press/descent progress. Successful segments are
saved as training clips; the start-frame offset widens as the near-goal success rate rises
(the curriculum). Evidence base: RFCL (ICLR24) — reverse curriculum from demo-state resets was
the only method to crack precision-contact tasks; our 0.5mm playback makes stage-1 ~free.

Pilot: N demos × K attempts from OFFSET steps before each demo's stage-1→2 transition
(initiation frames from the stage labels). Horizon 400 steps. Outputs:
  /root/rc_clips/rc_{demo}_{attempt}.npz  (proprio+actions, kept only on progress/success)
  /root/rc_stats.json                     (per-offset success rates — the curriculum signal)

Run INSIDE the behavior env with the policy server up:
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 python reverse_curriculum_collect.py --demos 3 --attempts 4
"""

import argparse
import glob
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

# Exact list from omnigibson/eval/r1pro.yaml — the 61-d layout the serving RobotConfig indexes
# (playback default is a 48-d subset -> server IndexError at trunk_qpos 53:57).
EVAL_PROPRIO_KEYS = [
    "base_qvel", "arm_left_qpos", "arm_left_qvel", "eef_left_pos", "eef_left_quat",
    "gripper_left_qpos", "gripper_left_qvel", "arm_right_qpos", "arm_right_qvel",
    "eef_right_pos", "eef_right_quat", "gripper_right_qpos", "gripper_right_qvel",
    "trunk_qpos", "trunk_qvel",
]


def initiation_frame(demo):
    """First ACQUIRE->MANIPULATE transition from the banked stage labels (parquet columns
    were derived from these same arrays)."""
    # proxy: first frame where metalink height rises 3cm above start (lift = manip start)
    mw = np.load(f"/root/metalink_labels/ep{demo}.npz")["meta_world"]
    lifted = np.where(mw[:, 2] - mw[0, 2] > 0.03)[0]
    return int(lifted[0]) if len(lifted) else int(len(mw) * 0.6)


def restore_to_frame(wrapper, episode_id, t):
    """Jump the sim to recorded frame t without stepping physics. playback_episode is pure
    state playback (og.sim.load_state per frame), so we mirror its init sequence verbatim
    (scene restore -> init_metadata -> play -> reset -> schema align), then load state[0] plus every
    awake-object frame up to t, then state[t] (see the 2026-09-14 note below). include_robot_control=True (default) keeps controllers live for the
    policy handoff."""
    import omnigibson as og
    from omnigibson.envs.data_wrapper import _align_scene_object_states_with_recorded_schema
    from omnigibson.utils.python_utils import h5py_group_to_torch
    import omnigibson.robots.robot as _rb

    # SCHEMA-DRIFT SHIM: challenge demos' scene_file predates the controller_groups block in
    # Robot._load_state; absent -> KeyError. Inject empty dict — the per-controller loop skips
    # None entries, and the serialized per-frame load right after restores true state anyway.
    if not getattr(_rb.Robot._load_state, "_cg_shim", False):
        _orig_ls = _rb.Robot._load_state

        def _tolerant_ls(self, state):
            if "controller_groups" not in state:
                state = dict(state)
                state["controller_groups"] = {}
            return _orig_ls(self, state)

        _tolerant_ls._cg_shim = True
        _rb.Robot._load_state = _tolerant_ls

    tg = h5py_group_to_torch(wrapper.input_hdf5["data"][f"demo_{episode_id}"])
    state, state_size = tg["state"], tg["state_size"]
    t = min(int(t), len(state) - 1)

    wrapper.scene.restore(wrapper.scene_file, update_initial_file=True)
    og.sim.stop()
    for i, obj in enumerate(wrapper.scene.objects):
        for attr, vals in tg["init_metadata"].items():
            val = vals[i]
            setattr(obj, attr, val.item() if val.ndim == 0 else val)
    og.sim.play()
    wrapper.reset()
    _align_scene_object_states_with_recorded_schema(
        scene=wrapper.scene, recorded_scene_file=wrapper.recorded_scene_file
    )
    # 2026-09-14 FIX (template-pose bug): the recorder stores only AWAKE objects per frame (robot +
    # whatever is moving), so a direct load of state[t] left every asleep object -- the radio before the
    # human touches it -- at the SCENE-FILE pose, i.e. the task template's fixed pose, not this demo's
    # sampled pose (5 cm .. 1.1 m and up to 178 deg of yaw away across the 38 factory demos; the approach
    # factory's "pull" matched that template-vs-sampled offset to 4 mm on 37/38). Replay the full frame-0
    # state and then every frame <= t that carries more than the robot-only baseline (an awake object),
    # so each object ends at its last recorded pose before t; state[t] last so the robot is exact.
    n_total = int(state.shape[0])
    sizes_all = [int(state_size[k]) for k in range(n_total)]
    base = min(sizes_all[1:]) if n_total > 1 else sizes_all[0]
    og.sim.load_state(state[0, : sizes_all[0]], serialized=True)
    replayed = [k for k in range(1, t + 1) if sizes_all[k] > base]
    for k in replayed:
        og.sim.load_state(state[k, : sizes_all[k]], serialized=True)
    og.sim.load_state(state[t, : sizes_all[t]], serialized=True)
    restore_to_frame.last_replayed = len(replayed)
    og.sim.render()  # refresh sensors from restored state without a physics step
    return t


def _np(v):
    return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)


def _find_obs(obs, substrs):
    """Walk nested-or-flattened obs dicts; return first leaf whose joined '::' key path
    contains ALL substrs (handles both {'robot_r1': {'proprio': ..}} and 'robot_r1::proprio')."""
    hits = []

    def walk(prefix, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}::{k}" if prefix else str(k), v)
        else:
            hits.append((prefix, node))

    walk("", obs if isinstance(obs, dict) else {})
    for key, v in hits:
        if all(s in key for s in substrs):
            return key, v
    return None, None


def _head_rgb(obs):
    """Best-effort head-camera frame from the raw OG obs dict (zed = head on R1Pro)."""
    _, v = _find_obs(obs, ("zed", "rgb"))
    return _np(v)[..., :3].astype(np.uint8) if v is not None else None


def _enc_jpg(arr):
    from io import BytesIO

    from PIL import Image
    b = BytesIO()
    Image.fromarray(arr).save(b, "JPEG", quality=92)
    return b.getvalue()


def _enc_png16(depth_m):
    """Depth meters -> uint16 mm PNG bytes (matches the 12-bit-mm training video convention)."""
    from io import BytesIO

    from PIL import Image
    mm = np.clip(_np(depth_m) * 1000.0, 0, 65535).astype(np.uint16)
    b = BytesIO()
    Image.fromarray(mm, mode="I;16").save(b, "PNG")
    return b.getvalue()


def _cam_capture(obs):
    """{short: (rgb_uint8, depth_float_or_None)} for head/left/right wrist."""
    out = {}
    for short, tag in (("head", "zed"), ("left", "left_realsense"), ("right", "right_realsense")):
        _, rgb = _find_obs(obs, (tag, "rgb"))
        _, dep = _find_obs(obs, (tag, "depth_linear"))
        if rgb is not None:
            out[short] = (_np(rgb)[..., :3].astype(np.uint8), dep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=3)
    ap.add_argument("--demo-id", type=int, default=None,
                    help="run ONE raw demo id in this process (Isaac env-reuse leaks; chunk like LIBERO)")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--offset", type=int, default=150, help="steps BEFORE initiation to start")
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--start-frame", type=int, default=None,
                    help="absolute restore frame (overrides offset; e.g. post-press control)")
    ap.add_argument("--play-demo-actions", action="store_true",
                    help="CONTROL: replay the demo's own recorded actions from the restored "
                         "frame instead of policy actions (handoff-fidelity test; controllers "
                         "are absolute-position so faithful restore => demo completes)")
    ap.add_argument("--settle", type=int, default=0,
                    help="hold the first demo action N steps post-restore before the splice "
                         "(lets the cold contact solver stabilize; rescue for marginal grasps)")
    ap.add_argument("--capture-train", action="store_true",
                    default=os.environ.get("RC_CAPTURE") == "1",
                    help="record EVERY frame: 3-cam RGB (JPEG q92) + depth (uint16-mm PNG), "
                         "paired obs_t->action_t — the training-format capture")
    ap.add_argument("--perturb", type=float, default=0.0,
                    help="uniform arm-joint jitter (rad) applied post-restore; with "
                         "--play-demo-actions + outcome filter this is the DIVERSITY generator "
                         "(splice clips from perturbed neighborhoods, not points)")
    ap.add_argument("--dump-obs", action="store_true",
                    default=os.environ.get("RC_DUMP_OBS") == "1",
                    help="save head-cam frames every 15 steps into the clip npz (sanity diag)")
    a = ap.parse_args()

    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from omnigibson.eval.utils.network_utils import WebsocketClientPolicy

    if a.demo_id is not None:
        demos = [a.demo_id]
    else:
        emap = json.load(open("/root/episode_map.json"))
        demos = sorted(int(d) for d in emap["mapping"].values())[: a.demos]
    os.makedirs("/root/rc_clips", exist_ok=True)
    stats = {"offset": a.offset, "runs": []}

    for demo in demos:
        t_init = initiation_frame(demo)
        t_start = a.start_frame if a.start_frame is not None else max(0, t_init - a.offset)
        h5 = f"/root/rawdemos/task-0000/episode_{demo:08d}.hdf5"
        print(f"demo {demo}: initiation ~f{t_init}, starting at f{t_start}", flush=True)

        wrapper = HDF5PlaybackWrapper.create_from_hdf5(
            input_path=h5, output_path=f"/root/rc_tmp_{demo}.hdf5",
            robot_obs_modalities=("rgb", "depth_linear", "proprio"),
            robot_proprio_keys=EVAL_PROPRIO_KEYS,
        )
        env = wrapper.env
        # task-agnostic tracked objects for capture labeling (conversion derives target_points
        # from THESE poses — the clip's own world, not the demo timeline, which diverges)
        _subs = os.environ.get("RC_TRACK_OBJECTS", "radio").lower().split(",")
        track_objs = [o for o in wrapper.scene.objects
                      if any(s in o.name.lower() for s in _subs)]
        print(f"  tracking objects: {[o.name for o in track_objs]}", flush=True)
        # the raw challenge files key episodes as demo_<id>, not demo_0 — detect it
        epid = sorted(int(k.split("_")[1]) for k in wrapper.input_hdf5["data"].keys()
                      if k.startswith("demo_"))[0]
        t_start = restore_to_frame(wrapper, epid, t_start)
        print(f"  state-restored to f{t_start} (demo_{epid})", flush=True)

        def _apply_perturb():
            if a.perturb <= 0:
                return
            import omnigibson as og
            robot = wrapper.scene.robots[0]
            q = robot.get_joint_positions()
            q = q.clone() if hasattr(q, "clone") else np.array(q)
            arm_idx = [i for i, nm in enumerate(robot.dof_names_ordered)
                       if "arm" in nm.lower()]
            for i in arm_idx:
                q[i] = q[i] + float(np.random.uniform(-a.perturb, a.perturb))
            robot.set_joint_positions(q)
            og.sim.render()

        _apply_perturb()

        if a.play_demo_actions:
            import h5py
            with h5py.File(h5, "r") as f:
                demo_actions = f[f"data/demo_{epid}/action"][:]
            policy = None
        else:
            policy = WebsocketClientPolicy(host="127.0.0.1", port=8000)
        for att in range(a.attempts):
            traj, prop, frames = [], [], []
            cap = {}
            success = False
            if a.settle > 0 and policy is None:
                for _ in range(a.settle):
                    env.step(demo_actions[min(t_start, len(demo_actions) - 1)])
                print(f"  settled {a.settle} steps", flush=True)
            obs = env.get_obs()[0] if hasattr(env, "get_obs") else None
            for t in range(a.horizon):
                if policy is None:
                    act = demo_actions[min(t_start + t, len(demo_actions) - 1)]
                else:
                    act = policy.act(obs)
                # capture obs_t (the obs this action was computed from) BEFORE stepping —
                # training rows pair obs_t -> action_t
                pk, pv = _find_obs(obs, ("proprio",))
                if pv is not None:
                    prop.append(_np(pv).reshape(-1))
                if t == 0:
                    print(f"  obs probe: proprio_key={pk}", flush=True)
                if a.capture_train:
                    for short, (rgb, dep) in _cam_capture(obs).items():
                        cap.setdefault(f"{short}_rgb", []).append(_enc_jpg(rgb))
                        if dep is not None:
                            cap.setdefault(f"{short}_depth", []).append(_enc_png16(dep))
                    for o in track_objs:
                        p, q = o.get_position_orientation()
                        cap.setdefault(f"objpose_{o.name}", []).append(
                            np.concatenate([_np(p).reshape(-1), _np(q).reshape(-1)]))
                elif a.dump_obs and t % 15 == 0:
                    fr = _head_rgb(obs)
                    if fr is not None:
                        frames.append(fr)
                traj.append(np.asarray(act).reshape(-1))
                out = env.step(act)
                obs = out[0]
                done = bool(out[2]) if isinstance(out, tuple) and len(out) > 3 else False
                if done:
                    success = True
                    break
            if a.capture_train and not success:
                # outcome filter at the source: failed splices keep only light telemetry
                cap = {k: v for k, v in cap.items() if k.startswith("objpose_")}
            extra = {k: (np.asarray(v, dtype=float) if k.startswith("objpose_")
                         else np.array(v, dtype=object)) for k, v in cap.items()}
            np.savez_compressed(f"/root/rc_clips/rc_{demo}_{att}.npz",
                                actions=np.array(traj), proprio=np.array(prop),
                                frames=np.array(frames) if frames else np.zeros(0),
                                success=success, start_frame=t_start, init_frame=t_init,
                                perturb=a.perturb, settle=a.settle, **extra)
            rec = {"demo": demo, "att": att, "success": success, "steps": len(traj),
                   "offset": a.offset, "start": int(t_start), "init": int(t_init)}
            stats["runs"].append(rec)
            with open("/root/rc_stats.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  attempt {att}: success={success} steps={len(traj)}", flush=True)
            if att < a.attempts - 1:
                try:
                    restore_to_frame(wrapper, epid, t_start)
                    _apply_perturb()
                except Exception as e:
                    print(f"  re-reset failed: {e}", flush=True)
                    break
        og.clear()
    sr = sum(1 for r in stats["runs"] if r["success"]) / max(len(stats["runs"]), 1)
    print(f"RC_PILOT_DONE success_rate={sr:.2f} over {len(stats['runs'])} attempts")


if __name__ == "__main__":
    main()
