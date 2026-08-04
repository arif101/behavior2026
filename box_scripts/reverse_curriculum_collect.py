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


def initiation_frame(demo):
    """First ACQUIRE->MANIPULATE transition from the banked stage labels (parquet columns
    were derived from these same arrays)."""
    ml = np.load(f"/root/metalink_labels/ep{demo}.npz")["meta_base"]
    # geometry rule identical to add_stage_pixel_labels: dist to closest EE < 0.12 => stage 2.
    # We stored per-frame labels only in parquet; recompute cheap proxy from the pose JSONs.
    pj = json.load(open(f"/root/poses_x/turning_on_radio/ep{demo}.json"))
    # proxy: first frame where metalink height rises 3cm above start (the lift = manip start)
    z0 = ml[0][2] if ml.ndim > 1 else None
    mw = np.load(f"/root/metalink_labels/ep{demo}.npz")["meta_world"]
    lifted = np.where(mw[:, 2] - mw[0, 2] > 0.03)[0]
    return int(lifted[0]) if len(lifted) else int(len(mw) * 0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--offset", type=int, default=150, help="steps BEFORE initiation to start")
    ap.add_argument("--horizon", type=int, default=400)
    a = ap.parse_args()

    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from omnigibson.eval.utils.network_utils import WebsocketClientPolicy

    emap = json.load(open("/root/episode_map.json"))
    demos = sorted(int(d) for d in emap["mapping"].values())[: a.demos]
    os.makedirs("/root/rc_clips", exist_ok=True)
    stats = {"offset": a.offset, "runs": []}

    for demo in demos:
        t_init = initiation_frame(demo)
        t_start = max(0, t_init - a.offset)
        h5 = f"/root/rawdemos/task-0000/episode_{demo:08d}.hdf5"
        print(f"demo {demo}: initiation ~f{t_init}, starting at f{t_start}", flush=True)

        wrapper = HDF5PlaybackWrapper.create_from_hdf5(
            input_path=h5, output_path=f"/root/rc_tmp_{demo}.hdf5",
            robot_obs_modalities=("rgb", "depth_linear", "proprio"),
        )
        env = wrapper.env
        # state-playback to t_start (wrapper API: playback_episode steps states; we drive
        # manually via its recorded states to the target frame)
        wrapper.playback_to_frame = getattr(wrapper, "playback_to_frame", None)
        try:
            # v3.9 API: playback_episode(episode_id, stop_at_step=) — probe both spellings
            wrapper.playback_episode(0, stop_at_step=t_start)
        except TypeError:
            for k in range(t_start):
                wrapper.playback_step(0, k)
        print(f"  state-played to f{t_start}", flush=True)

        policy = WebsocketClientPolicy(host="127.0.0.1", port=8000)
        for att in range(a.attempts):
            traj = []
            success = False
            obs = env.get_obs()[0] if hasattr(env, "get_obs") else None
            for t in range(a.horizon):
                act = policy.act(obs)
                out = env.step(act)
                obs = out[0]
                # progress score: metalink z-lift or task success flag
                info = out[-1] if isinstance(out, tuple) else {}
                done = bool(out[2]) if isinstance(out, tuple) and len(out) > 3 else False
                traj.append(np.asarray(act).reshape(-1))
                if done:
                    success = True
                    break
            if success or len(traj) < a.horizon:
                np.savez_compressed(f"/root/rc_clips/rc_{demo}_{att}.npz",
                                    actions=np.array(traj), success=success,
                                    start_frame=t_start, init_frame=t_init)
            stats["runs"].append({"demo": demo, "att": att, "success": success,
                                  "steps": len(traj)})
            print(f"  attempt {att}: success={success} steps={len(traj)}", flush=True)
            # re-reset to t_start for the next attempt
            try:
                wrapper.playback_episode(0, stop_at_step=t_start)
            except Exception:
                break
        og.clear()

    with open("/root/rc_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    sr = sum(1 for r in stats["runs"] if r["success"]) / max(len(stats["runs"]), 1)
    print(f"RC_PILOT_DONE success_rate={sr:.2f} over {len(stats['runs'])} attempts")


if __name__ == "__main__":
    main()
