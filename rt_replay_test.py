"""Replay test: does NATIVE assisted AG fire when d30's own actions are replayed
through the contact plunge (closure-20 .. closure+22) at eval physics?

Fixes the rt proto's fatal flaw: the proto stopped replay at closure+1, one frame
before the demo's 3-frame plunge to contact (ground-truth film 08-23).

Pass 0 captures the ground-truth fingertip/qpos path via state restore, so each
replay frame reports drift vs the real tape — the number that decides whether
failures are fidelity or physics.

Run (training paused):  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root \
    OMNIGIBSON_HEADLESS=1 python -u rt_replay_test.py --demo 30
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=30)
    a = ap.parse_args()

    import h5py
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np, P, A_TORSO

    man = json.load(open(f"/root/snapshot_bank_v21/manifest_d{a.demo}.json"))
    G = next(r for r in man["entries"] if str(r["stage"]) == "G")
    closure = G["grasp_closure_frame"]
    lift_z = G["lift_z"]
    f0 = closure - 20
    f_end = closure + 22

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path="/root/rtt_tmp.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]

    with h5py.File(f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5", "r") as f:
        key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
        acts = f[f"data/{key}/action"][:]

    def q61():
        obs = wrapper.env.get_obs()[0]
        def find(node, sub):
            if isinstance(node, dict):
                for k, v in node.items():
                    r = find(v, sub)
                    if r is not None:
                        return r
                    if sub in str(k):
                        return v
            return None
        return _np(find(obs, "proprio")).reshape(-1)

    def fingertips():
        return _np(th.stack([l.get_position_orientation()[0]
                             for l in rob.finger_links["right"]]).mean(dim=0))

    def contact():
        try:
            cs, _ = rob._find_gripper_contacts(arm="right")
            return any(radio.name in c for c in cs)
        except Exception:  # noqa: BLE001
            return False

    def native_ag():
        return rob._ag_obj_constraint_params.get("right") is not None

    def step_cmd(cmd, settle=8):
        for _ in range(settle):
            wrapper.env.step(cmd)
            q = q61()
            if (np.abs(q[P["left"]["arm_qpos"]] - cmd[7:14]).max() < 0.03
                    and np.abs(q[P["right"]["arm_qpos"]] - cmd[15:22]).max() < 0.03
                    and np.abs(q[P["trunk_qpos"]] - cmd[A_TORSO]).max() < 0.03):
                break

    # ---- Pass 0: ground truth path via state restore ----
    ft_gt, q_gt = {}, {}
    for t in range(f0, f_end + 1):
        restore_to_frame(wrapper, 0, t)
        ft_gt[t] = fingertips()
        q_gt[t] = q61()[P["right"]["arm_qpos"]].copy()
    print("GT captured", len(ft_gt), "frames", flush=True)

    rng = np.random.default_rng(7)
    summaries = []
    for sig, reps in ((0.0, 2), (0.005, 1)):
        for rep in range(reps):
            restore_to_frame(wrapper, 0, f0)
            if sig > 0:
                rp0, rq0 = radio.get_position_orientation()
                jit = rng.normal(0, sig, 3); jit[2] = 0.0
                radio.set_position_orientation(
                    position=th.as_tensor(_np(rp0) + jit, dtype=th.float32),
                    orientation=rq0)
            ag_frame, drifts, plunge_drifts = None, [], []
            contact_frames = 0
            for t in range(f0, f_end + 1):
                step_cmd(np.asarray(acts[t], np.float32))
                ft = fingertips()
                d = float(np.linalg.norm(ft - ft_gt[t]))
                drifts.append(d)
                if t >= closure:
                    plunge_drifts.append(d)
                c = contact()
                contact_frames += int(c)
                if native_ag() and ag_frame is None:
                    ag_frame = t
                    print(f"AG_FIRED sig={sig} rep={rep} at rel={t - closure}",
                          flush=True)
                print(f"RP sig={sig} rep={rep} rel={t - closure:+d} "
                      f"drift={d:.4f} contact={c} ag={native_ag()}", flush=True)
                if ag_frame is not None:
                    break
            # if AG fired: finish with the demo's own lift, grip held closed
            lifted = still = False
            if ag_frame is not None:
                tp_prev, dwell = None, 0
                for t in range(ag_frame + 1, min(closure + 80, len(acts))):
                    cmd = np.asarray(acts[t], np.float32).copy()
                    cmd[22] = -1.0
                    step_cmd(cmd)
                    tp = _np(radio.get_position_orientation()[0])
                    lifted = lifted or tp[2] > lift_z + 0.05
                    stillnow = (tp_prev is not None and
                                float(np.linalg.norm(tp - tp_prev)) < 0.005)
                    tp_prev = tp.copy()
                    dwell = dwell + 1 if (lifted and stillnow and native_ag()) else 0
                    if dwell >= 15:
                        still = True
                        break
            s = dict(sig=sig, rep=rep,
                     ag_fired=ag_frame is not None,
                     ag_rel=None if ag_frame is None else ag_frame - closure,
                     mean_drift=round(float(np.mean(drifts)), 4),
                     max_drift=round(float(np.max(drifts)), 4),
                     max_drift_plunge=(round(float(np.max(plunge_drifts)), 4)
                                       if plunge_drifts else None),
                     contact_frames=contact_frames,
                     lifted=lifted, still15=still)
            summaries.append(s)
            print("SUMMARY", json.dumps(s), flush=True)
    json.dump(summaries, open("/root/rt_replay_summary.json", "w"), indent=1)
    print("DONE", flush=True)
    os._exit(0)


main()
