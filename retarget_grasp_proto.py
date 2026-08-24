"""Retarget + contact-extension prototype (MimicGen-style, minimal).

Question: can EVAL-LEGAL grasps (native assisted-mode trigger: contact + raycast +
sustained closure — NO magnetic establish anywhere) be manufactured geometrically from
demo segments, by extending the demo's hover the last ~3 cm to true contact?

Per trial: playback-restore at closure-20 -> closed-loop replay to closure (estimating
a finite-difference Jacobian from the replay) -> servo fingertips toward the radio until
finger contact -> hold closure -> wait for NATIVE AG -> replay demo lift -> still-hold.

Run (pauses training):  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
    python -u retarget_grasp_proto.py --demo 30 --trials 4
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, default=30)
    ap.add_argument("--trials", type=int, default=4)
    a = ap.parse_args()

    import h5py
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np, P, A_TORSO, A_ARM, A_GRIP

    man = json.load(open(f"/root/snapshot_bank_v21/manifest_d{a.demo}.json"))
    G = next(r for r in man["entries"] if str(r["stage"]) == "G")
    closure = G["grasp_closure_frame"]
    f0 = max(0, closure - 20)
    lift_z = G["lift_z"]

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path="/root/rt_tmp.hdf5",
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

    rng = np.random.default_rng(23)
    results = []
    for sig in (0.0, 0.01):
        for trial in range(a.trials if sig > 0 else max(2, a.trials // 2)):
            restore_to_frame(wrapper, 0, f0)
            if sig > 0:
                rp, rq = radio.get_position_orientation()
                jit = rng.normal(0, sig, 3); jit[2] = 0.0
                radio.set_position_orientation(
                    position=th.as_tensor(_np(rp) + jit, dtype=th.float32),
                    orientation=rq)
            # phase A: replay hover, record (qpos_R, fingertip) pairs for Jacobian
            qs, es = [], []
            for t in range(f0, closure + 2):
                cmd = np.asarray(acts[t], np.float32)
                step_cmd(cmd)
                qs.append(q61()[P["right"]["arm_qpos"]].copy())
                es.append(fingertips())
            # finite-difference Jacobian (3x7) from replay motion
            dq = np.diff(np.stack(qs), axis=0)
            de = np.diff(np.stack(es), axis=0)
            keep = np.linalg.norm(dq, axis=1) > 1e-4
            J = None
            if keep.sum() >= 3:
                J, *_ = np.linalg.lstsq(dq[keep], de[keep], rcond=None)  # dq @ J ~ de
            # phase B: contact extension via ACTIVE system ID — probe each joint,
            # measure fingertip response, build the true local Jacobian, servo down
            hover_gap = float(np.linalg.norm(
                _np(radio.get_position_orientation()[0]) - fingertips()))
            got_contact = contact()
            base_cmd = np.asarray(acts[closure + 1], np.float32).copy()
            Jcols = []
            for j in range(7):
                q = q61()
                e0 = fingertips()
                probe = base_cmd.copy()
                probe[15:22] = q[P["right"]["arm_qpos"]]
                probe[15 + j] += 0.04
                probe[22] = 1.0
                step_cmd(probe, settle=6)
                Jcols.append((fingertips() - e0) / 0.04)
                undo = probe.copy()
                undo[15 + j] -= 0.04
                step_cmd(undo, settle=6)
            J = np.stack(Jcols, axis=1)   # 3x7: de = J @ dq
            for it in range(60):
                if got_contact:
                    break
                gap = _np(radio.get_position_orientation()[0]) - fingertips()
                direction = gap / (np.linalg.norm(gap) + 1e-9)
                dq_step = np.clip(np.linalg.pinv(J) @ (direction * 0.008),
                                  -0.03, 0.03)
                q = q61()
                base_cmd[15:22] = q[P["right"]["arm_qpos"]] + dq_step
                base_cmd[22] = 1.0
                step_cmd(base_cmd, settle=6)
                got_contact = contact()
            # phase C: sustained closure -> NATIVE trigger only
            ag = native_ag()
            if got_contact and not ag:
                q = q61()
                base_cmd[15:22] = q[P["right"]["arm_qpos"]]
                base_cmd[22] = -1.0
                for it in range(40):
                    wrapper.env.step(base_cmd)
                    if native_ag():
                        ag = True
                        break
            # phase D: lift via demo frames, grip held closed
            lifted = still = False
            if ag:
                tp_prev = None; dwell = 0
                for t in range(closure + 2, min(closure + 60, len(acts))):
                    cmd = np.asarray(acts[t], np.float32).copy()
                    cmd[22] = -1.0
                    step_cmd(cmd)
                    tp = _np(radio.get_position_orientation()[0])
                    lifted = lifted or tp[2] > lift_z + 0.05
                    stillnow = tp_prev is not None and \
                        float(np.linalg.norm(tp - tp_prev)) < 0.005
                    tp_prev = tp.copy()
                    dwell = dwell + 1 if (lifted and stillnow and native_ag()) else 0
                    if dwell >= 15:
                        still = True
                        break
            ok = bool(ag and lifted and still)
            results.append((sig, ok))
            print(f"RT sig={sig:.2f} trial{trial}: hover_gap="
                  f"{hover_gap if hover_gap is None else round(hover_gap,3)} "
                  f"contact={got_contact} NATIVE_AG={ag} lifted={lifted} "
                  f"still15={still} -> {'SUCCESS' if ok else 'fail'}", flush=True)
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for sig, ok in results:
        agg[sig][1] += 1; agg[sig][0] += ok
    for sig in sorted(agg):
        s, n = agg[sig]
        print(f"RT_SUMMARY sigma={sig:.2f}: {s}/{n}", flush=True)
    os._exit(0)


main()
