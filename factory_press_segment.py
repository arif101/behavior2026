"""Factory press-segment variant: weld LATE (press0 - t0back, where the demo radio has already drifted to its press attitude), carry the short remainder, replay the demo press. Gate: ToggledOn AND weld intact. Closes the 38-deg seam between transport and press segments.

Recipe (validated on d30, rounds 17-25): restore closure+6 -> streak-gated
verified weld (contact ∧ inhand continuous 296+ of 320 steps, 4-step gap
tolerance) -> closed-loop replay of the demo's own actions through transport
(to press0-60 when press entries exist, else closure+280) -> gate on
lifted ∧ weld intact. Exports the executed command stream + meta; the run is
deterministic, so (demo, t0, cmds) reproduces the clip for re-rendering.

Run (one process per demo; og.clear() is forbidden):
  OG_PLAYBACK_REAL_FREQS=1 PYTHONPATH=/root OMNIGIBSON_HEADLESS=1 \
      python -u /root/factory_grasp_transport.py --demo 20
"""
import argparse
import json
import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

OUT = "/root/factory_clips_press"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, required=True)
    ap.add_argument("--tol", type=float, default=0.006)
    ap.add_argument("--settle", type=int, default=40)
    ap.add_argument("--t0back", type=int, default=100)
    a = ap.parse_args()

    import h5py
    import torch as th
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True; gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import _np, P, A_TORSO

    os.makedirs(OUT, exist_ok=True)
    man = json.load(open(f"/root/snapshot_bank_v21/manifest_d{a.demo}.json"))
    G = next(r for r in man["entries"] if str(r["stage"]) == "G")
    closure = G["grasp_closure_frame"]
    lift_z = G["lift_z"]
    press = [int(r["frame"]) for r in man["entries"]
             if r.get("family") == "press"]
    if not press:
        print(f"RESULT d{a.demo} NO_PRESS_ENTRIES", flush=True)
        os._exit(0)
    press0 = max(press)
    horizon_end = press0 + 40

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5",
        output_path=f"/root/fct_tmp_{a.demo}.hdf5",
        robot_obs_modalities=("proprio",), robot_proprio_keys=EVAL_PROPRIO_KEYS)
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]
    rob = wrapper.scene.robots[0]

    with h5py.File(f"/root/rawdemos/task-0000/episode_{a.demo:08d}.hdf5", "r") as f:
        key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
        acts = f[f"data/{key}/action"][:]
    horizon_end = min(horizon_end, len(acts) - 1)

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

    def links():
        return [_np(l.get_position_orientation()[0])
                for l in rob.finger_links["right"]]

    def contact():
        try:
            cs, _ = rob._find_gripper_contacts(arm="right")
            return any(radio.name in c for c in cs)
        except Exception:  # noqa: BLE001
            return False

    def inhand():
        try:
            return rob._calculate_in_hand_object(arm="right") is not None
        except Exception:  # noqa: BLE001
            return False

    def native_ag():
        return rob._ag_obj_constraint_params.get("right") is not None

    cmds_log = []

    def step_cmd(cmd):
        q = q61()
        for _ in range(a.settle):
            wrapper.env.step(cmd)
            cmds_log.append(np.asarray(cmd, np.float32).copy())
            q = q61()
            if (np.abs(q[P["left"]["arm_qpos"]] - cmd[7:14]).max() < a.tol
                    and np.abs(q[P["right"]["arm_qpos"]] - cmd[15:22]).max() < a.tol
                    and np.abs(q[P["trunk_qpos"]] - cmd[A_TORSO]).max() < a.tol):
                break

    t0 = press0 - a.t0back
    restore_to_frame(wrapper, 0, t0)
    rob._refresh_rigid_contact_view()
    q = q61()
    qL0, qR0 = q[P["left"]["arm_qpos"]].copy(), q[P["right"]["arm_qpos"]].copy()
    tr0 = q[P["trunk_qpos"]].copy()
    green, gap, weld_k = 0, 0, None
    for k in range(400):
        cmd = np.asarray(acts[t0], np.float32).copy()
        cmd[0:3] = 0.0
        cmd[7:14] = qL0
        cmd[15:22] = qR0
        cmd[A_TORSO] = tr0
        cmd[22] = -1.0
        wrapper.env.step(cmd)
        cmds_log.append(cmd.copy())
        if contact() and inhand():
            green += 1
            gap = 0
        else:
            gap += 1
            if gap > 4:
                green = 0
        if native_ag():
            weld_k = k
            print(f"NATIVE_AG k={k}", flush=True)
            break
        if k >= 320 and green >= 296:
            blname = radio.root_link_name
            cp = th.as_tensor(np.mean(links(), axis=0), dtype=th.float32)
            rob._establish_grasp(radio, blname, "right", cp, "FixedJoint")
            weld_k = k
            print(f"VERIFIED_WELD k={k} streak={green}", flush=True)
            break
    if not native_ag():
        print(f"RESULT d{a.demo} WELD_FAILED", flush=True)
        os._exit(0)

    from omnigibson.object_states import ToggledOn as _TO

    def toggled():
        try:
            return bool(radio.states[_TO].get_value())
        except Exception:  # noqa: BLE001
            return False

    lifted, ag_lost, tgl = False, None, None
    for t in range(t0 + 1, horizon_end):
        cmd = np.asarray(acts[t], np.float32).copy()
        cmd[22] = -1.0
        step_cmd(cmd)
        tp = _np(radio.get_position_orientation()[0])
        lifted = lifted or tp[2] > lift_z + 0.05
        if not native_ag() and ag_lost is None:
            ag_lost = t
            print(f"AG_LOST t={t}", flush=True)
            break
        if tgl is None and toggled():
            tgl = t
            print(f"TOGGLED t={t} (press0 {press0})", flush=True)
        if t % 40 == 0:
            print(f"CH t={t} z={tp[2]:.3f} tg={tgl is not None}", flush=True)
        if tgl is not None and t > tgl + 15:
            break
    ok = bool(native_ag() and ag_lost is None and tgl is not None)
    meta = dict(demo=a.demo, t0=t0, closure=closure, press0=press0,
                toggled_frame=tgl, weld_k=weld_k,
                streak=green, horizon_end=horizon_end, lifted=bool(lifted),
                ag_intact=bool(native_ag()), ag_lost=ag_lost, ok=ok,
                n_cmds=len(cmds_log),
                radio_z_end=round(float(
                    _np(radio.get_position_orientation()[0])[2]), 4))
    if ok:
        np.savez_compressed(f"{OUT}/d{a.demo:03d}_press_segment.npz",
                            cmds=np.stack(cmds_log),
                            meta=json.dumps(meta))
    json.dump(meta, open(f"{OUT}/d{a.demo:03d}_meta.json", "w"), indent=1)
    print("RESULT", json.dumps(meta), flush=True)
    os._exit(0)


main()
