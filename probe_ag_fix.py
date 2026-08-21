"""AG-FIX PROBE: can we re-establish the assisted-grasp constraint after a state restore?

Restores demo 10 @ f1300 (radio held aloft in the closed left gripper on the demo timeline),
then:
  1. DIAGNOSIS: does the auto-AG path see the object? (_find_gripper_contacts,
     _calculate_in_hand_object, applying-grasp internals) + 10 replay steps to reproduce
     the no-engagement failure.
  2. FIX: call robot._establish_grasp(...) directly (documented as externally callable),
     contact position from _find_finger_contact_position or fingertip midpoint fallback.
  3. VERIFY: 120 held-action steps; PASS iff AG engaged AND |radio z drift| < 1 mm over the
     first 30 settle steps (the [A-2026-08-13] spec validity criterion).

Run: PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 python -u probe_ag_fix.py
"""

import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

DEMO = 10
T_START = 1300

from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame  # noqa: E402


def _np(v):
    return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)


def main():
    import h5py
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper

    h5 = f"/root/rawdemos/task-0000/episode_{DEMO:08d}.hdf5"
    with h5py.File(h5, "r") as f:
        demo_actions = f["data/demo_0/action"][:]

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=h5, output_path="/root/ag_probe_tmp.hdf5",
        robot_obs_modalities=("rgb", "proprio"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS,
    )
    env = wrapper.env
    robot = wrapper.scene.robots[0]
    radio = [o for o in wrapper.scene.objects if "radio" in o.name.lower()][0]

    restore_to_frame(wrapper, 0, T_START)
    print("AGFIX arms:", robot.arm_names, "grasping_mode:", robot.grasping_mode, flush=True)
    arm = next((a for a in robot.arm_names if "left" in a), robot.arm_names[0])

    def zpos():
        return float(_np(radio.get_position_orientation()[0])[2])

    def ag_state():
        p = getattr(robot, "_ag_obj_constraint_params", {})
        ih = getattr(robot, "_ag_obj_in_hand", {})
        return {k: (v is not None) for k, v in p.items()}, {k: (v is not None) for k, v in ih.items()}

    # ---- 1. diagnosis ---------------------------------------------------------------------
    print("AGFIX pre: constraint/in_hand:", *ag_state(), "radio_z:", round(zpos(), 4), flush=True)
    try:
        contacts, links = robot._find_gripper_contacts(arm=arm)
        print("AGFIX gripper contacts:", sorted(contacts), flush=True)
    except Exception as e:  # noqa: BLE001
        print("AGFIX _find_gripper_contacts raised:", repr(e), flush=True)
    try:
        in_hand = robot._calculate_in_hand_object(arm=arm)
        print("AGFIX _calculate_in_hand_object:", in_hand, flush=True)
    except Exception as e:  # noqa: BLE001
        print("AGFIX _calculate_in_hand_object raised:", repr(e), flush=True)

    act = demo_actions[T_START]
    for t in range(10):
        env.step(act)
    print("AGFIX after 10 replay steps: constraint/in_hand:", *ag_state(),
          "radio_z:", round(zpos(), 4), flush=True)
    try:
        contacts, links = robot._find_gripper_contacts(arm=arm)
        print("AGFIX contacts after steps:", sorted(contacts), flush=True)
        in_hand = robot._calculate_in_hand_object(arm=arm)
        print("AGFIX in_hand after steps:", in_hand, flush=True)
    except Exception as e:  # noqa: BLE001
        print("AGFIX post-step contact check raised:", repr(e), flush=True)

    # ---- 2. fix: direct establish ---------------------------------------------------------
    target_obj, target_link_name = None, None
    if in_hand is not None:
        try:
            target_obj, target_link_name = in_hand
            target_link_name = target_link_name if isinstance(target_link_name, str) else str(target_link_name)
        except Exception:  # noqa: BLE001
            target_obj = None
    if target_obj is None:
        target_obj = radio
        target_link_name = radio.root_link.body_name if hasattr(radio.root_link, "body_name") \
            else list(radio.links.keys())[0]
    print("AGFIX establishing on:", target_obj.name, target_link_name, flush=True)

    joint_type = robot._get_assisted_grasp_joint_type(target_obj, target_link_name)
    print("AGFIX joint_type:", joint_type, flush=True)
    link_prim_path = target_obj.links[target_link_name].prim_path
    contact_pos = None
    try:
        contact_pos = robot._find_finger_contact_position(arm, link_prim_path)
    except Exception as e:  # noqa: BLE001
        print("AGFIX _find_finger_contact_position raised:", repr(e), flush=True)
    if contact_pos is None:
        import torch as th
        tips = [l.get_position_orientation()[0] for l in robot.finger_links[arm]]
        contact_pos = th.stack(tips).mean(dim=0)
        print("AGFIX contact fallback = fingertip midpoint", flush=True)
    robot._establish_grasp(target_obj, target_link_name, arm, contact_pos, joint_type)
    print("AGFIX post-establish: constraint/in_hand:", *ag_state(), flush=True)

    # ---- 3. verify ------------------------------------------------------------------------
    z0 = zpos()
    settle_drift = None
    for t in range(120):
        env.step(act)
        if t == 29:
            settle_drift = abs(zpos() - z0)
        if t % 10 == 0 or t == 119:
            print(f"AGFIX t={t:3d} radio_z={zpos():.4f} dz={zpos() - z0:+.4f} "
                  f"AG={ag_state()[0]}", flush=True)
    ok = settle_drift is not None and settle_drift < 0.001
    print(f"AGFIX settle_drift_30={settle_drift if settle_drift is None else round(settle_drift, 5)} "
          f"final_dz={zpos() - z0:+.4f}", flush=True)
    print("AG_FIX_PROBE", "PASS" if ok else "FAIL", flush=True)
    print("PROBE_DONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
