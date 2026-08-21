"""RESTORE-FIDELITY PROBE: why did demo-action replay from a restored frame miss the grasp?

Restores demo 10 to f1100 (25 steps before lift), then numerically diffs the sim against the
demo recording while replaying the demo's own actions:
  - radio object world pose from the live scene registry vs metalink_labels meta_world[f]
  - both EEF positions + gripper widths (61-d eval proprio layout)
  - assisted-grasp constraint state (robot._ag_obj_constraint_params)
Prints every 10 steps for 120 steps (covers close ~f1110-1160 and lift f1125+).

Run inside the behavior env (no policy server needed):
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 python probe_restore_fidelity.py
"""

import os

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

DEMO = 10
T_START = 1300
STEPS = 120

from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame  # noqa: E402


def main():
    import h5py
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper

    ml = np.load(f"/root/metalink_labels/ep{DEMO}.npz")["meta_world"]
    h5 = f"/root/rawdemos/task-0000/episode_{DEMO:08d}.hdf5"
    with h5py.File(h5, "r") as f:
        demo_actions = f["data/demo_0/action"][:]

    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=h5, output_path="/root/rc_probe_tmp.hdf5",
        robot_obs_modalities=("rgb", "proprio"),
        robot_proprio_keys=EVAL_PROPRIO_KEYS,
    )
    env = wrapper.env

    radios = [o for o in wrapper.scene.objects if "radio" in o.name.lower()]
    print("PROBE radio objects:", [o.name for o in radios], flush=True)
    radio = radios[0]
    robot = wrapper.scene.robots[0] if hasattr(wrapper.scene, "robots") else None

    restore_to_frame(wrapper, 0, T_START)
    pos, orn = radio.get_position_orientation()
    pos = np.asarray(pos.cpu() if hasattr(pos, "cpu") else pos, dtype=float)
    print(f"PROBE radio@restore: sim={np.round(pos, 4).tolist()} "
          f"demo_label={np.round(ml[T_START], 4).tolist()} "
          f"diff_m={np.linalg.norm(pos - ml[T_START]):.4f}", flush=True)

    def _p(v):
        return np.asarray(v.cpu() if hasattr(v, "cpu") else v, dtype=float)

    obs = env.get_obs()[0]

    def prop(o):
        node = o
        for k in ("robot_r1", "robot_r1::proprio", "proprio"):
            if isinstance(node, dict) and k in node:
                node = node[k]
        return _p(node).reshape(-1)

    for t in range(STEPS):
        act = demo_actions[min(T_START + t, len(demo_actions) - 1)]
        out = env.step(act)
        obs = out[0]
        if t % 10 == 0 or t == STEPS - 1:
            pos, _ = radio.get_position_orientation()
            pos = _p(pos)
            f = T_START + t
            lbl = ml[min(f, len(ml) - 1)]
            pp = prop(obs)
            eefL, gL = pp[17:20], pp[24:26].sum()
            ag = getattr(robot, "_ag_obj_constraint_params", None) if robot is not None else None
            ag_on = {k: (v is not None) for k, v in ag.items()} if isinstance(ag, dict) else "n/a"
            print(f"PROBE t={t:3d} f={f} radio_sim={np.round(pos,3).tolist()} "
                  f"radio_demo={np.round(lbl,3).tolist()} "
                  f"drift={np.linalg.norm(pos-lbl):.3f} eefL={np.round(eefL,3).tolist()} "
                  f"eefL_to_radio={np.linalg.norm(eefL-pos):.3f} gripL={gL:.3f} AG={ag_on}",
                  flush=True)
    print("PROBE_DONE", flush=True)
    og.clear()


if __name__ == "__main__":
    main()
