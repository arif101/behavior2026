"""V2 snapshot bank builder (SKILL_TRAINER_V2_SPEC.md D4).

Per demo: boot playback env once, then for each bank entry run the FULL v1 validation path
(playback restore + AG re-establish + settle checks) with up to --retries attempts; on the
first valid settle, freeze og.sim.dump_state -> npz. V2 resets then load the snapshot
(~1 s, dz 0.33 mm validated by the day-0 profiler) instead of re-paying 34-71 s per episode.

Also records per-scene radio pose/yaw + robot base pose into the manifest (feeds the
cold-scene orientation analysis for free), and documents unrecoverable entries (the
d30/d50 adjudication).

One demo per process (Isaac env-reuse leak). Driver: snapshot_bank_driver.sh.
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    python -u build_snapshot_bank.py --demo-id 30 --retries 6
"""

import argparse
import json
import math
import os
import time

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")

OUT_DIR = "/root/snapshot_bank"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-id", type=int, required=True)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--bank", default="/root/skill_start_bank.json")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--grip-fidelity", action="store_true",
                    help="V2.1-a: record demo-true in-hand pose at raw restore and warp "
                         "the object back to it after AG re-establish + settle")
    a = ap.parse_args()
    out_dir = a.out_dir
    os.makedirs(out_dir, exist_ok=True)

    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS
    from skill_env_wrapper import SkillCommitEnv, q2r, _np

    h5 = f"/root/rawdemos/task-0000/episode_{a.demo_id:08d}.hdf5"
    t0 = time.time()
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=h5, output_path=f"/root/snapbank_tmp_{a.demo_id}.hdf5",
        robot_obs_modalities=("proprio",),          # render-off (D5)
        robot_proprio_keys=EVAL_PROPRIO_KEYS)
    bank = json.load(open(a.bank))
    entries = [dict(e) for e in bank["entries"] if e["demo"] == a.demo_id]
    env = SkillCommitEnv(wrapper, list(entries), shaping=False, seed=0, randomize=False)
    print(f"BANKBUILD d{a.demo_id}: booted {time.time() - t0:.0f}s, {len(entries)} entries",
          flush=True)

    from reverse_curriculum_collect import restore_to_frame
    from skill_env_wrapper import P
    epid = int(sorted(k.split("_")[1] for k in wrapper.input_hdf5["data"].keys()
                      if k.startswith("demo_"))[0])

    def ee_world(arm):
        s = env._proprio61()
        bp, bq = env.robot.get_position_orientation()
        Rb = q2r(_np(bq))
        p = _np(bp) + Rb @ s[P[arm]["eef_pos"]]
        R = Rb @ q2r(s[P[arm]["eef_quat"]])
        return p, R

    def rel_pose(arm):
        tp, tq = env.target.get_position_orientation()
        pe, Re = ee_world(arm)
        return Re.T @ (_np(tp) - pe), Re.T @ q2r(_np(tq))

    def grip_fidelity_reset(e):
        """V2.1-a: restore raw, capture demo-true in-hand pose, establish AG at the true
        state, settle, and measure relative-pose drift. Returns (ok, drift_mm, mode)."""
        arm = e["holding_arm"]
        restore_to_frame(wrapper, epid, e["frame"])
        p_rel0, R_rel0 = rel_pose(arm)
        env._reestablish_ag(arm)                      # anchor from the TRUE finger state
        import h5py
        with h5py.File(wrapper.input_hdf5.filename, "r") as f:
            key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
            hold_act = f[f"data/{key}/action"][min(e["frame"],
                                                   f[f"data/{key}/action"].shape[0] - 1)]
        for _ in range(60):
            env.env.step(hold_act)
        p_rel1, _ = rel_pose(arm)
        drift = float(np.linalg.norm(p_rel1 - p_rel0) * 1000)
        # stillness check as in v1
        z0 = float(_np(env.target.get_position_orientation()[0])[2])
        for _ in range(30):
            env.env.step(hold_act)
        dz = abs(float(_np(env.target.get_position_orientation()[0])[2]) - z0) * 1000
        ok = env._ag_ok(arm) and dz < 1.0 and drift < 8.0
        return ok, drift, dz

    manifest = {"demo": a.demo_id, "entries": [], "scene": {}}
    for e in entries:
        rec = {"stage": e["stage"], "frame": e["frame"], "family": e.get("family", "press"),
               "holding_arm": e.get("holding_arm"), "active_arm": e["active_arm"],
               "lift_z": e.get("lift_z"), "status": "INVALID", "attempts": 0}
        for attempt in range(a.retries):
            rec["attempts"] = attempt + 1
            try:
                if a.grip_fidelity and e.get("holding_arm"):
                    ok, drift, dz = grip_fidelity_reset(e)
                    rec["rel_drift_mm"] = round(drift, 1)
                    rec["dz_mm"] = round(dz, 2)
                    if not ok:
                        continue
                else:
                    env.bank = [dict(e)]
                    env.reset()      # full v1 validation: restore + AG + settle checks
            except RuntimeError:
                continue             # discarded this attempt; retry fresh
            snap = og.sim.dump_state(serialized=True)   # harvest-proven flat format
            path = f"{out_dir}/d{a.demo_id}_s{e['stage']}_f{e['frame']}.npz"
            np.savez_compressed(path, state=snap, state_size=np.int64(len(snap)))
            rec["status"] = "OK"
            rec["snapshot"] = path
            # scene geometry (once per demo is enough, but cheap to record per entry)
            tp, tq = env.target.get_position_orientation()
            tp, tq = _np(tp), _np(tq)
            yaw = math.degrees(math.atan2(2 * (tq[3] * tq[2] + tq[0] * tq[1]),
                                          1 - 2 * (tq[1] ** 2 + tq[2] ** 2)))
            bp, _ = env.robot.get_position_orientation()
            manifest["scene"] = {"radio_pos": [round(float(x), 4) for x in tp],
                                 "radio_yaw_deg": round(yaw, 1),
                                 "robot_base": [round(float(x), 4) for x in _np(bp)[:2]]}
            break
        manifest["entries"].append(rec)
        print(f"BANKBUILD d{a.demo_id} s{e['stage']} f{e['frame']}: {rec['status']} "
              f"(attempts={rec['attempts']})", flush=True)

    json.dump(manifest, open(f"{out_dir}/manifest_d{a.demo_id}.json", "w"), indent=1)
    ok = sum(1 for r in manifest["entries"] if r["status"] == "OK")
    print(f"BANKBUILD_DONE d{a.demo_id}: {ok}/{len(entries)} snapshots, "
          f"{time.time() - t0:.0f}s total", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
