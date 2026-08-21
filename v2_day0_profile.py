"""Day-0 throughput/restore profiler (SKILL_TRAINER_V2_SPEC.md §4 d0).

One booted sim per invocation; measures the quantities that price the V2 rebuild:
  - boot time
  - v1 reset path (SkillCommitEnv.reset: playback restore + AG + settle)  x3
  - bare playback restore_to_frame                                        x3
  - snapshot path: dump_state once; then (load_state + re-AG + 30-step
    dz-validity check) x5  -> is a validated snapshot a valid start state?
  - stepping rate over 300 steps through the full training path (env.step)
  - VRAM after boot and at end

Run (behavior env, ALWAYS -u; os._exit at end per og.shutdown-hangs gotcha):
  PYTHONPATH=/root OMNI_KIT_ALLOW_ROOT=1 OMNIGIBSON_HEADLESS=1 \
    python -u v2_day0_profile.py --demo-id 80 --modalities rgb,proprio \
    --out /root/day0_profile_A.json
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

os.environ.setdefault("OMNI_KIT_ALLOW_ROOT", "1")


def vram():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-id", type=int, default=80)
    ap.add_argument("--modalities", default="rgb,proprio")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"demo": a.demo_id, "modalities": a.modalities, "errors": []}

    t0 = time.time()
    import omnigibson as og
    from omnigibson.macros import gm
    gm.HEADLESS = True
    gm.ENABLE_TRANSITION_RULES = False
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper
    from reverse_curriculum_collect import EVAL_PROPRIO_KEYS, restore_to_frame
    from skill_env_wrapper import SkillCommitEnv

    h5 = f"/root/rawdemos/task-0000/episode_{a.demo_id:08d}.hdf5"
    mods = tuple(a.modalities.split(","))
    wrapper = HDF5PlaybackWrapper.create_from_hdf5(
        input_path=h5, output_path=f"/root/day0_tmp_{a.demo_id}_{len(mods)}.hdf5",
        robot_obs_modalities=mods, robot_proprio_keys=EVAL_PROPRIO_KEYS)
    res["boot_s"] = round(time.time() - t0, 1)
    res["vram_after_boot_mb"] = vram()
    print(f"BOOT {res['boot_s']}s vram={res['vram_after_boot_mb']}MB", flush=True)

    def fresh_entries():
        bank = json.load(open("/root/skill_start_bank.json"))
        return [dict(e) for e in bank["entries"] if e["demo"] == a.demo_id]

    env = SkillCommitEnv(wrapper, fresh_entries(), shaping=True, seed=0, randomize=False)
    # parse the real hdf5 group episode id (group name != demo id; wrapper pattern)
    epid = int(sorted(k.split("_")[1] for k in wrapper.input_hdf5["data"].keys()
                      if k.startswith("demo_"))[0])
    hold_arm = fresh_entries()[0].get("holding_arm")
    frame0 = fresh_entries()[0]["frame"]

    # --- v1 full reset path x3 ---------------------------------------------------------
    v1_reset = []
    for i in range(3):
        env.bank = fresh_entries()
        t = time.time()
        try:
            env.reset()
            v1_reset.append(round(time.time() - t, 2))
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"v1_reset[{i}]: {e}")
            break
    res["v1_reset_s"] = v1_reset
    print(f"V1_RESET {v1_reset}", flush=True)

    # --- bare playback restore x3 ------------------------------------------------------
    bare = []
    for i in range(3):
        t = time.time()
        try:
            restore_to_frame(wrapper, epid, frame0)
            bare.append(round(time.time() - t, 2))
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"bare_restore[{i}]: {e}")
            break
    res["bare_restore_s"] = bare
    print(f"BARE_RESTORE {bare}", flush=True)

    # --- snapshot path ------------------------------------------------------------------
    # capture a VALIDATED state: full v1 reset (AG live, settled), then dump.
    try:
        env.bank = fresh_entries()
        env.reset()
        t = time.time()
        snap = og.sim.dump_state(serialized=False)
        res["dump_state_s"] = round(time.time() - t, 3)
        hold = env.hold_act
        snap_load, snap_dz_mm, snap_ag = [], [], []
        for i in range(5):
            for _ in range(20):          # walk away from the state first
                env.env.step(hold + np.random.uniform(-0.02, 0.02, hold.shape).astype(hold.dtype))
            t = time.time()
            og.sim.load_state(snap, serialized=False)
            ag = False
            if hold_arm:
                ag = env._reestablish_ag(hold_arm)
            load_s = time.time() - t
            # validity: 30-step dz stability (the v1 acceptance test, no 60-step presettle)
            z0 = None
            import torch as th  # noqa: F401
            p = env.target.get_position_orientation()[0]
            z0 = float(p[2] if not hasattr(p, "cpu") else p.cpu().numpy()[2])
            for _ in range(30):
                env.env.step(hold)
            p = env.target.get_position_orientation()[0]
            z1 = float(p[2] if not hasattr(p, "cpu") else p.cpu().numpy()[2])
            snap_load.append(round(load_s, 3))
            snap_dz_mm.append(round(abs(z1 - z0) * 1000, 2))
            snap_ag.append(bool(ag))
        res["snap_load_s"] = snap_load
        res["snap_dz_mm"] = snap_dz_mm
        res["snap_ag_ok"] = snap_ag
        print(f"SNAP load={snap_load} dz_mm={snap_dz_mm} ag={snap_ag}", flush=True)
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"snapshot: {e}")

    # --- stepping rate through the training path ---------------------------------------
    try:
        env.bank = fresh_entries()
        env.reset()
        t = time.time()
        n = 0
        done = False
        while n < a.steps:
            if done:
                env.bank = fresh_entries()
                env.reset()
                done = False
            _, _, done, _ = env.step(np.zeros(12, np.float32))
            n += 1
        dt = time.time() - t
        res["step_rate_hz"] = round(n / dt, 2)
        res["s_per_step"] = round(dt / n, 3)
        print(f"STEP_RATE {res['step_rate_hz']}/s ({res['s_per_step']} s/step, n={n})",
              flush=True)
    except Exception as e:  # noqa: BLE001
        res["errors"].append(f"stepping: {e}")

    res["vram_end_mb"] = vram()
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"DAY0_PROFILE_DONE -> {a.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
