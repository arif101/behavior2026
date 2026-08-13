"""Build the curriculum start-state bank + the wrapper action-scale file.

Sources (spec §4 + [A-2026-08-13]): TRAIN-demo states only — press-frame anchors derived
from the REFERENCE parquet (b1k_radio_map: target_points + stage + observation.state per
frame, joined to raw demos via episodes-meta raw_episode_id). The Run-2 instance-301 eval
campaign is NOT a source (pinned). Policy-visited harvest extends this bank later via
harvest pass; stage-0/1/2 anchors from demo timelines get training started.

Press-frame heuristic per demo: first stage==2 frame whose closest-arm |EE-metalink| is
within 5 mm of that demo's stage-2 minimum (press initiation, not the global argmin —
approach can graze the minimum late). Active arm = argmin arm at that frame. Holding arm =
the OTHER arm iff its gripper qpos sum < 0.02 (closed) there, else None.

Bank entries: frame = press - k for k in {5, 25, 100} tagged stage 0/1/2. Frames are DEMO
frames for restore_to_frame; AG validity is enforced at reset by the wrapper (invalid
entries get discarded there, not here — needs live sim).

SCALE (12-d, single source of truth for wrapper + learner prior-normalization):
arm+torso = clip(2 * prior-buffer |p99|, 0.01, 0.08) rad; gripper = (center, half) span of
the active-arm gripper channel over all demo actions in the bank.

HOLD-OUT (pre-registered here, before any training): every 5th source demo in sorted
order — recorded in the bank json as held_out_demos; the sim-gate eval draws start states
ONLY from those demos, and the prior-buffer loader excludes them from training data.

Usage: python build_start_bank.py  (box; reads /root/b1k_radio_map + /root/skill_buffer_prior)
"""

import glob
import json

import h5py
import numpy as np
import pyarrow.parquet as pq

REF = "/root/b1k_radio_map"
GRIP_L, GRIP_R = slice(24, 26), slice(49, 51)
# Curriculum rungs (stage -> nominal steps before press). Stages >= 1 are re-anchored to
# the nearest QUASI-STATIC frame (metalink motion < 2 mm/frame over 5 frames) at or
# before the nominal offset: fixed offsets landed in the mid-lift band on 2 of the first
# 4 demos (radio restored IN MOTION -> AG validity discard, 2026-08-13 stage-1 pass), and
# dynamic starts are off the previous rung's visited distribution. press-10 rung added to
# bridge the -5 -> -25 gap (0/20 on the fixed-offset pass).
OFFSETS = {0: 5, 1: 10, 2: 25, 3: 100}
QS_WINDOW = 5
QS_THRESH = 0.002


def quasi_static_anchor(mw, press, k_nominal):
    """Latest frame f in [press - k_nominal - 15, press - k_nominal] whose trailing
    QS_WINDOW metalink displacements are all < QS_THRESH; falls back to the nominal."""
    motion = np.linalg.norm(np.diff(mw, axis=0), axis=1)
    lo = max(press - k_nominal - 15, QS_WINDOW + 1)
    for f in range(press - k_nominal, lo - 1, -1):
        if f < QS_WINDOW + 1:
            break
        if motion[f - QS_WINDOW:f].max() < QS_THRESH:
            return f
    return max(press - k_nominal, 1)


def main():
    emeta = pq.read_table(sorted(glob.glob(f"{REF}/meta/episodes/**/*.parquet",
                                           recursive=True))[0],
                          columns=["episode_index", "raw_episode_id",
                                   "data/chunk_index", "data/file_index"]).to_pydict()
    on_box = {int(p.split("_")[-1].split(".")[0])
              for p in glob.glob("/root/rawdemos/task-0000/episode_*.hdf5")}
    rows = [dict(ep=e, demo=r, chunk=c, file=f) for e, r, c, f in
            zip(emeta["episode_index"], emeta["raw_episode_id"],
                emeta["data/chunk_index"], emeta["data/file_index"]) if r in on_box]
    print(f"{len(rows)} reference episodes have raw demos on box")

    bank, grip_vals = [], []
    for rec in rows:
        t = pq.read_table(f"{REF}/data/chunk-{rec['chunk']:03d}/file-{rec['file']:03d}.parquet",
                          columns=["episode_index", "target_points", "stage",
                                   "observation.state"])
        m = t["episode_index"].to_numpy() == rec["ep"]
        tp = np.stack(t["target_points"].to_numpy()[m])
        stage = t["stage"].to_numpy()[m]
        st = np.stack(t["observation.state"].to_numpy()[m])
        dL, dR = np.linalg.norm(tp[:, :3], axis=1), np.linalg.norm(tp[:, 3:], axis=1)
        dmin = np.minimum(dL, dR)
        s2 = np.where(stage == 2)[0]
        if not len(s2):
            print(f"demo {rec['demo']}: no stage-2 frames -- skipped")
            continue
        lo = dmin[s2].min()
        press = int(s2[np.argmax(dmin[s2] <= lo + 0.005)])
        active = "left" if dL[press] < dR[press] else "right"

        def holding_at(f):
            """Arm(s) holding the radio AT frame f: gripper closed (sum<0.02) AND EE near
            the metalink (<0.25 m). Evaluated per entry frame — press-k frames can be
            mid-carry even when the press frame itself has the radio resting."""
            held = [arm for arm, (gs, d) in
                    (("left", (GRIP_L, dL[f])), ("right", (GRIP_R, dR[f])))
                    if st[f, gs].sum() < 0.02 and d < 0.25]
            if not held:
                return None
            other = "right" if active == "left" else "left"
            return other if other in held else held[0]

        mw = np.load(f"/root/metalink_labels/ep{rec['demo']}.npz")["meta_world"]
        for stg, k in OFFSETS.items():
            f0 = press - k if stg == 0 else quasi_static_anchor(mw, press, k)
            if 0 < f0 < len(st):
                bank.append(dict(demo=int(rec["demo"]), frame=int(f0), stage=stg,
                                 active_arm=active, holding_arm=holding_at(f0),
                                 press_frame=press))
        with h5py.File(f"/root/rawdemos/task-0000/episode_{rec['demo']:08d}.hdf5") as f:
            key = sorted(k for k in f["data"].keys() if k.startswith("demo_"))[0]
            acts = f[f"data/{key}/action"][:]
        gcol = 21 if active == "left" else 22
        grip_vals.append([float(acts[:, gcol].min()), float(acts[:, gcol].max())])
        anchors = {s: (press - k if s == 0 else quasi_static_anchor(mw, press, k))
                   for s, k in OFFSETS.items()}
        print(f"demo {rec['demo']}: press f{press} (d={dmin[press]:.3f}) active={active} "
              f"anchors={anchors} holding={[holding_at(f) for f in anchors.values()]}")

    meta = json.load(open("/root/skill_buffer_prior/skill_buffer_meta.json"))
    p99 = np.asarray(meta["action_abs_p99"][:11])
    scale = np.clip(2 * p99, 0.01, 0.08)
    g = np.asarray(grip_vals)
    gmin, gmax = float(g[:, 0].min()), float(g[:, 1].max())
    json.dump({"scale": [*scale.tolist(), 1.0],
               "grip_center": (gmin + gmax) / 2, "grip_half": (gmax - gmin) / 2,
               "derivation": "arm/torso clip(2*prior_p99,0.01,0.08); grip span of demo actions"},
              open("/root/skill_wrapper_scale.json", "w"), indent=2)

    demos = sorted({b["demo"] for b in bank})
    held_out = demos[::5]
    json.dump({"entries": bank, "held_out_demos": held_out,
               "holdout_rule": "every 5th source demo, sorted (pre-registered pre-launch)"},
              open("/root/skill_start_bank.json", "w"), indent=2)
    print(f"BANK_OK {len(bank)} entries from {len(demos)} demos; "
          f"held_out={held_out}; scale={np.round(scale, 4).tolist()} "
          f"grip=({(gmin + gmax) / 2:.3f}±{(gmax - gmin) / 2:.3f})")


if __name__ == "__main__":
    main()
