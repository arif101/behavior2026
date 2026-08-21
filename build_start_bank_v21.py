"""V2.1 start-bank generator (SKILL_TRAINER_V2_SPEC.md §6, V2.1-c/-d).

Extends the v1 bank logic (build_start_bank.py, logic reused verbatim) with:
  - press−50 rung (stage 4) bridging the measured −25 → −100 cliff (0/301 in V2 round 0)
  - GRASP-INITIATION anchors (stage "G", family pick_up_from): quasi-static frame ~10
    steps before the radio first lifts >3 cm off its resting height; lift_z baseline
    recorded for the AG∧lifted∧held success predicate.

Writes /root/skill_start_bank_v21.json ONLY (v1 bank + scale file remain frozen).
CPU-only (parquet + metalink labels) — safe to run beside the live training run.
"""

import glob
import json

import h5py  # noqa: F401  (parity with v1 imports)
import numpy as np
import pyarrow.parquet as pq

REF = "/root/b1k_radio_map"
GRIP_L, GRIP_R = slice(24, 26), slice(49, 51)
OFFSETS = {0: 5, 1: 10, 2: 25, 3: 100, 4: 50}   # stage ids stable; order by VALUE
QS_WINDOW = 5
QS_THRESH = 0.002
LIFT_THRESH = 0.03       # metres above resting height = "lift started"
GRASP_PRE = 10           # nominal steps before first-lift for the grasp anchor
LIFT_SUCCESS = 0.05      # success predicate: lifted ≥5 cm above anchor height


def quasi_static_anchor(mw, ref_frame, k_nominal):
    motion = np.linalg.norm(np.diff(mw, axis=0), axis=1)
    lo = max(ref_frame - k_nominal - 15, QS_WINDOW + 1)
    for f in range(ref_frame - k_nominal, lo - 1, -1):
        if f < QS_WINDOW + 1:
            break
        if motion[f - QS_WINDOW:f].max() < QS_THRESH:
            return f
    return max(ref_frame - k_nominal, 1)


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
    print(f"{len(rows)} reference episodes on box")

    bank = []
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
            held = [arm for arm, (gs, d) in
                    (("left", (GRIP_L, dL[f])), ("right", (GRIP_R, dR[f])))
                    if st[f, gs].sum() < 0.02 and d < 0.25]
            if not held:
                return None
            other = "right" if active == "left" else "left"
            return other if other in held else held[0]

        mw = np.load(f"/root/metalink_labels/ep{rec['demo']}.npz")["meta_world"]

        # press rungs (incl. new press-50)
        for stg, k in OFFSETS.items():
            f0 = press - k if stg == 0 else quasi_static_anchor(mw, press, k)
            if 0 < f0 < len(st):
                bank.append(dict(demo=int(rec["demo"]), frame=int(f0), stage=stg,
                                 family="press", active_arm=active,
                                 holding_arm=holding_at(f0), press_frame=press))

        # grasp-initiation anchor (V2.1-c)
        z = mw[:, 2]
        rest = float(np.median(z[:min(300, press)]))
        lifted = np.where(z[:press] > rest + LIFT_THRESH)[0]
        if len(lifted):
            first_lift = int(lifted[0])
            fg = quasi_static_anchor(mw, first_lift, GRASP_PRE)
            if 0 < fg < len(st):
                bank.append(dict(demo=int(rec["demo"]), frame=int(fg), stage="G",
                                 family="pick_up_from", active_arm=active,
                                 holding_arm=holding_at(fg),
                                 lift_z=float(z[fg]), lift_success=LIFT_SUCCESS,
                                 first_lift_frame=first_lift, press_frame=press))
            print(f"demo {rec['demo']}: press f{press} first_lift f{first_lift} "
                  f"grasp_anchor f{fg} rest_z={rest:.3f} holding@anchor={holding_at(fg)}")
        else:
            print(f"demo {rec['demo']}: NO lift detected (press-in-place demo?) — "
                  f"no grasp anchor")

    held_out = sorted({e["demo"] for e in bank})[::5]
    json.dump({"entries": bank, "held_out_demos": held_out,
               "holdout_rule": "every 5th source demo (same as v1)",
               "offsets": OFFSETS,
               "grasp": {"lift_thresh": LIFT_THRESH, "pre": GRASP_PRE,
                         "success_lift": LIFT_SUCCESS}},
              open("/root/skill_start_bank_v21.json", "w"), indent=1)
    from collections import Counter
    c = Counter(str(e["stage"]) for e in bank)
    print(f"V21_BANK_DONE entries={len(bank)} by stage={dict(c)} held_out={held_out}")


if __name__ == "__main__":
    main()
