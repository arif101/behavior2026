"""Convert splice-clip episodes (b1k_radio_corrective, LeRobot-v3) into an RLPD prior buffer
of contact-skill tuples per CONTACT_SKILL_SPEC_v1.md SS2/SS4.

INPUT is the CONVERTED parquet dataset, not the raw rc_*.npz clips: the parquet's
target_points column already carries metalink-minus-EE in base frame via the validated
base-pose machinery (convert_clips_to_parquet.py: Kabsch resid 0.01-0.02 mm, EE->metalink
0.030 m at press), and every corrective episode is success=True by construction (the
converter's outcome filter). Re-deriving labels from raw npz here would duplicate that
machinery unvalidated.

Per-tuple derivation (obs layout recorded in <out>/skill_buffer_meta.json -- the env wrapper
MUST build its obs from that file, not a copy of these constants):

  obs (198-d) =
    proprio stack (130): [arm qpos 7, arm qvel 7, gripper qpos 2, gripper qvel 2,
                          trunk qpos 4, trunk qvel 4] = 26/frame x LAST 5 FRAMES
                          (window-entry padded by repeating the entry frame).
                          Active arm chosen PER EPISODE: the arm whose |EE-metalink| tail
                          minimum is smaller (logged; ties broken right).
    affordance point (3): target FROZEN at window entry (spec: no re-targeting mid-attempt)
                          = meta_base at the entry frame, re-expressed each step in the
                          CURRENT EE frame: R(eef_quat)^T @ (target_base - eef_pos).
    aff confidence  (1):  constant --aff-conf (default 0.75 ~ serving conf_p50 band
                          0.68-0.82). Head-error noise injection (sigma~2.6 cm) belongs to
                          the TRAINING-side randomization, not baked into stored tuples.
    L2 geometry    (64):  ZEROS + TODO. Wrist-local occupancy pooling is not derivable from
                          clip data (no map stack here). OPEN DESIGN ITEM: zeros in prior
                          data vs live features online is a prior/online obs mismatch --
                          resolve before the first launch (either drop L2 from skill obs v1
                          or compute it for prior data on-box from stored depth).

  action (12-d) = [arm delta qpos 7, torso delta qpos 4, gripper cmd 1]
    Stored 23-d actions are ABSOLUTE position targets (base 0:3, torso 3:7, armL 7:14,
    armR 14:21, grip 21:23 -- analyze_hover.py). Skill deltas = target - current qpos.
    Stored UNSQUASHED in physical units; the RLPD loader owns tanh normalization with the
    wrapper's limits (recorded here as per-dim p99 |delta| stats for a sanity check).
    Gripper passes the absolute command through (serving-contract semantics) -- the env
    wrapper must use the same convention. Base channels are dropped (splice base cmds are
    measured 0; the skill does not own the base).

  reward: 0 everywhere, +--success-reward (default 1.0) on each episode's final tuple
    (all corrective episodes are successes; asserted via next.terminated).
    dist(EE, frozen target) per tuple is stored alongside so the SS4 shaping terms can be
    recomputed by the loader without touching this file (shaping is ablatable).

  window: tuples start at the first frame with dist(EE_active, live target) < --window-radius
    (default 0.10 m; gate fires at 0.07 -- margin for the pre-gate approach tail) and run to
    episode end. Episodes that never enter the window are dropped (logged).

  splits: raw_episode_id (source demo) is stored per tuple -- hold out BY DEMO, never by
    tuple (same-demo states are near-duplicates; a tuple-level split leaks).

Usage (box):
  /root/miniconda3/envs/openpi/bin/python convert_splices_to_skill_buffer.py \
      --ds /root/b1k_radio_corrective --out /root/skill_buffer_prior
"""

import argparse
import glob
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

# 61-d eval proprio layout (EVAL_PROPRIO_KEYS order, reverse_curriculum_collect.py)
P = {
    "base_qvel": slice(0, 3),
    "arm_left_qpos": slice(3, 10), "arm_left_qvel": slice(10, 17),
    "eef_left_pos": slice(17, 20), "eef_left_quat": slice(20, 24),
    "gripper_left_qpos": slice(24, 26), "gripper_left_qvel": slice(26, 28),
    "arm_right_qpos": slice(28, 35), "arm_right_qvel": slice(35, 42),
    "eef_right_pos": slice(42, 45), "eef_right_quat": slice(45, 49),
    "gripper_right_qpos": slice(49, 51), "gripper_right_qvel": slice(51, 53),
    "trunk_qpos": slice(53, 57), "trunk_qvel": slice(57, 61),
}
# 23-d action layout (analyze_hover.py)
A_TORSO, A_ARM = slice(3, 7), {"left": slice(7, 14), "right": slice(14, 21)}
A_GRIP = {"left": 21, "right": 22}
STACK = 5


def q2r(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def per_frame_proprio(state, arm):
    return np.concatenate([
        state[P[f"arm_{arm}_qpos"]], state[P[f"arm_{arm}_qvel"]],
        state[P[f"gripper_{arm}_qpos"]], state[P[f"gripper_{arm}_qvel"]],
        state[P["trunk_qpos"]], state[P["trunk_qvel"]],
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="/root/b1k_radio_corrective")
    ap.add_argument("--out", default="/root/skill_buffer_prior")
    ap.add_argument("--window-radius", type=float, default=0.10)
    ap.add_argument("--success-reward", type=float, default=1.0)
    ap.add_argument("--aff-conf", type=float, default=0.75)
    ap.add_argument("--l2-dim", type=int, default=64)
    a = ap.parse_args()
    ds, out = pathlib.Path(a.ds), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    fp = sorted(glob.glob(str(ds / "data" / "**" / "*.parquet"), recursive=True))
    t = pq.read_table(fp[0], columns=["episode_index", "observation.state", "action",
                                      "target_points", "next.terminated"])
    ep_idx = t["episode_index"].to_numpy()
    state = np.stack(t["observation.state"].to_numpy()).astype(np.float64)
    act23 = np.stack(t["action"].to_numpy()).astype(np.float64)
    tp = np.stack(t["target_points"].to_numpy()).astype(np.float64)
    term = t["next.terminated"].to_numpy()

    emeta = pq.read_table(sorted(glob.glob(
        str(ds / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))[0],
        columns=["episode_index", "raw_episode_id"]).to_pydict()
    demo_of = dict(zip(emeta["episode_index"], emeta["raw_episode_id"]))

    obs_l, nobs_l, act_l, rew_l, done_l, dist_l, demo_l, ep_l = ([] for _ in range(8))
    dropped = 0
    for ep in np.unique(ep_idx):
        m = ep_idx == ep
        st, ac, tpe = state[m], act23[m], tp[m]
        n = len(st)
        assert term[m][-1], f"ep{ep}: final row not terminated -- unexpected for corrective"

        tail = slice(int(0.6 * n), n)
        dL = np.linalg.norm(tpe[:, :3], axis=1)
        dR = np.linalg.norm(tpe[:, 3:], axis=1)
        arm = "left" if dL[tail].min() < dR[tail].min() else "right"
        d_live = dL if arm == "left" else dR
        ee_s, eq_s = (P["eef_left_pos"], P["eef_left_quat"]) if arm == "left" \
            else (P["eef_right_pos"], P["eef_right_quat"])

        in_w = np.where(d_live < a.window_radius)[0]
        if not len(in_w):
            dropped += 1
            print(f"ep{ep} (demo {demo_of.get(ep, -1)}): never within "
                  f"{a.window_radius} m (min {d_live.min():.3f}) -- dropped", flush=True)
            continue
        w0 = int(in_w[0])
        # frozen target in base frame at window entry (tp = meta_base - ee)
        target_base = st[w0, ee_s] + (tpe[w0, :3] if arm == "left" else tpe[w0, 3:])

        def obs_at(s):
            idx = np.clip(np.arange(s - STACK + 1, s + 1), w0, None)
            stack = np.concatenate([per_frame_proprio(st[i], arm) for i in idx])
            v_base = target_base - st[s, ee_s]
            v_ee = q2r(st[s, eq_s]).T @ v_base
            return np.concatenate([stack, v_ee, [a.aff_conf], np.zeros(a.l2_dim)])

        obs_seq = [obs_at(s) for s in range(w0, n)]
        for k, s in enumerate(range(w0, n - 1)):
            delta_arm = ac[s, A_ARM[arm]] - st[s, P[f"arm_{arm}_qpos"]]
            delta_torso = ac[s, A_TORSO] - st[s, P["trunk_qpos"]]
            obs_l.append(obs_seq[k])
            nobs_l.append(obs_seq[k + 1])
            act_l.append(np.concatenate([delta_arm, delta_torso, [ac[s, A_GRIP[arm]]]]))
            last = s == n - 2
            rew_l.append(a.success_reward if last else 0.0)
            done_l.append(last)
            dist_l.append(np.linalg.norm(target_base - st[s, ee_s]))
            demo_l.append(int(demo_of.get(ep, -1)))
            ep_l.append(int(ep))
        print(f"ep{ep} (demo {demo_of.get(ep, -1)}): arm={arm} window [{w0}:{n}] "
              f"({n - 1 - w0} tuples), entry dist {d_live[w0]:.3f} m, "
              f"final dist {d_live[-1]:.3f} m", flush=True)

    obs = np.asarray(obs_l, np.float32)
    acts = np.asarray(act_l, np.float32)
    np.savez_compressed(
        out / "prior_buffer.npz", obs=obs, next_obs=np.asarray(nobs_l, np.float32),
        action=acts, reward=np.asarray(rew_l, np.float32), done=np.asarray(done_l),
        dist=np.asarray(dist_l, np.float32), demo=np.asarray(demo_l, np.int64),
        episode=np.asarray(ep_l, np.int64))
    meta = {
        "obs_layout": {"proprio_stack": [0, STACK * 26], "aff_point_ee": [130, 133],
                       "aff_conf": [133, 134], "l2_geometry": [134, 134 + a.l2_dim]},
        "per_frame_proprio": ["arm_qpos*7", "arm_qvel*7", "grip_qpos*2", "grip_qvel*2",
                              "trunk_qpos*4", "trunk_qvel*4"],
        "action_layout": ["arm_delta_qpos*7", "torso_delta_qpos*4", "gripper_abs_cmd*1"],
        "action_units": "raw physical (UNSQUASHED); loader owns tanh normalization",
        "action_abs_p99": np.percentile(np.abs(acts), 99, axis=0).tolist(),
        "l2_geometry": "ZEROS -- open item, see module docstring",
        "window_radius_m": a.window_radius, "aff_conf_const": a.aff_conf,
        "stack": STACK, "success_reward": a.success_reward,
        "source_demos": sorted(set(demo_l)),
    }
    (out / "skill_buffer_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSKILL_BUFFER_OK {len(obs)} tuples from "
          f"{len(set(ep_l))} episodes ({dropped} dropped), "
          f"{len(set(demo_l))} source demos -> {out}")
    print(f"action |p99| per dim: {np.round(meta['action_abs_p99'], 4)}")


if __name__ == "__main__":
    main()
