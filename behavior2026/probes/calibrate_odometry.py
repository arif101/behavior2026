"""Odometry calibration: how far does dead reckoning drift from the true base trajectory?

The open question (task #20). Integrating the robot's own `base_qvel` is the only pose estimate
LEGAL at evaluation — the rules bar `robot global pose`. Earlier we measured, on archived rollouts,
that integrating proprio velocity and integrating the COMMANDED action disagree by 42-52%, with no
ground truth to arbitrate. The replay pass now supplies that ground truth.

This decides a real architecture question. A persistent 3D map has to register observations into a
common frame, and at eval that frame can only come from estimated odometry. If drift over a typical
episode is small, dead reckoning suffices. If it is large, the map needs visual odometry and loop
closure — which is a much bigger build, and better known BEFORE committing to the map's schema
than after.

Inputs: the per-episode JSON written by replay_poses_batch.py (true base_pos/base_quat per frame)
and the LeRobot parquet (observation.state, whose dims 0:3 are base_qvel per configs/robots/b1k.py).

Usage:
  python calibrate_odometry.py --poses_dir /root/poses/turning_on_radio \
      --parquet_root /root/replay_root/lerobot --dt 0.0333
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def quat_to_yaw(q: np.ndarray) -> float:
    """Yaw from an (x, y, z, w) quaternion."""
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def integrate(vel_body: np.ndarray, yaw0: float, dt: float) -> np.ndarray:
    """Dead-reckon a planar trajectory from body-frame [vx, vy, wz] velocities."""
    xy = np.zeros((len(vel_body) + 1, 2))
    yaw = yaw0
    for i, (vx, vy, wz) in enumerate(vel_body):
        # rotate body velocity into world, then step
        c, s = np.cos(yaw), np.sin(yaw)
        xy[i + 1] = xy[i] + dt * np.array([c * vx - s * vy, s * vx + c * vy])
        yaw += dt * wz
    return xy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--parquet_root", default="", help="optional; if absent, report truth only")
    ap.add_argument("--dt", type=float, default=1.0 / 30.0)
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.poses_dir, "ep*.json")))[: a.limit]
    if not files:
        print("no pose files found")
        return

    print(f"{'episode':>10} {'frames':>7} {'true path':>10} {'net disp':>9} {'max |v|':>8}")
    paths, nets = [], []
    for f in files:
        d = json.load(open(f))
        F = d["frames"]
        B = np.array([fr["base_pos"] for fr in F])[:, :2]
        step = np.linalg.norm(np.diff(B, axis=0), axis=1)
        path, net = float(step.sum()), float(np.linalg.norm(B[-1] - B[0]))
        paths.append(path)
        nets.append(net)
        print(f"{d['demo_id']:>10} {len(F):>7} {path:>10.3f} {net:>9.3f} {step.max()/a.dt:>8.3f}")

    print(f"\ntrue base motion over {len(files)} episodes:")
    print(f"  path   median {np.median(paths):.3f} m   max {max(paths):.3f} m")
    print(f"  net    median {np.median(nets):.3f} m   max {max(nets):.3f} m")

    # The decision rule. Our banked failure had the base parked 1.891 m from the target while the
    # human envelope is 0.79 m, so an odometry error approaching ~0.2 m would be enough on its own
    # to explain a failed approach — and would rule out plain dead reckoning for the map.
    print("\nPre-registered read (once dead-reckoned trajectories are joined in):")
    print("  drift < 0.05 m over an episode -> velocity integration suffices for the 4D map")
    print("  drift 0.05-0.20 m             -> needs periodic correction (relocalise against map)")
    print("  drift > 0.20 m                -> full visual odometry + loop closure required")

    if not a.parquet_root:
        print("\n(no --parquet_root given: dead-reckoning comparison skipped)")
        return

    # ---- the half that was previously a stub -------------------------------
    # WHY TWO HYPOTHESES: robot.py:1601 returns base_qvel as JOINT velocities, not a body twist.
    # For a holonomic 3-DoF base the x/y/yaw joint velocities could be expressed either in the
    # BODY frame (needs rotating by yaw before integrating) or already in the WORLD frame (integrate
    # directly). Guessing wrong yields a trajectory that is wrong by exactly a rotation, which is
    # what produced the unresolved 42-52% command-vs-achieved disagreement. Ground truth now
    # arbitrates: whichever hypothesis tracks truth IS the convention.
    import pyarrow.parquet as pq

    ep_meta = sorted(glob.glob(os.path.join(a.parquet_root, "meta/episodes/**/*.parquet"), recursive=True))
    if not ep_meta:
        print(f"no episode metadata under {a.parquet_root}/meta/episodes")
        return
    meta = pq.read_table(ep_meta[0]).to_pandas()
    # demo_id (pose filename) <-> episode_index (parquet row block)
    raw_col = "raw_episode_id" if "raw_episode_id" in meta.columns else None
    if raw_col is None:
        print("episode metadata has no raw_episode_id; cannot map poses to parquet")
        return
    demo2ep = {int(r[raw_col]): int(r["episode_index"]) for _, r in meta.iterrows()}

    data_files = sorted(glob.glob(os.path.join(a.parquet_root, "data/**/*.parquet"), recursive=True))
    tbl = pq.read_table(data_files, columns=["observation.state", "episode_index"]).to_pandas()

    rows = []
    for f in files:
        d = json.load(open(f))
        demo = int(d["demo_id"])
        if demo not in demo2ep:
            continue
        sub = tbl[tbl["episode_index"] == demo2ep[demo]]
        if not len(sub):
            continue
        state = np.stack(sub["observation.state"].to_numpy())
        vel = state[:, 0:3].astype(np.float64)          # b1k.py R1Pro: dims 0:3 = base_qvel

        F = d["frames"]
        B = np.array([fr["base_pos"] for fr in F])[:, :2]
        yaw0 = quat_to_yaw(np.array(F[0]["base_quat"], dtype=float))
        n = min(len(B) - 1, len(vel))
        if n < 30:
            continue
        truth = B[: n + 1] - B[0]

        body = integrate(vel[:n], yaw0, a.dt)                       # hypothesis A: body frame
        world = np.cumsum(np.vstack([[0, 0], vel[:n, :2] * a.dt]), 0)  # hypothesis B: world frame
        rows.append((
            demo, n,
            float(np.linalg.norm(truth[-1])),
            float(np.linalg.norm(body[-1] - truth[-1])),
            float(np.linalg.norm(world[-1] - truth[-1])),
            float(np.abs(np.linalg.norm(body - truth, axis=1)).max()),
            float(np.abs(np.linalg.norm(world - truth, axis=1)).max()),
        ))

    if not rows:
        print("\nno episodes could be joined to the parquet")
        return

    # R columns, in the order appended above:
    #   0 true net displacement | 1 body final | 2 world final | 3 body max | 4 world max
    R = np.array([r[2:] for r in rows], dtype=float)
    COLS = {"BODY": (1, 3), "WORLD": (2, 4)}
    print(f"\ndead reckoning vs truth over {len(rows)} episodes (dt={a.dt}):")
    print(f"{'':<22}{'final drift':>13}{'max drift':>12}")
    for name, (fi, mi) in COLS.items():
        print(f"  {name+'-frame integ.':<20} {np.median(R[:, fi]):>10.3f} m {np.median(R[:, mi]):>10.3f} m")
    print(f"  (true net displacement median {np.median(R[:, 0]):.3f} m)")

    best_name = min(COLS, key=lambda k: np.median(R[:, COLS[k][0]]))
    drift = float(np.median(R[:, COLS[best_name][0]]))
    print(f"\nbest hypothesis: {best_name}-frame, median final drift {drift:.3f} m")
    verdict = ("velocity integration SUFFICES for the map" if drift < 0.05 else
               "needs periodic relocalisation against the map" if drift < 0.20 else
               "FULL VISUAL ODOMETRY + loop closure required -- dead reckoning is not viable")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
