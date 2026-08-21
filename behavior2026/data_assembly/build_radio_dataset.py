"""Box-side: fetch radio's LeRobot shards and join per-frame `target_points` into them.

One command turns the replayed pose data into a trainable dataset:

    python build_radio_dataset.py --points radio_points_200.npz --out /root/b1k_radio

Contract (must match scripts/b1k/add_target_points.py and B1KInputs, which reshape to (2,3)):
    target_points       float32, flat list[6]  ->  [left_xyz, right_xyz]
    target_points_mask  bool,    list[2]       ->  False = that arm has no target this frame

The value is the target in the BASE frame, not a world coordinate. A world point is unusable at
eval (robot global pose is prohibited by the rules) and is the wrong quantity anyway — the +25-46
point gains in arXiv:2606.27663 came from a displacement.

WHY THIS EXISTS
---------------
Phase A carried real points on 2.67% of frames, so the channel was masked on 36 of every 37
samples. The action-expert probe reads R2 0.369 for decoding the point but 0.046 for decoding the
future action: present, never converted to intent. This raises coverage to ~100% on 200 episodes /
430k frames of turning_on_radio.

THE HAZARD IS FRAME ALIGNMENT
-----------------------------
Replay produced 1957 frames where the parquet had 1956. An off-by-one silently shifts every label
by one timestep across the whole dataset — and it would look completely plausible. So we do not
assume: per episode we assert the length discrepancy is within --max_skew, truncate to the common
length, and record the skew. Anything larger is dropped loudly, not fudged.

This is where all three of the G3-era silent-corruption bugs happened (see
reference_g3_conversion_bugs), plus four more in today's batch runner. Every one produced
plausible-looking output rather than an error.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "behavior-1k/2026-challenge-demos"
TASK_INDEX = 0  # turning_on_radio


def fetch(files: list[str], local_dir: str, token: str | None) -> None:
    """Download with Xet high-performance ON. Measured ~430 MB/s vs 12 MB/s without it — the old
    HF_HUB_ENABLE_HF_TRANSFER flag is deprecated and silently ignored on huggingface_hub>=1.x."""
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import hf_hub_download

    for f in files:
        if not os.path.exists(os.path.join(local_dir, f)):
            hf_hub_download(REPO, f, repo_type="dataset", local_dir=local_dir, token=token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="npz from build_target_points.py")
    ap.add_argument("--out", required=True, help="output LeRobot root")
    ap.add_argument("--src", default="/root/b1k_src", help="where to stage the fetched shards")
    ap.add_argument("--token", default="/root/.hf_token")
    ap.add_argument("--max_skew", type=int, default=4,
                    help="max |len(replay) - len(parquet)| tolerated per episode")
    ap.add_argument("--videos", action="store_true", help="also fetch video shards (~32 GB)")
    a = ap.parse_args()

    tok = open(a.token).read().strip() if os.path.exists(a.token) else None
    pts = np.load(a.points)
    have = sorted({int(k[2:].split("_")[0]) for k in pts.files})
    print(f"points file: {len(have)} episodes")

    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    files = api.list_repo_files(REPO, repo_type="dataset")

    # meta/episodes tells us which data shard each episode lives in.
    metas = sorted(f for f in files if f.startswith("meta/episodes/"))
    fetch(metas + ["meta/tasks.parquet", "meta/info.json"], a.src, tok)
    ep = pd.concat([pd.read_parquet(os.path.join(a.src, m)) for m in metas])

    # demo_id -> episode_index. demo_id encodes task: task = demo_id // 10000 (replay_obs.py:39),
    # and within a task the episodes are the task's slice in episode_index order.
    task_eps = ep[ep["tasks"].astype(str).str.contains("turning_on_radio", na=False)]
    if task_eps.empty:                      # fall back to task_index if `tasks` is an id column
        tasks = pd.read_parquet(os.path.join(a.src, "meta/tasks.parquet"))
        name = {int(v): str(k) for k, v in tasks["task_index"].items()}
        task_eps = ep[ep["episode_index"].map(lambda e: name.get(TASK_INDEX)) == "turning_on_radio"]
    print(f"radio episodes in meta: {len(task_eps)}")

    shards = sorted({(int(r["data/chunk_index"]), int(r["data/file_index"]))
                     for _, r in task_eps.iterrows()})
    shard_files = [f"data/chunk-{c:03d}/file-{f:03d}.parquet" for c, f in shards]
    print(f"fetching {len(shard_files)} data shards…")
    fetch(shard_files, a.src, tok)

    if a.videos:
        vids = [f for f in files if f.startswith("videos/")
                and any(f"chunk-{c:03d}/file-{f_:03d}" in f for c, f_ in shards)]
        print(f"fetching {len(vids)} video shards (~32 GB)…")
        fetch(vids, a.src, tok)

    # episode_index -> demo_id. NOT demo_id = 10*(episode_index+1): radio's demo_ids are SPARSE
    # (10, 20, ... 3000 with gaps), while episode_index is dense. Assuming a dense mapping joined
    # only 9 of 200 episodes with frame skews from -2766 to +2544 — i.e. it was pairing unrelated
    # episodes. The correct mapping is positional: the k-th radio episode_index corresponds to the
    # k-th radio demo_id, both in sorted order.
    ep_idx_sorted = sorted(int(x) for x in task_eps["episode_index"].unique())
    demo_sorted = sorted(have)
    if len(ep_idx_sorted) != len(demo_sorted):
        print(f"  NOTE: {len(ep_idx_sorted)} episodes in meta vs {len(demo_sorted)} replayed; "
              f"pairing the first {min(len(ep_idx_sorted), len(demo_sorted))}")
    idx2demo = dict(zip(ep_idx_sorted, demo_sorted))

    os.makedirs(a.out, exist_ok=True)
    report = {"episodes": [], "dropped": [], "skew": {}}
    n_frames = n_masked = 0

    for sf in shard_files:
        t = pq.read_table(os.path.join(a.src, sf))
        df = t.to_pandas()
        tp = np.zeros((len(df), 6), dtype=np.float32)
        tm = np.zeros((len(df), 2), dtype=bool)

        for e in sorted(df["episode_index"].unique()):
            sel = df["episode_index"].values == e
            n_par = int(sel.sum())
            demo = idx2demo.get(int(e))
            if demo is None:
                continue          # episode_index not part of this task
            key = f"ep{demo}_pts"
            if key not in pts:
                report["dropped"].append({"episode": int(e), "why": "no replayed points"})
                continue
            P, M = pts[key], pts[f"ep{demo}_mask"]
            skew = len(P) - n_par
            report["skew"][str(demo)] = int(skew)
            if abs(skew) > a.max_skew:
                report["dropped"].append({"episode": int(e), "why": f"skew {skew}"})
                continue
            n = min(len(P), n_par)
            idx = np.where(sel)[0][:n]
            # LEFT arm carries the target; right arm has none for this task.
            tp[idx, 0:3] = P[:n]
            tm[idx, 0] = M[:n]
            n_frames += n
            n_masked += int(M[:n].sum())
            report["episodes"].append(int(demo))

        t = t.append_column("target_points", pa.array(list(tp)))
        t = t.append_column("target_points_mask", pa.array(list(tm)))
        dst = os.path.join(a.out, sf)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        pq.write_table(t, dst)
        print(f"  wrote {dst}  rows={t.num_rows}")

    cov = 100.0 * n_masked / max(n_frames, 1)
    report["coverage_pct"] = cov
    json.dump(report, open(os.path.join(a.out, "join_report.json"), "w"), indent=1)

    print(f"\njoined {len(report['episodes'])} episodes, dropped {len(report['dropped'])}")
    print(f"COVERAGE: {cov:.1f}%   (Phase A was 2.67%)")
    sk = np.array(list(report["skew"].values()))
    if sk.size:
        print(f"frame skew: median {np.median(sk):+.0f}  min {sk.min():+d}  max {sk.max():+d}")
    if cov < 95.0:
        print("  !! coverage below 95% — do NOT train on this; investigate before spending GPU")


if __name__ == "__main__":
    main()
