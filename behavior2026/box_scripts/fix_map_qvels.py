"""Surgically fix the corrupted qvel slices in /root/b1k_radio_map's observation.state.

Organizers' 2026-08-08 fix (BEHAVIOR-1K#2325): the non-base qvel fields in
2026-challenge-demos observation.state were garbage (joint-angle-like values, 14-112x
magnitude, ~0.04-0.25 correlation with true rates). Fixed slices:
  arm_left_qvel [10:17], gripper_left_qvel [26:28], arm_right_qvel [35:42],
  gripper_right_qvel [51:53], trunk_qvel [57:61]
Everything else (actions, videos, annotations, non-qvel dims, base_qvel) unchanged.

Our policy input NEVER consumed these dims (configs/robots/b1k.py proprio = base_qvel +
qpos slices only) — this is a HYGIENE fix so the raw column is semantically uniform
across the Run-2 mix (corrective/rac carry live-captured correct qvels) and safe for
any future direct consumer (e.g. ResFiT critic features).

Method: fetch the updated task-0 data shards from the source repo, join rows on
(raw_episode_id-mapped organizer episode_index, frame_index), overwrite ONLY the five
slices in-place. Prints before/after magnitude evidence (frame-0 rest check = the
issue's own diagnostic). Idempotent (second run overwrites with identical values).
"""

import glob
import json
import os
import pathlib

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SRC_REPO = "behavior-1k/2026-challenge-demos"
QVEL_SLICES = [(10, 17), (26, 28), (35, 42), (51, 53), (57, 61)]


def main():
    root = pathlib.Path("/root/b1k_radio_map")
    cache = "/root/orgdata_qvelfix"
    tok = open("/root/.hf_token").read().strip()

    # organizer episodes meta rows for task 0 (reuse the cached meta from the depth pass)
    meta_fps = sorted(glob.glob("/root/orgdepth/meta/episodes/**/*.parquet", recursive=True))
    assert meta_fps, "organizer meta cache missing (run label_map_depth_from_source first)"
    cols = ["task_index", "raw_episode_id", "episode_index",
            "data/chunk_index", "data/file_index"]
    org = pa.concat_tables([pq.read_table(p, columns=cols) for p in meta_fps]).to_pydict()
    sel = [i for i, ti in enumerate(org["task_index"]) if int(ti) == 0]
    raw2orgep = {int(org["raw_episode_id"][i]): int(org["episode_index"][i]) for i in sel}
    shards = sorted({(int(org["data/chunk_index"][i]), int(org["data/file_index"][i]))
                     for i in sel})
    print(f"task-0 organizer rows: {len(sel)}, data shards: {shards}", flush=True)

    # fetch UPDATED shards and index their states by (org_ep, frame)
    org_state = {}
    for ch, fi in shards:
        fn = f"data/chunk-{ch:03d}/file-{fi:03d}.parquet"
        p = hf_hub_download(SRC_REPO, fn, repo_type="dataset", local_dir=cache,
                            token=tok, force_download=True)  # force: must be the FIXED rev
        t = pq.read_table(p, columns=["episode_index", "frame_index", "observation.state"])
        ep = t["episode_index"].to_numpy()
        fr = t["frame_index"].to_numpy()
        st = np.stack(t["observation.state"].to_numpy())
        for e in np.unique(ep):
            m = ep == e
            org_state[int(e)] = st[m][np.argsort(fr[m])]
        print(f"indexed {fn}: {t.num_rows} rows", flush=True)

    # local episode -> organizer episode
    lem = pq.read_table(sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"),
                                         recursive=True))[0]).to_pydict()
    l2o = {int(lem["episode_index"][i]): raw2orgep[int(lem["raw_episode_id"][i])]
           for i in range(len(lem["episode_index"]))}

    changed = kept = 0
    rest_before, rest_after = [], []
    for dfp in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        t = pq.read_table(dfp)
        ep = t["episode_index"].to_numpy()
        fr = t["frame_index"].to_numpy()
        st = np.stack(t["observation.state"].to_numpy()).copy()
        for e in np.unique(ep):
            oe = l2o[int(e)]
            src = org_state[oe]
            m = np.where(ep == e)[0]
            f = fr[m].astype(int)
            assert f.max() < len(src), (int(e), oe, f.max(), len(src))
            frame0 = m[f == 0]
            for a, b in QVEL_SLICES:
                if len(frame0):
                    rest_before.append(float(np.abs(st[frame0[0], a:b]).mean()))
                    rest_after.append(float(np.abs(src[0, a:b]).mean()))
                st[m, a:b] = src[f, a:b]
            changed += len(m)
        # untouched dims must be byte-identical to what we had
        ref = np.stack(t["observation.state"].to_numpy())
        untouched = np.ones(st.shape[1], bool)
        for a, b in QVEL_SLICES:
            untouched[a:b] = False
        assert np.array_equal(st[:, untouched], ref[:, untouched]), "non-qvel dims drifted!"
        kept += int(untouched.sum())
        field = t.schema.field("observation.state")
        arr = pa.array([row for row in st.astype(np.float32)], type=field.type)
        t = t.set_column(t.schema.get_field_index("observation.state"),
                         field, arr)
        pq.write_table(t, dfp)
        print(f"rewrote {dfp} ({len(ep)} rows)", flush=True)

    print(json.dumps({
        "rows_fixed": changed,
        "rest_check_abs_mean_before": round(float(np.mean(rest_before)), 4),
        "rest_check_abs_mean_after": round(float(np.mean(rest_after)), 4),
    }), flush=True)
    print("QVEL_FIX_DONE (rest-check: before should be ~1 rad-scale garbage, after ~0)",
          flush=True)


if __name__ == "__main__":
    main()
