"""Restore b1k_radio_map's videos on a training box from the ORGANIZERS' consolidated repo.

The HF backup of b1k_radio_map is meta+data only (videos died with the July box). Rather
than re-encoding a reconstruction of the original local 8-file-per-stream layout, this
fetches the organizer video files that CONTAIN the 200 radio episodes (all 6 streams),
renumbers them densely per stream (file-000..N-1 in sorted organizer order), and REWRITES
the episodes-meta video columns (chunk/file/from/to) to point at them via the
raw_episode_id join against the organizers' meta/episodes. Data parquets and every label
column are untouched; from/to timestamps are the organizer within-file values (content
unchanged by renaming). Non-radio episodes sharing those files are dead weight on disk.

Builds /root/b1k_radio_map as: data -> symlink into the backup snapshot, meta = copied +
rewritten, videos = fetched. Idempotent-ish: re-running refetches nothing already present.

Run on the trainer AFTER the backup snapshot is in /root/backup (fetch_stage1).
"""

import glob
import json
import os
import pathlib
import shutil

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SRC_REPO = "behavior-1k/2026-challenge-demos"
BACKUP_MAP = pathlib.Path("/root/backup/b1k_radio_map")
OUT = pathlib.Path("/root/b1k_radio_map")
CACHE = "/root/orgvid_cache"
BUDGET_GB = 120.0
VIDEO_KEYS = [
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
    "observation.depth_linear.zed_link_camera_0",
    "observation.depth_linear.left_realsense_link_camera_0",
    "observation.depth_linear.right_realsense_link_camera_0",
]


def main():
    tok = open("/root/.hf_token").read().strip()

    # organizer episodes meta
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    meta_files = [f for f in api.list_repo_files(SRC_REPO, repo_type="dataset")
                  if f.startswith("meta/episodes/") and f.endswith(".parquet")]
    cols = ["task_index", "raw_episode_id"] + [
        f"videos/{k}/{s}" for k in VIDEO_KEYS
        for s in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    tabs = []
    for fn in meta_files:
        p = hf_hub_download(SRC_REPO, fn, repo_type="dataset", local_dir=CACHE, token=tok)
        t = pq.read_table(p)
        tabs.append(t.select([c for c in cols if c in t.schema.names]))
    org = pa.concat_tables(tabs).to_pydict()
    sel = [i for i, ti in enumerate(org["task_index"]) if int(ti) == 0]
    org_by_raw = {int(org["raw_episode_id"][i]): i for i in sel}
    print(f"organizer task-0 rows: {len(sel)}", flush=True)

    # local episodes meta
    lem_fp = sorted(glob.glob(str(BACKUP_MAP / "meta" / "episodes" / "**" / "*.parquet"),
                              recursive=True))[0]
    lem = pq.read_table(lem_fp)
    lem_d = lem.to_pydict()
    n_eps = len(lem_d["episode_index"])

    # per stream: unique organizer (chunk,file) -> dense new index; fetch
    total_b = 0
    remap = {}
    for k in VIDEO_KEYS:
        pairs = sorted({(int(org[f"videos/{k}/chunk_index"][org_by_raw[int(r)]]),
                         int(org[f"videos/{k}/file_index"][org_by_raw[int(r)]]))
                        for r in lem_d["raw_episode_id"]})
        remap[k] = {pr: new for new, pr in enumerate(pairs)}
        print(f"{k}: {len(pairs)} organizer files", flush=True)
        for (ch, fi) in pairs:
            fn = f"videos/{k}/chunk-{ch:03d}/file-{fi:03d}.mp4"
            p = hf_hub_download(SRC_REPO, fn, repo_type="dataset", local_dir=CACHE, token=tok)
            total_b += os.path.getsize(p)
            if total_b / 1e9 > BUDGET_GB:
                raise SystemExit(f"video budget exceeded ({total_b / 1e9:.1f} GB)")
            dst = OUT / "videos" / k / "chunk-000" / f"file-{remap[k][(ch, fi)]:03d}.mp4"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                os.link(p, dst)
    print(f"fetched/linked {total_b / 1e9:.2f} GB of organizer video", flush=True)

    # rewrite local episodes meta video columns
    for k in VIDEO_KEYS:
        ck, fk = f"videos/{k}/chunk_index", f"videos/{k}/file_index"
        ft, tt = f"videos/{k}/from_timestamp", f"videos/{k}/to_timestamp"
        new_c, new_f, new_ft, new_tt = [], [], [], []
        for i in range(n_eps):
            oi = org_by_raw[int(lem_d["raw_episode_id"][i])]
            pr = (int(org[f"videos/{k}/chunk_index"][oi]), int(org[f"videos/{k}/file_index"][oi]))
            new_c.append(0)
            new_f.append(remap[k][pr])
            new_ft.append(float(org[f"videos/{k}/from_timestamp"][oi]))
            new_tt.append(float(org[f"videos/{k}/to_timestamp"][oi]))
        for col, vals, typ in ((ck, new_c, pa.int64()), (fk, new_f, pa.int64()),
                               (ft, new_ft, pa.float64()), (tt, new_tt, pa.float64())):
            idx = lem.schema.get_field_index(col)
            lem = lem.set_column(idx, lem.schema.field(col), pa.array(vals, type=lem.schema.field(col).type))

    # build the output tree: data symlink, meta copy (rewritten), videos already placed
    OUT.mkdir(exist_ok=True)
    if not (OUT / "data").exists():
        os.symlink(BACKUP_MAP / "data", OUT / "data")
    (OUT / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for f in glob.glob(str(BACKUP_MAP / "meta" / "*")):
        if os.path.isfile(f):
            shutil.copy(f, OUT / "meta" / os.path.basename(f))
    pq.write_table(lem, OUT / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    info = json.loads((OUT / "meta" / "info.json").read_text())
    print(f"meta rewritten: {n_eps} episodes, info totals "
          f"{info['total_episodes']}/{info['total_frames']}", flush=True)
    print("MAP_VIDEOS_RESTORED", flush=True)


if __name__ == "__main__":
    main()
