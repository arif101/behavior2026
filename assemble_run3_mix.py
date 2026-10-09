"""Assemble the RUN-3 training mix: b1k_radio_map (+sample_weight poison windows) + b1k_radio_factory (honest grasp segments) + b1k_radio_episodes (grasp + learned press)
-> one merged LeRobot-v3 root (/root/b1k_radio_run3mix). Schema-exactness (verified at
each source's conversion) is what makes this a mechanical merge: no re-encoding, no row
rewrites beyond index columns.

Merge rules:
  data     one output parquet per source data file, episode_index/index shifted by the
           running offsets; column set asserted identical to source 0.
  videos   files are copied/hardlinked to sequential file indices per stream; per-file
           from/to timestamps are file-relative and stay VALID unchanged.
  episodes concat with re-indexed episode_index + remapped data/video file indices.
  info     totals summed; splits = {"train": "0:N"}.
  provenance/sampling  meta/run3_sources.json records per-episode source + episode range
           + the suggested sampler weights (consumed at launch by the stage-oversample
           machinery; source weights are ADVISORY here, decided at launch):
             map=1.0 baseline; corrective: weight so corrective frames ~ match the
             map's stage-1->2 transition-neighborhood mass (printed, computed from real
             counts); rac: 2.0 (far-field approach-recovery, plentiful in base too).

Run on the TRAINING box at bring-up (map videos don't exist on the A5000). --dry-run
validates schemas/counts and prints the plan without writing (works on the A5000 for
the corrective+rac sources; map passes data checks, videos flagged VERIFY-ON-TRAINER).
Every source must already carry gt_depth_ds (depth_aux repack KeyErrors otherwise).
"""

import argparse
import glob
import json
import os
import pathlib
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

VIDEO_KEYS = [
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
    "observation.depth_linear.zed_link_camera_0",
    "observation.depth_linear.left_realsense_link_camera_0",
    "observation.depth_linear.right_realsense_link_camera_0",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+",
                    default=["/root/b1k_radio_map", "/root/b1k_radio_factory",
                             "/root/b1k_radio_episodes"])
    ap.add_argument("--out", default="/root/b1k_radio_run3mix")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    if out.exists() and not a.dry_run:
        if not a.overwrite:
            raise SystemExit(f"{out} exists (pass --overwrite)")
        shutil.rmtree(out)

    # ---- validate sources ---------------------------------------------------------------
    srcs = []
    ref_schema = None
    for s in a.sources:
        root = pathlib.Path(s)
        dfs = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
        assert dfs, f"{s}: no data parquets"
        sch = pq.ParquetFile(dfs[0]).schema_arrow
        if ref_schema is None:
            ref_schema = sch
        else:
            assert sch.equals(ref_schema), f"{s}: data schema differs from {a.sources[0]}"
        assert "gt_depth_ds" in sch.names, f"{s}: missing gt_depth_ds (run depth labels first)"
        assert "sample_weight" in sch.names, f"{s}: missing sample_weight (run add_sample_weights.py first)"
        em = sorted(glob.glob(str(root / "meta" / "episodes" / "**" / "*.parquet"),
                    recursive=True))
        assert em, f"{s}: no episodes meta"
        info = json.loads((root / "meta" / "info.json").read_text())
        vids = {}
        for vk in VIDEO_KEYS:
            vfs = sorted(glob.glob(str(root / "videos" / vk / "**" / "*.mp4"),
                                   recursive=True))
            vids[vk] = vfs
        n_vid = {k: len(v) for k, v in vids.items()}
        missing_vids = [k for k, n in n_vid.items() if n == 0]
        srcs.append({"root": root, "data": dfs, "emeta": em[0], "info": info,
                     "vids": vids, "missing_vids": missing_vids})
        print(f"{s}: {info['total_episodes']} eps / {info['total_frames']} frames, "
              f"data files {len(dfs)}, video files/stream {sorted(set(n_vid.values()))}"
              + (f"  !! NO VIDEO FILES for {len(missing_vids)} streams "
                 "(VERIFY-ON-TRAINER)" if missing_vids else ""), flush=True)
        if missing_vids and not a.dry_run:
            raise SystemExit(f"{s}: missing videos — full assembly must run where videos exist")

    total_eps = sum(s["info"]["total_episodes"] for s in srcs)
    total_frames = sum(s["info"]["total_frames"] for s in srcs)
    print(f"MERGED: {total_eps} episodes / {total_frames} frames", flush=True)

    # ---- sampler-weight advisory (from real counts) --------------------------------------
    tstats = []
    for s in srcs:
        st = np.concatenate([pq.read_table(f, columns=["stage"])["stage"].to_numpy()
                             for f in s["data"]])
        trans = int(np.sum(np.abs(np.diff((st >= 2).astype(np.int8))) > 0))
        tstats.append({"frames": len(st), "stage12_transitions": trans})
    base_neigh = tstats[0]["stage12_transitions"] * 50  # +/-25-frame neighborhoods
    corr_frames = tstats[1]["frames"] if len(tstats) > 1 else 0
    w_corr = round(max(1.0, base_neigh / max(corr_frames, 1)), 2)
    weights = {"map": 1.0, "factory": w_corr, "episodes": 2.0}
    print(f"advisory sampler weights: {weights} "
          f"(base transition-neighborhood ~{base_neigh} frames vs corrective {corr_frames})",
          flush=True)

    if a.dry_run:
        print("DRY-RUN: no writes. Plan validated.", flush=True)
        return

    # ---- data ---------------------------------------------------------------------------
    (out / "data" / "chunk-000").mkdir(parents=True)
    ep_off = row_off = fidx = 0
    src_records = []
    for si, s in enumerate(srcs):
        first_ep = ep_off
        for f in s["data"]:
            t = pq.read_table(f)
            ep = pa.array(t["episode_index"].to_numpy() + ep_off, type=t.schema.field("episode_index").type)
            ix = pa.array(t["index"].to_numpy() + row_off, type=t.schema.field("index").type)
            t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index", ep)
            t = t.set_column(t.schema.get_field_index("index"), "index", ix)
            pq.write_table(t, out / "data" / "chunk-000" / f"file-{fidx:03d}.parquet")
            fidx += 1
        # episodes meta for this source
        emt = pq.read_table(s["emeta"]).to_pydict()
        n = len(emt["episode_index"])
        src_records.append({"source": str(s["root"]), "episodes": [first_ep, first_ep + n - 1],
                            "n_frames": s["info"]["total_frames"]})
        s["_emeta_dict"], s["_ep_off"], s["_row_off"] = emt, ep_off, row_off
        ep_off += n
        row_off += s["info"]["total_frames"]

    # ---- videos (hardlink, sequential per-stream file indices) ---------------------------
    vid_off = {vk: 0 for vk in VIDEO_KEYS}
    src_vid_off = []
    for s in srcs:
        src_vid_off.append(dict(vid_off))
        for vk in VIDEO_KEYS:
            for f in s["vids"][vk]:
                dst = out / "videos" / vk / "chunk-000" / f"file-{vid_off[vk]:03d}.mp4"
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy(f, dst)
                vid_off[vk] += 1

    # ---- episodes meta ------------------------------------------------------------------
    merged = None
    for si, s in enumerate(srcs):
        d = s["_emeta_dict"]
        n = len(d["episode_index"])
        d["episode_index"] = [int(v) + s["_ep_off"] for v in d["episode_index"]]
        for k in ("dataset_from_index", "dataset_to_index"):
            d[k] = [int(v) + s["_row_off"] for v in d[k]]
        d["data/file_index"] = [0] * n  # rewritten below per-source sequentially
        # data files were written sequentially; map each source's original file idx
        base_fidx = sum(len(x["data"]) for x in srcs[:si])
        d["data/file_index"] = [base_fidx + int(v) for v in s["_emeta_dict"].get(
            "data/file_index", [0] * n)]
        d["data/chunk_index"] = [0] * n
        for vk in VIDEO_KEYS:
            ck, fk = f"videos/{vk}/chunk_index", f"videos/{vk}/file_index"
            if fk in d:
                d[fk] = [src_vid_off[si][vk] + int(v) for v in d[fk]]
                d[ck] = [0] * n
        if merged is None:
            merged = {k: list(v) for k, v in d.items()}
        else:
            for k in merged:
                merged[k] += list(d[k])
    ref_ep_schema = pq.ParquetFile(srcs[0]["emeta"]).schema_arrow
    import pandas as pd
    edf = pd.DataFrame({name: merged[name] for name in ref_ep_schema.names})
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    edf.to_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    # ---- info / tasks / provenance ------------------------------------------------------
    info = json.loads(json.dumps(srcs[0]["info"]))
    info["total_episodes"] = total_eps
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{total_eps}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    shutil.copy(srcs[0]["root"] / "meta" / "tasks.parquet", out / "meta" / "tasks.parquet")
    (out / "meta" / "run3_sources.json").write_text(json.dumps(
        {"sources": src_records, "advisory_weights": weights,
         "transition_stats": tstats}, indent=2))
    print(f"ASSEMBLED {out}: {total_eps} eps / {total_frames} frames", flush=True)
    print("NEXT: recompute norm stats over the mix, then pi05_radio_run2 + "
          "B1K_STAGE_OVERSAMPLE=8", flush=True)


if __name__ == "__main__":
    main()
