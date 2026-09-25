"""Write per-row columns from an assembled mix back into its SOURCE roots (the assembler concatenates sources in order,
shifting episode_index by cumulative episode counts and preserving frame order), plus `cam_pose` from the FK arrays
keyed by the MIX (episode_index, frame_index). Verifies per-episode frame counts. Registers features in each source's
meta/info.json.
  python unassemble_columns.py --mix /root/mixes_bk/mix_full --fk /root/mixes_bk/fk --sources MAP FAC EPI [APP] --cols stage_v2 progress toggled target_points_v2 gist_head hist_geo
"""
import argparse, glob, json, pathlib
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser(); ap.add_argument("--mix", required=True); ap.add_argument("--fk"); ap.add_argument("--sources", nargs="+", required=True)
ap.add_argument("--cols", nargs="+", default=["stage_v2", "progress", "toggled", "target_points_v2", "gist_head", "hist_geo"])
a = ap.parse_args(); mix = pathlib.Path(a.mix)
info_mix = json.loads((mix / "meta/info.json").read_text()); feats = info_mix["features"]
mfiles = sorted(glob.glob(str(mix / "data/**/*.parquet"), recursive=True))
cols = ["episode_index", "frame_index"] + a.cols
mt = pa.concat_tables([pq.read_table(f, columns=cols) for f in mfiles])
mep = mt["episode_index"].to_numpy(); mfr = mt["frame_index"].to_numpy()
fk = None
if a.fk:
    cp = np.load(f"{a.fk}/cam_pose.npy").astype(np.float32); ix = np.load(f"{a.fk}/index.npy")
    fk = {(int(e), int(f)): i for i, (e, f) in enumerate(ix)}
lut = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(mep, mfr))}
ep_off = 0
for s in a.sources:
    root = pathlib.Path(s); info = json.loads((root / "meta/info.json").read_text())
    sfiles = sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)); n_ep = info["total_episodes"]; n_rows = 0; miss = 0
    for f in sfiles:
        t = pq.read_table(f); ep = t["episode_index"].to_numpy(); fr = t["frame_index"].to_numpy()
        rows = np.array([lut.get((int(e) + ep_off, int(r)), -1) for e, r in zip(ep, fr)]); miss += int((rows < 0).sum())
        rows = np.where(rows < 0, 0, rows)
        for c in a.cols:
            if c in t.schema.names: t = t.drop([c])
            t = t.append_column(c, mt[c].take(pa.array(rows)))
        if fk is not None:
            fr_rows = np.array([fk.get((int(e) + ep_off, int(r)), -1) for e, r in zip(ep, fr)]); miss += int((fr_rows < 0).sum())
            vals = cp[np.where(fr_rows < 0, 0, fr_rows)]
            if "cam_pose" in t.schema.names: t = t.drop(["cam_pose"])
            t = t.append_column("cam_pose", pa.array(vals.tolist(), type=pa.list_(pa.float32(), 21)))
        pq.write_table(t, f); n_rows += len(t)
    for c in a.cols: info["features"][c] = feats[c]
    if fk is not None: info["features"]["cam_pose"] = {"dtype": "float32", "shape": [21], "names": None}
    (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"UNASSEMBLED {root.name}: episodes {n_ep} (mix offset {ep_off}) rows {n_rows} missing {miss}", flush=True)
    assert miss == 0, f"{root}: {miss} rows had no mix counterpart (ordering assumption broken)"
    ep_off += n_ep
print("UNASSEMBLE_OK")
