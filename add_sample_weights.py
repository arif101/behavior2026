"""Add a per-frame `sample_weight` column (float32) to every data parquet of a
LeRobot-v3 root. Default 1.0 everywhere. With --poison poison_windows.json the
rigged-grasp windows of the HUMAN-DEMO root (b1k_radio_map episode indices)
get --poison-weight (default 0.1): rows [closure_row - window_pre,
closure_row + window_post) per episode. Idempotent (--overwrite-col redoes).
Consumed by the trainer via B1K_SAMPLE_WEIGHT_COL=sample_weight
(fork_snapshot/data_loader.py). Run on every source of a Run-3 mix so the
merged schema stays identical across sources.

  python add_sample_weights.py --root /root/b1k_radio_map --poison /root/poison_windows.json
  python add_sample_weights.py --root /root/b1k_radio_factory
"""
import argparse
import glob
import json
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--poison", default=None, help="poison_windows.json (human-demo root only)")
    ap.add_argument("--poison-weight", type=float, default=0.1)
    ap.add_argument("--overwrite-col", action="store_true")
    a = ap.parse_args()
    pw = json.load(open(a.poison)) if a.poison else None
    files = sorted(glob.glob(str(pathlib.Path(a.root) / "data" / "**" / "*.parquet"), recursive=True))
    assert files, f"no data parquets under {a.root}"
    tot = down = 0
    for f in files:
        t = pq.read_table(f)
        if "sample_weight" in t.schema.names:
            if not a.overwrite_col:
                print(f"SKIP {f}: sample_weight present"); continue
            t = t.drop(["sample_weight"])
        ep = t["episode_index"].to_numpy(); fi = t["frame_index"].to_numpy()
        w = np.ones(len(ep), np.float32)
        if pw is not None:
            pre, post = int(pw["window_pre"]), int(pw["window_post"])
            for k, rec in pw["episodes"].items():
                c = int(rec["closure_row"]); m = (ep == int(k)) & (fi >= c - pre) & (fi < c + post)
                w[m] = a.poison_weight
        t = t.append_column("sample_weight", pa.array(w, type=pa.float32()))
        pq.write_table(t, f)
        tot += len(w); down += int((w < 1).sum())
        print(f"{f}: {len(w)} rows, {int((w < 1).sum())} down-weighted")
    print(f"DONE {a.root}: {tot} rows, {down} down-weighted ({100 * down / max(tot, 1):.2f}%)")


main()
