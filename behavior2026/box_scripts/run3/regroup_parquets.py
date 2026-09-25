"""Rewrite every data parquet of a LeRobot root with row groups of <= --rows rows (default 16384), streaming (iter_batches),
atomic (tmp + os.replace). Why: the arrow parquet reader reconstructs each row group's list columns with int32 offsets, so a
single-row-group file with 229,565 rows x 32,768-wide `hist_tok` (7.5e9 elements) fails with "OSError: List index overflow"
(reproduced on the trainer 2026-09-25; a 24,794-row group = 8.1e8 elements reads fine). Skips files already grouped finely.
  python regroup_parquets.py --root /root/b1k_radio_mix_4d [--rows 16384]
"""
import argparse, glob, os, pathlib, time
import pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--rows", type=int, default=16384)
a = ap.parse_args(); t0 = time.time(); total = 0
for f in sorted(glob.glob(str(pathlib.Path(a.root) / "data/**/*.parquet"), recursive=True)):
    pf = pq.ParquetFile(f); md = pf.metadata
    if max(md.row_group(i).num_rows for i in range(md.num_row_groups)) <= a.rows:
        print(f"  {pathlib.Path(f).name}: {md.num_rows} rows in {md.num_row_groups} groups already <= {a.rows}, skipped", flush=True); total += md.num_rows; continue
    n = 0
    with pq.ParquetWriter(f + ".tmp", pf.schema_arrow) as w:
        for b in pf.iter_batches(batch_size=a.rows):
            w.write_batch(b); n += b.num_rows
    assert n == md.num_rows, (f, n, md.num_rows)
    md2 = pq.ParquetFile(f + ".tmp").metadata; assert md2.num_rows == n and md2.num_row_groups >= (n + a.rows - 1) // a.rows, (n, md2.num_row_groups)
    os.replace(f + ".tmp", f); total += n
    print(f"  {pathlib.Path(f).name}: {n} rows -> {md2.num_row_groups} row groups, {time.time()-t0:.0f}s", flush=True)
print(f"REGROUP_OK {pathlib.Path(a.root).name}: {total} rows, {time.time()-t0:.0f}s", flush=True)
