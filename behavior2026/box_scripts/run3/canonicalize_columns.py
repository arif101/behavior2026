"""Rewrite every data parquet of a LeRobot root with columns in a canonical order (the assembler asserts schema equality
across sources, and pyarrow schema equality is order-sensitive). Columns not in --order are dropped (and deregistered);
missing -> error. STREAMING (2026-09-29): reads with iter_batches and writes 16k-row row groups via ParquetWriter, because a
file carrying `hist_tok` (32768-wide) as ONE row group of 229k rows cannot be read whole (arrow int32 list-offset
overflow, "OSError: List index overflow") — unassemble_columns.py writes exactly such files for the map root.
  python canonicalize_columns.py --root R --order-from OTHER_ROOT   |   --order c1 c2 ...
"""
import argparse, glob, json, os, pathlib
import pyarrow as pa, pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--order", nargs="*"); ap.add_argument("--order-from")
ap.add_argument("--rows", type=int, default=16384)
a = ap.parse_args(); root = pathlib.Path(a.root)
order = a.order
if a.order_from:
    order = pq.ParquetFile(sorted(glob.glob(f"{a.order_from}/data/**/*.parquet", recursive=True))[0]).schema_arrow.names
n = 0
for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)):
    pf = pq.ParquetFile(f); names = pf.schema_arrow.names; missing = [c for c in order if c not in names]
    assert not missing, f"{f}: missing {missing}"
    schema = pa.schema([pf.schema_arrow.field(c) for c in order]); rows = 0
    with pq.ParquetWriter(f + ".tmp", schema) as w:
        for b in pf.iter_batches(batch_size=a.rows, columns=order):
            w.write_batch(pa.RecordBatch.from_arrays([b.column(c) for c in order], schema=schema)); rows += b.num_rows
    assert rows == pf.metadata.num_rows, (f, rows, pf.metadata.num_rows)
    os.replace(f + ".tmp", f); n += rows
info = json.loads((root / "meta/info.json").read_text())
dropped = [c for c in list(info["features"]) if c not in order and not c.startswith("observation.")]
for c in dropped: del info["features"][c]
if dropped: (root / "meta/info.json").write_text(json.dumps(info, indent=4)); print(f"  deregistered dropped columns: {dropped}")
print(f"CANONICAL_OK {root.name} rows={n} cols={len(order)} (row groups <= {a.rows})")
