"""Rewrite every data parquet of a LeRobot root with columns in a canonical order (the assembler asserts schema equality
across sources, and pyarrow schema equality is order-sensitive). Columns not in --order are dropped; missing -> error.
  python canonicalize_columns.py --root R --order-from OTHER_ROOT   |   --order c1 c2 ...
"""
import argparse, glob, json, pathlib
import pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--order", nargs="*"); ap.add_argument("--order-from")
a = ap.parse_args(); root = pathlib.Path(a.root)
order = a.order
if a.order_from:
    order = pq.ParquetFile(sorted(glob.glob(f"{a.order_from}/data/**/*.parquet", recursive=True))[0]).schema_arrow.names
n = 0
for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)):
    t = pq.read_table(f); missing = [c for c in order if c not in t.schema.names]
    assert not missing, f"{f}: missing {missing}"
    pq.write_table(t.select(order), f); n += len(t)
info = json.loads((root / "meta/info.json").read_text())
dropped = [c for c in list(info["features"]) if c not in order and not c.startswith("observation.")]
for c in dropped: del info["features"][c]
if dropped: (root / "meta/info.json").write_text(json.dumps(info, indent=4)); print(f"  deregistered dropped columns: {dropped}")
print(f"CANONICAL_OK {root.name} rows={n} cols={len(order)}")
