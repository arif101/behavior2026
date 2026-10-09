"""Per-frame sample_weight for the APPROACH source by its `stage` column (S1 packing note, RUN3_PACKING.md 2026-09-15):
stage 1 (pre-lift = the unique pre-contact approach) -> --w-approach (default 2.0); stage >= 2 (the transport tail,
which replays the human's actions already covered by source 2 and the demos) -> --w-tail (default 0.5).
Run AFTER add_sample_weights.py (which registers the column); rewrites the values in place.
  python set_stage_weights.py --root /root/manufactured/b1k_radio_approach_v2
"""
import argparse, glob, pathlib
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--w-approach", type=float, default=2.0); ap.add_argument("--w-tail", type=float, default=0.5)
a = ap.parse_args()
files = sorted(glob.glob(str(pathlib.Path(a.root) / "data" / "**" / "*.parquet"), recursive=True)); assert files
n1 = n2 = 0
for f in files:
    t = pq.read_table(f); assert "stage" in t.schema.names and "sample_weight" in t.schema.names, f
    st = np.asarray(t.column("stage").to_pylist()).reshape(len(t), -1)[:, 0].astype(int)
    w = np.where(st <= 1, a.w_approach, a.w_tail).astype(np.float32)
    n1 += int((st <= 1).sum()); n2 += int((st > 1).sum())
    t = t.drop(["sample_weight"]).append_column("sample_weight", pa.array(w, type=pa.float32()))
    pq.write_table(t, f)
print(f"STAGE_WEIGHTS_OK {a.root}: approach frames {n1} @ {a.w_approach}, tail frames {n2} @ {a.w_tail} (mass {n1*a.w_approach:.0f} vs {n2*a.w_tail:.0f})")
