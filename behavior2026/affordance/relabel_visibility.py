"""Relabel visibility with the body-geometry-aware asymmetric window + store the depth gap.

Finding (diag_depth_gap.py): the togglebutton metalink sits INSIDE the radio at mid-height
(11.75 cm half-height; offset z -1.25 cm), so label depth exceeds visible-surface depth by
~12 cm whenever the camera looks at the body. The old symmetric 15 cm tolerance sat ON the
bias -> visibility collapsed in side-view episodes and eval measured surface-vs-interior.

New rule on gap = z_label - measured(label pixel):
    -0.05 < gap < 0.25   -> VISIBLE (pixel on the radio body; usable positive)
    gap >= 0.25          -> occluded by another object
    gap <= -0.05         -> pixel off-body (background) -> negative
Appends gap as a new column (col 22). No video re-decode: uses the stored depth PNGs.
"""

import glob

import numpy as np
from PIL import Image

tot = vis = 0
for f in sorted(glob.glob("/root/aff_data/ep*_labels.npz")):
    demo = int(f.split("ep")[-1].split("_")[0])
    rows = np.load(f)["rows"]
    out = np.zeros((len(rows), rows.shape[1] + 1), np.float32)
    out[:, :-1] = rows
    for i, r in enumerate(rows):
        t, u, v, z = int(r[0]), r[1], r[2], r[3]
        gap = np.nan
        newv = 0.0
        if z > 0.05 and 0 <= u < 720 and 0 <= v < 720:
            dep = np.asarray(Image.open(f"/root/aff_data/frames/ep{demo}_f{t}_d.png"),
                             np.float32) / 1000.0
            meas = dep[min(int(v / 4), dep.shape[0] - 1), min(int(u / 4), dep.shape[1] - 1)]
            if np.isfinite(meas) and meas > 0.05:
                gap = z - meas
                newv = float(-0.05 < gap < 0.25)
        out[i, 4] = newv
        out[i, -1] = gap if np.isfinite(gap) else 99.0
    np.savez_compressed(f, rows=out)
    tot += len(out)
    vis += int(out[:, 4].sum())
print(f"relabeled: {tot} samples, visible {vis} ({vis / tot:.0%}) [was 15753 = 29%]")
