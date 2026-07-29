"""Discriminate the 13 cm depth-gap source: camera link-vs-optical offset OR metalink-in-body.

gap(t) = z_label − measured_depth(label pixel). Overlays show pixels are RIGHT but 3D is off
13 cm, tightly clustered. Hypotheses:
  A) robot2cam_pose is the zed LINK frame; optical center sits ~13 cm along the view axis.
     Prediction: gap ≈ CONSTANT across episodes, ranges, and viewing directions.
  B) The togglebutton metalink sits inside the radio body behind the visible surface.
     Prediction: gap varies with viewing direction / episode (radio yaw differs per init).

Runs on already-extracted data (labels npz + depth PNGs). Also fits the best along-axis
correction c and reports the residual after applying it.
"""

import glob

import numpy as np
from PIL import Image

gaps, zs, eps, uvs = [], [], [], []
for f in sorted(glob.glob("/root/aff_data/ep*_labels.npz")):
    demo = int(f.split("ep")[-1].split("_")[0])
    rows = np.load(f)["rows"]
    for r in rows:
        if r[4] < 0.5:          # visible only
            continue
        t, u, v, z = int(r[0]), r[1], r[2], r[3]
        dep = np.asarray(Image.open(f"/root/aff_data/frames/ep{demo}_f{t}_d.png"),
                         np.float32) / 1000.0
        meas = dep[min(int(v / 4), dep.shape[0] - 1), min(int(u / 4), dep.shape[1] - 1)]
        if not np.isfinite(meas) or meas < 0.05:
            continue
        gaps.append(z - meas)
        zs.append(z)
        eps.append(demo)
        uvs.append((u, v))

gaps, zs, eps = np.array(gaps), np.array(zs), np.array(eps)
uvs = np.array(uvs)
print(f"n={len(gaps)} visible samples with finite depth")
print(f"gap: median {np.median(gaps)*100:.1f} cm  mean {gaps.mean()*100:.1f}  "
      f"std {gaps.std()*100:.1f}  p10 {np.percentile(gaps,10)*100:.1f}  "
      f"p90 {np.percentile(gaps,90)*100:.1f}")

# range dependence: constant (additive offset) vs proportional (scale error)
for lo, hi in ((0.3, 0.7), (0.7, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0)):
    m = (zs >= lo) & (zs < hi)
    if m.sum() > 20:
        print(f"  z in [{lo},{hi}): gap median {np.median(gaps[m])*100:6.1f} cm  (n={m.sum()})")

# ratio test: if multiplicative, gap/z is constant instead
ratio = gaps / zs
print(f"gap/z: median {np.median(ratio):.4f}  std {ratio.std():.4f}")

# per-episode spread: geometry hypothesis predicts inter-episode variation (different yaws)
ep_med = {}
for e in np.unique(eps):
    m = eps == e
    if m.sum() >= 10:
        ep_med[e] = np.median(gaps[m])
med = np.array(list(ep_med.values()))
print(f"per-episode median gap over {len(med)} eps: spread p10 {np.percentile(med,10)*100:.1f} "
      f"p90 {np.percentile(med,90)*100:.1f} cm  std {med.std()*100:.2f} cm")

# image-position dependence (off-axis pixels see an along-axis offset differently)
cx, cy = 364.7, 356.2
r_pix = np.linalg.norm(uvs - [cx, cy], axis=1)
for lo, hi in ((0, 120), (120, 240), (240, 400)):
    m = (r_pix >= lo) & (r_pix < hi)
    if m.sum() > 20:
        print(f"  pixel radius [{lo},{hi}): gap median {np.median(gaps[m])*100:6.1f} cm  (n={m.sum()})")

c = float(np.median(gaps))
print(f"\nbest along-axis correction c = {c*100:.2f} cm; residual |gap - c|: "
      f"median {np.median(np.abs(gaps - c))*100:.2f} cm  p90 "
      f"{np.percentile(np.abs(gaps - c), 90)*100:.2f} cm")
