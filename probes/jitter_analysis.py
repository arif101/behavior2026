"""Is manipulation-phase action variance SEAM-ALIGNED or WITHIN-CHUNK? Five archived failures.

Context: winning config (oracle points + h32) converts 1/6. All failures share one shape:
drive ~600 steps, arrive, manipulate ~2600 steps, never convert. The user's video review of the
one success flagged visible jitter exactly during grasp/button-press. Two candidate mechanisms
with DIFFERENT fixes:

  SEAM-ALIGNED  -> each 32-step replan draws a fresh flow sample; mode-switching mid-grasp.
                   Fix = winner's generation-time inpainting (arXiv 2512.06951).
  WITHIN-CHUNK  -> the sampled chunks are themselves erratic at contact.
                   Fix = sampling/temperature or, more likely, corrective data / RL (the policy's
                   terminal skill is weak: organizers' ckpt 0/10, winner binary 11-12%).

Method per run: split at the arrival point (last step where base_cmd > 0.05, + 50); within the
manipulation phase, compare mean |Δaction| at seam steps (i mod 32 == 31) vs interior steps,
for the ARM dims specifically (7:14, 15:22) since grasp is what fails.
"""

import glob
import json

import numpy as np

runs = sorted(glob.glob("/root/rate/run_*/action_log.jsonl"))
print(f"archived runs: {len(runs)}")

agg_seam, agg_int = [], []
for fp in runs:
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    rows = [r for r in rows if "full" in r]
    if len(rows) < 500:
        continue
    A = np.array([r["full"] for r in rows])
    base = np.linalg.norm(A[:, 0:2], axis=1)
    drive_idx = np.where(base > 0.05)[0]
    arrive = int(drive_idx[-1]) + 50 if len(drive_idx) else 600
    M = A[arrive:]                      # manipulation phase
    if len(M) < 300:
        continue
    arms = np.concatenate([M[:, 7:14], M[:, 15:22]], axis=1)
    d = np.abs(np.diff(arms, axis=0)).mean(axis=1)
    idx = np.arange(len(d)) + arrive    # absolute step index for seam alignment
    seam = d[(idx % 32) == 31]
    interior = d[(idx % 32) != 31]
    r = seam.mean() / max(interior.mean(), 1e-9)
    agg_seam.append(seam.mean()); agg_int.append(interior.mean())
    name = fp.split("/")[-2]
    print(f"  {name}: manip steps {len(M):>4}  arm |Δ| seam {seam.mean():.5f}  "
          f"interior {interior.mean():.5f}  ratio {r:.2f}")

if agg_seam:
    R = np.mean(agg_seam) / max(np.mean(agg_int), 1e-9)
    print(f"\nPOOLED seam/interior ratio (manipulation phase, arm dims): {R:.2f}")
    if R > 3:
        print("READ: SEAM-ALIGNED. Replan mode-switching dominates manipulation jitter.")
        print("      -> build the winner's generation-time inpainting; pre-register rate >=3/5.")
    elif R > 1.5:
        print("READ: MIXED. Seams contribute but within-chunk variance is substantial.")
        print("      -> inpainting helps; expect partial rate recovery. Corrective data still needed.")
    else:
        print("READ: WITHIN-CHUNK. The sampled chunks are erratic at contact independent of seams.")
        print("      -> inpainting will not fix the rate. The terminal skill itself is weak:")
        print("         corrective/perturb-return data (Phase 2) or RL (Phase 3) is the lever.")
