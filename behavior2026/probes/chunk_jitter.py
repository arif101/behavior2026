"""Is the visible jitter coming from receding-horizon chunk boundaries?

The emitted actions are smooth on average (step-to-step change <=13% of the training std) and
in-distribution (mean |z| 0.64), so the jitter the user sees is NOT command noise. Two candidates
remain: (a) discontinuities where the policy replans -- serve_b1k defaults --action-horizon 16, so
every 16th step starts a fresh chunk; (b) the controller failing to track absolute joint targets.

This separates (a): if the per-step delta spikes at multiples of the replan period, the stitching
is the culprit and the fix is temporal ensembling / blending across chunk boundaries.
"""

import json

import numpy as np

rows = [json.loads(line) for line in open("/root/action_log.jsonl")]
rows = [r for r in rows if "full" in r]
A = np.array([r["full"] for r in rows])
diff = np.abs(np.diff(A, axis=0)).mean(axis=1)   # mean |delta| across dims, per step

print(f"steps {len(A)}   mean |delta| {diff.mean():.5f}   max {diff.max():.5f}")
print()
for period in (8, 16, 32, 50):
    idx = np.arange(len(diff))
    on = diff[idx % period == period - 1]
    off = diff[idx % period != period - 1]
    if len(on) < 3:
        continue
    ratio = on.mean() / max(off.mean(), 1e-9)
    print(f"  period {period:>3}: boundary mean {on.mean():.5f}  interior mean {off.mean():.5f}  ratio {ratio:.2f}")

print()
print("largest single-step jumps (step, mean|delta|):")
for i in np.argsort(diff)[-8:][::-1]:
    print(f"   step {int(i):>4}  {diff[i]:.5f}   (i mod 16 = {int(i) % 16}, mod 32 = {int(i) % 32})")

print()
print("GRIPPER channels over time (should open/close; training std ~0.9):")
for name, i in (("l_grip", 14), ("r_grip", 22)):
    v = A[:, i]
    print(f"  {name}: min {v.min():.4f}  max {v.max():.4f}  std {v.std():.5f}  unique(3dp) {len(np.unique(np.round(v,3)))}")
