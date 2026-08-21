"""Z-score every emitted action dimension against the TRAINING action distribution.

A policy reproducing its training distribution emits actions whose per-dimension mean/std match
norm_stats. Systematic deviation means the SERVING path is transforming actions differently from
how they were learned -- a train/serve mismatch, not a learning failure.

Ruled out already:
  * extra_delta_transform pushes inputs (absolute->delta) and outputs (delta->absolute) TOGETHER,
    so False removes both sides symmetrically. Self-consistent.
  * use_delta_joint_actions belongs to LeRobotAlohaDataConfig; LeRobotB1KDataConfig has none.

What we already know is off:
  arm/torso action norm  training 2.007  vs  served 2.790 mean / 3.043 median
  base action std        training [0.092, 0.048, 0.104]  vs served translation max 0.342
"""

import json

import numpy as np

rows = [json.loads(line) for line in open("/root/action_log.jsonl")]
rows = [r for r in rows if "full" in r]
if not rows:
    raise SystemExit("no full-vector records — restart the server so the upgraded logger loads")

A = np.array([r["full"] for r in rows])           # (T, 23)
d = json.load(open("/root/ckpt/assets/b1k_radio/norm_stats.json"))
n = d.get("norm_stats", d)["actions"]
mu = np.array(n["mean"])
sd = np.array(n["std"])

print(f"samples {A.shape[0]}   dims {A.shape[1]}")
print()
GROUPS = [("base", 0, 3), ("torso", 3, 7), ("left_arm", 7, 14), ("l_grip", 14, 15),
          ("right_arm", 15, 22), ("r_grip", 22, 23)]
print(f"{'group':<11}{'train mean':>12}{'served mean':>13}{'train std':>11}{'served std':>12}{'|z| mean':>10}")
for name, i, j in GROUPS:
    tm, ts = mu[i:j].mean(), sd[i:j].mean()
    am, asd = A[:, i:j].mean(), A[:, i:j].std()
    z = np.abs((A[:, i:j] - mu[i:j]) / np.maximum(sd[i:j], 1e-6)).mean()
    print(f"{name:<11}{tm:>12.4f}{am:>13.4f}{ts:>11.4f}{asd:>12.4f}{z:>10.2f}")

z_all = np.abs((A - mu) / np.maximum(sd, 1e-6))
print()
print(f"overall mean |z| across all dims/steps : {z_all.mean():.2f}")
print(f"fraction of (step,dim) with |z| > 3    : {(z_all > 3).mean():.3f}")
print(f"fraction of (step,dim) with |z| > 5    : {(z_all > 5).mean():.3f}")

print()
print("JITTER — step-to-step change, served vs the training action std:")
diff = np.abs(np.diff(A, axis=0))
for name, i, j in GROUPS:
    print(f"  {name:<11} mean |Δ per step| {diff[:, i:j].mean():.4f}   "
          f"(training std for these dims {sd[i:j].mean():.4f})   "
          f"ratio {diff[:, i:j].mean() / max(sd[i:j].mean(), 1e-6):.2f}")

print()
if z_all.mean() > 2.0:
    print("READ: emitted actions are FAR outside the training distribution -> SERVING PATH BUG.")
elif diff.mean() / max(sd.mean(), 1e-6) > 1.0:
    print("READ: distribution matches but step-to-step CHANGE exceeds the whole training spread")
    print("      -> the chunk/receding-horizon stitching is injecting the jitter.")
else:
    print("READ: emitted actions are in-distribution and smooth -> serving is faithful;")
    print("      the policy genuinely learned this behavior.")
