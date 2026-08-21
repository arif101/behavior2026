"""Did the torso/arm commands destabilise the robot as it drove toward the target?

USER OBSERVATION (from the oracle-point video): the robot started moving toward the radio and then
FELL OVER. That reframes an earlier measurement -- torso variance 4.5x training (0.884 vs 0.194)
with arms sweeping 2-3x the human's path -- from a symptom into a candidate CAUSE. On a wheeled
mobile manipulator, swinging that much mass while driving is how you tip.

Candidate chain, every link already measured:
    points -> steers toward the radio (base travel 0.76 -> 2.53 m)
           -> 6.88x action discontinuity at each 16-step replan seam
           -> torso/arm jerk while the base is moving
           -> centre of mass shifts -> fall

This checks the profile over the rollout: if torso/arm command magnitude RISES as the base starts
travelling, the destabilisation hypothesis is supported. If torso is flat throughout, the fall has
another cause (e.g. base velocity alone, or terrain/collision) and temporal ensembling would not
help.

The action log accumulates across runs, so only the tail (this rollout) is used.
"""

import json

import numpy as np

rows = []
with open("/root/action_log.jsonl") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "full" in r:
            rows.append(r)

if not rows:
    raise SystemExit("no full-vector rows in the log")

# ~3225 steps at --action-horizon 16 -> ~202 policy calls for this rollout
TAIL = 202
A = np.array([r["full"] for r in rows[-TAIL:]])
print(f"total rows {len(rows)}   analysing last {len(A)} (this rollout)")

base = np.linalg.norm(A[:, 0:3], axis=1)
torso = np.linalg.norm(A[:, 3:7], axis=1)
arms = np.linalg.norm(A[:, 7:], axis=1)

n = len(A)
seg = max(n // 8, 1)
print()
print(f"  {'window':<18}{'base':>10}{'torso':>10}{'arms':>10}")
for i in range(0, n, seg):
    s = slice(i, min(i + seg, n))
    print(f"  calls {i:>3}-{min(i+seg,n):<10}{base[s].mean():>10.4f}{torso[s].mean():>10.4f}{arms[s].mean():>10.4f}")

print()
print(f"  torso  mean {torso.mean():.4f}  max {torso.max():.4f}   (training std for these dims 0.194)")
print(f"  base   mean {base.mean():.4f}  max {base.max():.4f}")

# does torso activity track base motion?
if n > 8:
    c = float(np.corrcoef(base, torso)[0, 1])
    print(f"  corr(base motion, torso magnitude) = {c:+.3f}")
    print()
    if c > 0.3:
        print("  READ: torso swings RISE with base motion -> destabilisation while driving is")
        print("        plausible, and smoothing the replan seam attacks it directly.")
    elif c < -0.3:
        print("  READ: torso activity FALLS as the base moves -> the two are anti-correlated;")
        print("        the fall is unlikely to be torso-while-driving.")
    else:
        print("  READ: no clear coupling between base motion and torso swing. The fall needs a")
        print("        different explanation -- check base velocity magnitude and contacts.")
