"""Decompose the policy's emitted base actions: is it commanding travel, or only yaw?

The first closed-loop rollout showed the robot rotate to bring the radio into view and then never
approach it (base path 0.76 m vs the human's 5.70 m, arms 3x the human's). The base is 3 DoF,
action[0:3], body-frame [vx, vy, wz]. If translation is near zero while yaw is live, the failure
is base action generation and no amount of grounding/memory/resolution addresses it.
"""

import json

import numpy as np

rows = [json.loads(line) for line in open("/root/action_log.jsonl")]
print("inference calls logged:", len(rows))
if not rows:
    raise SystemExit("no actions logged")

print("action_dim:", rows[0]["action_dim"])

tr = np.array([r["base_trans_mag"] for r in rows])
yw = np.array([r["base_yaw_mag"] for r in rows])
ar = np.array([r["arm_mag"] for r in rows])
vx = np.array([r["base_vx"] for r in rows])
vy = np.array([r["base_vy"] for r in rows])

print()
print(f"{'channel':<28}{'mean':>10}{'median':>10}{'p90':>10}{'max':>10}")
for name, v in (("base TRANSLATION |vx,vy|", tr), ("base YAW |wz|", yw), ("arms |a[3:]|", ar)):
    print(f"{name:<28}{v.mean():>10.4f}{np.median(v):>10.4f}{np.percentile(v, 90):>10.4f}{v.max():>10.4f}")

print()
print(f"  vx mean {vx.mean():+.4f}   vy mean {vy.mean():+.4f}")
print(f"  fraction of steps with |translation| < 0.01 : {(tr < 0.01).mean():.3f}")
print(f"  fraction of steps with |yaw|         < 0.01 : {(yw < 0.01).mean():.3f}")
print(f"  yaw-to-translation ratio (means)            : {yw.mean() / max(tr.mean(), 1e-9):.2f}")
print()
if tr.mean() < 0.02 and yw.mean() > 2 * tr.mean():
    print("READ: ROTATE-IN-PLACE CONFIRMED -- translation is being suppressed while yaw is live.")
    print("      The bottleneck is base action generation, not perception/memory/resolution.")
elif tr.mean() < 0.02 and yw.mean() < 0.02:
    print("READ: the base is barely commanded AT ALL (neither travel nor yaw).")
    print("      The policy is effectively ignoring the base DoFs.")
else:
    print("READ: translation IS being commanded -- rotate-in-place is REFUTED.")
    print("      The base commands exist but do not produce displacement: look at the")
    print("      controller//action scaling or at command-vs-achieved delivery instead.")
