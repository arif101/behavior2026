"""Point pi05_radio_gate at the converted dataset, add a val split, fix the step accounting.

Three corrections in one patch:

1. dataset_root -> /root/b1k_radio2 (the one written by the OFFICIAL add_target_points.py, whose
   meta is self-consistent with its data; the hand-grafted /root/b1k_radio inherited meta claiming
   20,000 episodes and died in LeRobot's index build).

2. episodes -> 0..179 for training, holding out 180..199 for validation. Without a held-out split
   there is NO way to tell "learned the task" from "memorised 200 trajectories" — 3B params over
   200 demos for ~11 epochs is a lot of capacity. Each episode is a DIFFERENT radio placement
   (x[3.18,3.87], y[4.60,6.04]), so held-out episodes test instance generalization, which is
   exactly what the point-conditioning hypothesis claims to buy.

3. save_interval 3360 -> 4480. 429,928 frames / (batch 32 x 3 GPUs = 96) = 4,478 steps per epoch.
   3360 assumed 4 GPUs; this box has 3 usable.
"""

import ast
import pathlib

P = pathlib.Path("/root/openpi/src/openpi/training/config.py")
s = P.read_text()

SUBS = [
    ('dataset_root="/root/b1k_radio",', 'dataset_root="/root/b1k_radio2",'),
    ('"episodes": list(range(200))', '"episodes": list(range(180))'),
    ('save_interval=3_360,', 'save_interval=4_480,'),
    ('save_interval=3360,', 'save_interval=4_480,'),
]

n = 0
for old, new in SUBS:
    if old in s:
        s = s.replace(old, new, 1)
        n += 1
        print(f"  {old!r} -> {new!r}")

ast.parse(s)
P.write_text(s)
print(f"applied {n} substitutions")

# Echo the resulting block so the change is visible, not assumed.
i = s.index('name="pi05_radio_gate"')
print("\n--- resulting config ---")
print("\n".join(s[i - 200:i + 1500].splitlines()[:34]))
