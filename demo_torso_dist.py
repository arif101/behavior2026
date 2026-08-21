"""Demo torso-command distribution BY STAGE — validates the torso-dive hypothesis.

The converting eval run's torso commands sat at [1.21, -1.80, -0.77] (dims 3:6 of the action);
every failing run sat shallow (dim-2 ~ -1.0). If the demos' MANIPULATE-stage torso matches the
converting run's depth, the failures are under-sampling a KNOWN demo mode (data/commitment
problem), not exceeding the robot's capability.

Uses the stage column we baked into /root/b1k_radio_map (0 APPROACH 1 ACQUIRE 2 MANIP 3 END).
"""

import glob

import numpy as np
import pyarrow.parquet as pq

A = []
S = []
for fp in sorted(glob.glob("/root/b1k_radio_map/data/**/*.parquet", recursive=True)):
    t = pq.read_table(fp, columns=["action", "stage"])
    A.append(np.stack([np.asarray(r, np.float32) for r in t["action"].to_pylist()]))
    S.append(t["stage"].to_numpy())
A = np.concatenate(A)
S = np.concatenate(S)
print(f"{len(A)} frames, action dim {A.shape[1]}")

names = {0: "APPROACH", 1: "ACQUIRE", 2: "MANIPULATE", 3: "END"}
print(f"{'stage':>10} {'n':>7}   torso dims 3:6 mean            p10(dim4)  p90(dim4)")
for st in (0, 1, 2, 3):
    m = S == st
    T = A[m][:, 3:6]
    print(f"{names[st]:>10} {m.sum():>7}   {np.round(T.mean(0), 3)}   "
          f"{np.percentile(T[:, 1], 10):.3f}   {np.percentile(T[:, 1], 90):.3f}")

deep = A[S == 2][:, 4]
print(f"\nMANIPULATE dim-4 (the 'dive' axis): median {np.median(deep):.3f}, "
      f"p25 {np.percentile(deep, 25):.3f}, frac deeper than -1.5: {np.mean(deep < -1.5):.3f}")
print("converting eval run sat at -1.80; failing runs at -0.95..-1.03")
