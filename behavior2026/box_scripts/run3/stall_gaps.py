import numpy as np, json, glob
from scipy.spatial.transform import Rotation as R
for f in sorted(glob.glob("/root/stall_states/h4d_tr*.npz")):
    z = np.load(f); pro = z["proprio"]; bp = z["base_pos"].astype(float); bq = z["base_quat"].astype(float); rp = z["radio_pos"].astype(float)
    Rb = R.from_quat(bq); eeR = bp + Rb.apply(pro[42:45]); eeL = bp + Rb.apply(pro[17:20]); tgt = rp + np.array([0, 0, 0.108])
    n = f.split("/")[-1]
    print(f"{n}: step {int(z['step'])} base->radio xy {np.linalg.norm((rp - bp)[:2]):.2f} | handR->radio {np.linalg.norm(eeR - rp):.3f} handR->grasp~ {np.linalg.norm(eeR - tgt):.3f} planar {np.linalg.norm((eeR - tgt)[:2]):.3f} | handL->radio {np.linalg.norm(eeL - rp):.3f} | handR z {eeR[2]:.2f} radio z {rp[2]:.2f}")
