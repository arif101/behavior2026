"""Canonical PRE-GRASP arm posture from the ODART clips in the readout mix: frames where the right hand is 9-13 cm from the
(rail) target (target_points_v2[right] in the base frame), gripper open -> median right-arm qpos (7) + trunk qpos (4),
plus the median hand->target offset vector in the base frame. Writes /root/canonical_pregrasp.json."""
import json, glob, numpy as np, pyarrow.parquet as pq
mix = "/root/b1k_radio_mix_readout"
src = json.load(open(f"{mix}/meta/run3_sources.json"))["sources"]
fac = [s for s in src if s["source"].endswith("b1k_radio_factory")][0]["episodes"]
cols = ["episode_index", "frame_index", "observation.state", "target_points_v2", "stage_v2"]
qa, qt, off, n_ep = [], [], [], set()
for f in sorted(glob.glob(f"{mix}/data/**/*.parquet", recursive=True)):
    t = pq.read_table(f, columns=cols)
    ep = t.column("episode_index").to_numpy()
    st = np.stack(t.column("observation.state").to_pylist()).astype(np.float64)
    tp = np.stack(t.column("target_points_v2").to_pylist()).astype(np.float64).reshape(len(ep), -1, 3)
    sv = t.column("stage_v2").to_numpy()
    odart = ~((ep >= fac[0]) & (ep <= fac[1]))
    dR = np.linalg.norm(tp[:, 1], axis=1)
    grip_open = st[:, 45] > 0.03 if st.shape[1] > 46 else np.ones(len(ep), bool)   # gripper qpos column (approx; open > 0.03)
    m = odart & (dR > 0.09) & (dR < 0.13) & (sv <= 1)
    qa.append(st[m, 28:35]); qt.append(st[m, 53:57]); off.append(tp[m, 1]); n_ep |= set(ep[m].tolist())
qa = np.concatenate(qa); qt = np.concatenate(qt); off = np.concatenate(off)
out = dict(n_frames=int(len(qa)), n_episodes=len(n_ep), arm_right_qpos=np.median(qa, 0).round(4).tolist(), arm_right_qpos_std=qa.std(0).round(3).tolist(),
           trunk_qpos=np.median(qt, 0).round(4).tolist(), hand_to_target_base=np.median(off, 0).round(4).tolist(), source="ODART clips, |tp_v2 right| in [0.09,0.13], stage<=1")
json.dump(out, open("/root/canonical_pregrasp.json", "w"), indent=1)
print("CANON_POSTURE", json.dumps(out))
