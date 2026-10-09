"""Replay a LeRobot root through the serve-side causal StageV2Tracker (button = target_points_R + EE_R, i.e. an ORACLE
prediction) and score agreement with the offline stage_v2 labels: per-class confusion, lead/lag of each transition."""
import glob, sys, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/root/behavior2026/behavior2026/eval")
from stage_v2_rule import StageV2Tracker
root = sys.argv[1]; max_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
files = sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))
conf = np.zeros((4, 4), np.int64); leads = {1: [], 2: [], 3: []}; n_eps = 0; tgt_err = []
for f in files:
    t = pq.read_table(f, columns=["episode_index", "target_points", "stage_v2", "progress", "target_points_v2", "observation.state"])
    ep = np.asarray(t.column("episode_index").to_pylist()); tp = np.asarray(t.column("target_points").to_pylist(), np.float32).reshape(len(t), 6)
    st = np.asarray(t.column("stage_v2").to_pylist()).reshape(-1); tp2 = np.asarray(t.column("target_points_v2").to_pylist(), np.float32).reshape(len(t), 6)
    state = np.asarray(t.column("observation.state").to_pylist(), np.float32)
    for e in np.unique(ep):
        if n_eps >= max_eps: break
        idx = np.flatnonzero(ep == e); tr = StageV2Tracker(); pred = np.zeros(len(idx), np.int32)
        for k, i in enumerate(idx):
            eL, eR = state[i, 17:20], state[i, 42:45]; p = tp[i, 3:6] + eR
            s_, pr_, pts = tr.step(p, eL, eR); pred[k] = s_
            tgt_err.append(float(np.abs(pts.reshape(-1) - tp2[i]).max()))
        lab = st[idx]
        for a, b in zip(lab, pred): conf[a, b] += 1
        for c in (1, 2, 3):
            fl = int(np.argmax(lab >= c)) if (lab >= c).any() else None; fp = int(np.argmax(pred >= c)) if (pred >= c).any() else None
            if fl is not None and fp is not None: leads[c].append(fp - fl)
        n_eps += 1
tot = conf.sum(); acc = np.trace(conf) / tot
print(f"MIRROR {root.split('/')[-1]}: {n_eps} eps, {tot} frames | frame agreement {100*acc:.1f}% | per-class recall {[round(conf[c,c]/max(conf[c].sum(),1),3) for c in range(4)]}")
print("   confusion rows=label cols=mirror:", conf.tolist())
q = lambda x: np.percentile(x, [5, 50, 95]).round(0).tolist() if x else None
print(f"   transition lead (mirror - label frames; negative = mirror early): grasp {q(leads[1])} lift {q(leads[2])} press {q(leads[3])}")
print(f"   target_points_v2 max abs error vs label: {max(tgt_err):.4f} m (oracle button -> should be ~0 except during stage disagreements)")
