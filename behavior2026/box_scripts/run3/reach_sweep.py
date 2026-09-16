import glob, json, sys, numpy as np, pyarrow.parquet as pq, h5py
sys.path.insert(0, "/root/run3"); sys.path.insert(0, "/root/behavior2026/behavior2026/eval")
import relabel_v2 as RL, stage_v2_rule as SR
emap = json.load(open("/root/backup/keys/episode_map.json"))["mapping"]
def load(root, max_eps, with_toggle):
    t = pq.read_table(sorted(glob.glob(root + "/data/**/*.parquet", recursive=True))[0], columns=["episode_index", "target_points", "stage", "observation.state"])
    ep = np.asarray(t.column("episode_index").to_pylist()); tp = np.asarray(t.column("target_points").to_pylist(), np.float32).reshape(len(t), 6)
    st1 = np.asarray(t.column("stage").to_pylist()).reshape(len(t), -1)[:, 0].astype(int); state = np.asarray(t.column("observation.state").to_pylist(), np.float32)
    eps = []
    for e in np.unique(ep)[:max_eps]:
        idx = np.flatnonzero(ep == e); tog = None; lifted = st1[idx] >= 2
        if with_toggle:
            d = int(emap[str(int(e))]); lab = np.load(f"/root/backup/metalink_labels/ep{d}.npz")["meta_world"][:len(idx)]
            lifted = np.maximum.accumulate((lab[:, 2] - lab[0, 2]) > 0.03)
            with h5py.File(f"/root/rawdemos/task-0000/episode_{d:08d}.hdf5", "r") as hf: rw = hf["data/demo_0/reward"][:]
            hit = np.flatnonzero(rw >= 0.5); tog = int(hit[0]) if len(hit) else None
        eps.append((tp[idx], st1[idx], lifted, tog, state[idx]))
    return eps
for name, root, max_eps, wt, anchor in (("map", "/root/b1k_radio_map", 80, True, "closest"), ("approach_v2", "/root/manufactured/b1k_radio_approach_v2", 99, False, "none"), ("episodes", "/root/manufactured/b1k_radio_episodes", 99, False, "closest")):
    eps = load(root, max_eps, wt)
    for rr in (0.25, 0.20, 0.15, 0.10):
        SR.REACH_R = rr; agree = 0; tot = 0; leads = []; tr_rec = [0, 0]; pr_rec = [0, 0]
        for tp, st1, lifted, tog, state in eps:
            lab = RL.relabel_episode(tp, st1, lifted, 0.12, 0.12, rr, toggle_idx=tog, press_anchor=anchor)[0]
            tr = SR.StageV2Tracker(); pred = np.zeros(len(tp), np.int32)
            for k in range(len(tp)): pred[k] = tr.step(tp[k, 3:6] + state[k, 42:45], state[k, 17:20], state[k, 42:45])[0]
            agree += int((lab == pred).sum()); tot += len(lab)
            tr_rec[0] += int(((lab == 2) & (pred == 2)).sum()); tr_rec[1] += int((lab == 2).sum()); pr_rec[0] += int(((lab == 3) & (pred == 3)).sum()); pr_rec[1] += int((lab == 3).sum())
            if (lab >= 3).any() and (pred >= 3).any(): leads.append(int(np.argmax(pred >= 3)) - int(np.argmax(lab >= 3)))
        q = np.percentile(leads, [5, 50, 95]).round(0).tolist() if leads else None
        print(f"{name:12s} reach_r={rr:.2f}: agreement {100*agree/tot:.1f}% | transport recall {tr_rec[0]/max(tr_rec[1],1):.3f} | press recall {pr_rec[0]/max(pr_rec[1],1):.3f} | press lead p5/50/95 {q} | label press frames {pr_rec[1]}")
