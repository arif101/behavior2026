"""Task-62 demo census from raw hdf5 only (no sim). Writes /root/t62_census.json.
Per demo: length, q_final, reward-event frames, rollback count, state_size profile,
gripper closure events per arm (23-d layout: grip_L=14, grip_R=22), and which arm closed
last before each reward event (arm audit, Law 5)."""
import h5py, numpy as np, json, glob, os, re
GL, GR = 14, 22
rows = []
cfg_seen = None
for p in sorted(glob.glob("/root/rawdemos/task-0062/episode_*.hdf5")):
    ep = int(re.search(r"episode_(\d+)", p).group(1))
    with h5py.File(p, "r") as f:
        keys = [k for k in f["data"] if k.startswith("demo_")]
        # pick the longest demo_* group as the real episode
        k = max(keys, key=lambda k: f["data"][k]["action"].shape[0])
        g = f["data"][k]
        a = g["action"][:]; r = g["reward"][:]; ss = g["state_size"][:]
        T = a.shape[0]
        ev = np.nonzero(np.abs(np.diff(r, prepend=0.0)) > 1e-6)[0]
        deltas = np.round(np.diff(r, prepend=0.0)[ev], 3)
        nrb = len(g["rollbacks"]) if "rollbacks" in g else 0
        # closures: sign crossing of gripper channel from open(+) to closed(-)
        def closures(ch):
            s = np.sign(a[:, ch]); c = np.nonzero((s[1:] < 0) & (s[:-1] >= 0))[0] + 1
            return c.tolist()
        cL, cR = closures(GL), closures(GR)
        def last_closure_before(fr):
            l = max([c for c in cL if c <= fr], default=-1); rr = max([c for c in cR if c <= fr], default=-1)
            if l < 0 and rr < 0: return "none"
            return "L" if l > rr else "R"
        # full-state frames: state_size equals max (first frames), then drops
        full = int(ss.max()); nfull = int((ss == full).sum()); first_partial = int(np.argmax(ss < full)) if (ss < full).any() else T
        base_moving = float((np.abs(a[:, :3]).max(1) > 1e-3).mean())
        if cfg_seen is None:
            c = json.loads(g.attrs["config"]) if "config" in g.attrs else json.loads(f["data"].attrs["config"])
            rb = c["robots"][0]; cfg_seen = dict(robot_name=rb.get("name"), grasping_mode=rb.get("grasping_mode"),
                action_normalize=rb.get("action_normalize"), freqs=(c["env"]["action_frequency"], c["env"]["physics_frequency"]),
                controllers={k: v.get("name") for k, v in rb.get("controller_config", {}).items()})
        rows.append(dict(ep=ep, group=k, T=T, q_final=float(r[-1]), n_events=len(ev), events=ev.tolist(),
                         deltas=deltas.tolist(), arm_at_event=[last_closure_before(e) for e in ev],
                         n_rollbacks=nrb, state_full=full, n_full_frames=nfull, first_partial=first_partial,
                         n_closures_L=len(cL), n_closures_R=len(cR), base_moving_frac=round(base_moving, 3),
                         grip_range=[float(a[:, GL].min()), float(a[:, GL].max()), float(a[:, GR].min()), float(a[:, GR].max())]))
json.dump(dict(config=cfg_seen, demos=rows), open("/root/t62_census.json", "w"))
T = np.array([x["T"] for x in rows]); q = np.array([x["q_final"] for x in rows])
print("config:", json.dumps(cfg_seen))
print(f"demos={len(rows)}  T mean={T.mean():.0f} med={np.median(T):.0f} min={T.min()} max={T.max()}  eval budget(1.5x mean)={1.5*T.mean():.0f}")
print(f"q_final: ==1.0: {(q>=0.999).sum()}  hist: {np.unique(np.round(q,2), return_counts=True)}")
print(f"n_events hist: {np.unique([x['n_events'] for x in rows], return_counts=True)}")
print(f"rollbacks: mean={np.mean([x['n_rollbacks'] for x in rows]):.1f} max={max(x['n_rollbacks'] for x in rows)}  demos with >0: {sum(x['n_rollbacks']>0 for x in rows)}")
print(f"full-state frames: n_full mean={np.mean([x['n_full_frames'] for x in rows]):.1f}  first_partial mean={np.mean([x['first_partial'] for x in rows]):.1f}  state_full dims={sorted(set(x['state_full'] for x in rows))[:5]}")
print(f"closures/demo: L mean={np.mean([x['n_closures_L'] for x in rows]):.1f}  R mean={np.mean([x['n_closures_R'] for x in rows]):.1f}")
print(f"base moving frac mean={np.mean([x['base_moving_frac'] for x in rows]):.2f}")
from collections import Counter
for i in range(5):
    arms = Counter(x["arm_at_event"][i] for x in rows if len(x["arm_at_event"]) > i)
    d = Counter(x["deltas"][i] for x in rows if len(x["deltas"]) > i)
    print(f"event {i}: arm={dict(arms)}  delta={dict(d)}")
neg = [x["ep"] for x in rows if any(d < 0 for d in x["deltas"])]
print("demos with negative reward deltas (regressions):", len(neg), neg[:10])
