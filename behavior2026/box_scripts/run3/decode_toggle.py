"""Decode the radio's per-frame state from the raw demo hdf5: the recorder stores awake objects as
[pos3, quat4, linvel3, angvel3, nonkin...]; frame 0 holds the full scene (radio at 484). Track the radio block through
awake frames by pose continuity and print the two non-kinematic floats (one should be ToggledOn)."""
import h5py, numpy as np, json, sys
bank = json.load(open("/root/bank_press_frames.json"))
def radio_track(d):
    f = h5py.File(f"/root/rawdemos/task-0000/episode_{d:08d}.hdf5", "r"); S = f["data/demo_0/state"]; ss = f["data/demo_0/state_size"][:]
    s0 = S[0]; pos = s0[484:487].astype(float); blk0 = s0[484:499]
    base = int(ss[1:].min()); track = []; prev = pos
    for t in range(len(ss)):
        n = int(ss[t]); s = S[t][:n]
        if t == 0: track.append((0, float(blk0[13]), float(blk0[14]), pos.copy())); continue
        if n <= base: track.append((t, None, None, None)); continue
        best = None
        for i in range(base, n - 15):
            q = s[i+3:i+7]
            if abs(np.linalg.norm(q) - 1) < 1e-3 and np.linalg.norm(s[i:i+3] - prev) < 0.15:
                best = i; break
        if best is None: track.append((t, None, None, None)); continue
        prev = s[best:best+3].astype(float); track.append((t, float(s[best+13]), float(s[best+14]), prev.copy()))
    return track, len(ss)
for d in [int(x) for x in sys.argv[1:]]:
    tr, n = radio_track(d)
    nk1 = [(t, a) for t, a, b, p in tr if a is not None]; nk2 = [(t, b) for t, a, b, p in tr if b is not None]
    ch1 = [nk1[k][0] for k in range(1, len(nk1)) if nk1[k][1] != nk1[k-1][1]]; ch2 = [nk2[k][0] for k in range(1, len(nk2)) if nk2[k][1] != nk2[k-1][1]]
    zs = [p[2] for t, a, b, p in tr if p is not None]
    bp = bank.get(str(d), {}).get("press_frames"); cl = bank.get(str(d), {}).get("closure")
    print(f"d{d}: n={n} awake-tracked={len(nk1)} | nk1 vals={sorted(set(v for _, v in nk1))[:4]} changes@{ch1[:5]} | nk2 vals={sorted(set(v for _, v in nk2))[:4]} changes@{ch2[:5]} | radio z range {min(zs):.2f}-{max(zs):.2f} | bank press {bp} closure {cl}")
