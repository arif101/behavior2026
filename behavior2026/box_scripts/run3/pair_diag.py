"""Why did the ODART/factory pairing find no shared prefix? Print per-frame |dstate| for a few pairs and the dims that differ."""
import json, glob, numpy as np, pyarrow.parquet as pq
mix = "/root/b1k_radio_mix_readout"; tmap = json.load(open("/root/odart_episode_map.json"))
srcs = json.load(open(f"{mix}/meta/run3_sources.json"))["sources"]; off = {s["source"].rstrip("/").split("/")[-1]: s["episodes"][0] for s in srcs}
st = {}; act = {}
for f in sorted(glob.glob(f"{mix}/data/**/*.parquet", recursive=True)):
    t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state", "action"])
    e = t.column("episode_index").to_numpy(); fr = t.column("frame_index").to_numpy()
    s = np.stack(t.column("observation.state").to_pylist()).astype(np.float32); a = np.stack(t.column("action").to_pylist()).astype(np.float32)
    for eid in np.unique(e):
        m = e == eid; o = np.argsort(fr[m]); st[int(eid)] = s[m][o]; act[int(eid)] = a[m][o]
fac = {int(d): int(e) for e, d in tmap["factory"].items()}
print("factory eps:", len(fac), "state dim:", next(iter(st.values())).shape[1])
shown = 0
for root, eps in tmap["odart"].items():
    for r in eps:
        mo = off[root] + r["ep"]; mf = off["b1k_radio_factory"] + fac[r["demo"]]
        so, sf = st[mo], st[mf]; n = min(len(so), len(sf)); d = np.abs(so[:n] - sf[:n])
        dm = d.max(1); print(f"{root} ep{r['ep']} d{r['demo']} {r['tag']}: len odart={len(so)} fac={len(sf)}; max|ds| first 12 frames: {np.round(dm[:12], 4).tolist()}")
        print(f"   frame0 dims with |ds|>1e-3: {np.where(d[0] > 1e-3)[0].tolist()} vals {np.round(d[0][d[0] > 1e-3], 4).tolist()}")
        print(f"   min over frames of max|ds| = {dm.min():.4f} at frame {int(dm.argmin())}; frames with max|ds|<0.02: {int((dm < 0.02).sum())}, <0.05: {int((dm < 0.05).sum())}")
        # which dims drive the difference on the first 30 frames
        print(f"   per-dim mean |ds| over first 30 frames (dims > 1e-3): { {int(i): round(float(v), 4) for i, v in enumerate(d[:30].mean(0)) if v > 1e-3} }")
        shown += 1
        if shown >= 4: break
    if shown >= 4: break
