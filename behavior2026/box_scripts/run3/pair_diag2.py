"""Do ODART clips of the SAME demo share a proprio prefix (same restore + settle, then goals diverge)? k0 per opposite-tag pair."""
import json, glob, numpy as np, pyarrow.parquet as pq, collections
mix = "/root/b1k_radio_mix_readout"; tmap = json.load(open("/root/odart_episode_map.json"))
srcs = json.load(open(f"{mix}/meta/run3_sources.json"))["sources"]; off = {s["source"].rstrip("/").split("/")[-1]: s["episodes"][0] for s in srcs}
st = {}
for f in sorted(glob.glob(f"{mix}/data/**/*.parquet", recursive=True)):
    t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"])
    e = t.column("episode_index").to_numpy(); fr = t.column("frame_index").to_numpy(); s = np.stack(t.column("observation.state").to_pylist()).astype(np.float32)
    for eid in np.unique(e):
        m = e == eid; st[int(eid)] = s[m][np.argsort(fr[m])]
by_demo = collections.defaultdict(dict)
for root, eps in tmap["odart"].items():
    for r in eps: by_demo[r["demo"]][r["tag"]] = off[root] + r["ep"]
opp = [("ol5", "olm5"), ("od5", "odm5"), ("oy15", "oym15"), ("omix1", "omix2")]
print("demos:", sorted(by_demo), "clips:", sum(len(v) for v in by_demo.values()))
tot = collections.Counter()
for d in sorted(by_demo):
    tags = by_demo[d]
    for a, b in opp:
        if a in tags and b in tags:
            sa, sb = st[tags[a]], st[tags[b]]; n = min(len(sa), len(sb)); dm = np.abs(sa[:n] - sb[:n]).max(1)
            k2 = int(np.where(dm < 2e-3)[0].max()) if (dm < 2e-3).any() else -1; k1 = int(np.where(dm < 1e-2)[0].max()) if (dm < 1e-2).any() else -1
            print(f"d{d:>3} {a:>5}/{b:<5} len {len(sa):>3}/{len(sb):<3} first10 max|ds| {np.round(dm[:10], 4).tolist()} k0(2e-3)={k2} k0(1e-2)={k1} |ds|@k0+1..+32 mean={dm[k2+1:k2+33].mean() if k2 >= 0 else float('nan'):.3f}")
            tot["pairs"] += 1; tot["k0>=5"] += (k2 >= 5)
    # also same-sign pairs, e.g. ol5 vs od5, as a sanity check on the shared prefix
    if "ol5" in tags and "od5" in tags:
        sa, sb = st[tags["ol5"]], st[tags["od5"]]; n = min(len(sa), len(sb)); dm = np.abs(sa[:n] - sb[:n]).max(1)
        print(f"     ol5/od5 k0(2e-3)={int(np.where(dm < 2e-3)[0].max()) if (dm < 2e-3).any() else -1}")
print("SUMMARY", dict(tot))
