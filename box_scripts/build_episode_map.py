"""Build the VERIFIED episode_index -> raw demo_id map + per-episode video segment offsets.

The naive assumption (episode_index i -> demo 10*(i+1)) held for eps 0-7 in the spot check but
BREAKS at ep_idx 98 (cam T=2035 vs label T=1911). So build the map properly:

  1. If episodes meta carries a raw/demo id column, use it.
  2. Else match by length fingerprint (T_parquet + 1 == T_poseJSON), then VERIFY every pairing
     physically: left-EE late-episode min distance to the composed metalink label must be < 8 cm
     (correct pairings measured 2-4 cm; wrong pairings ~0.3+ m).

Also extracts each episode's video location (chunk/file + timestamps) from the v3 episodes meta.
Writes /root/episode_map.json.
"""

import glob
import json

import numpy as np
import pyarrow.parquet as pq

ROOT = "/root/b1k_radio2"
EE_L = slice(17, 20)

meta_files = sorted(glob.glob(f"{ROOT}/meta/episodes/**/*.parquet", recursive=True))
print(f"episodes meta files: {meta_files}")
em = pq.read_table(meta_files[0]).to_pydict() if meta_files else None
cols = list(em.keys())
print("episode-meta columns:", cols)

n_ep = len(em["episode_index"])
lengths = em.get("length") or em.get("episode_length")
raw_col = "raw_episode_id" if "raw_episode_id" in cols else next(
    (c for c in cols if "raw" in c.lower() or "demo" in c.lower()), None)
print(f"n_ep={n_ep} raw_col={raw_col}")

# label lengths per demo id
lbl_len = {}
for fp in glob.glob("/root/metalink_labels/ep*.npz"):
    did = int(fp.split("ep")[-1].split(".")[0])
    lbl_len[did] = len(np.load(fp)["meta_base"])

mapping = {}
if raw_col:
    for i, r in zip(em["episode_index"], em[raw_col]):
        mapping[int(i)] = int(r)
    print(f"mapping from column {raw_col}; sample: {list(mapping.items())[:5]}")
    # physically verify 5 spread-out pairings even on the trusted column
    data_files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    checks = {i: mapping[i] for i in [0, 49, 98, 149, 199] if i in mapping}
    props = {}
    for df in data_files:
        t = pq.read_table(df, columns=["episode_index", "observation.state"])
        eps = t["episode_index"].to_numpy()
        st = np.stack(t["observation.state"].to_numpy())[:, EE_L]
        for e in np.unique(eps):
            if int(e) in checks:
                props[int(e)] = np.concatenate([props.get(int(e), np.zeros((0, 3))), st[eps == e]])
    for i, did in checks.items():
        p, l = props[i], np.load(f"/root/metalink_labels/ep{did}.npz")["meta_base"]
        T = min(len(p), len(l))
        s = int(T * 0.6)
        d = float(np.linalg.norm(p[s:T] - l[s:T], axis=1).min())
        print(f"  verify ep_idx {i} -> demo {did}: T {len(p)}/{len(l)} press-dist {d:.3f} m"
              f"{'  OK' if d < 0.08 and abs(len(p) - len(l)) <= 1 else '  <-- FAIL'}")
else:
    # length fingerprint
    from collections import defaultdict
    by_len = defaultdict(list)
    for did, L in lbl_len.items():
        by_len[L - 1].append(did)
    cand = {int(i): by_len.get(int(L), []) for i, L in zip(em["episode_index"], lengths)}
    n_uni = sum(1 for v in cand.values() if len(v) == 1)
    print(f"unique-length matches: {n_uni}/{n_ep}; ambiguous: "
          f"{sum(1 for v in cand.values() if len(v) > 1)}; "
          f"unmatched: {sum(1 for v in cand.values() if not v)}")

    # load left-EE proprio for verification / disambiguation
    data_files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
    props = {}
    for df in data_files:
        t = pq.read_table(df, columns=["episode_index", "observation.state"])
        eps = t["episode_index"].to_numpy()
        st = np.stack(t["observation.state"].to_numpy())[:, EE_L]
        for e in np.unique(eps):
            props[int(e)] = np.concatenate([props.get(int(e), np.zeros((0, 3))), st[eps == e]])

    def press_dist(ep_i, did):
        p = props[ep_i]
        l = np.load(f"/root/metalink_labels/ep{did}.npz")["meta_base"]
        T = min(len(p), len(l))
        s = int(T * 0.6)
        return float(np.linalg.norm(p[s:T] - l[s:T], axis=1).min())

    used = set()
    n_verified = 0
    report = []
    for i in sorted(cand):
        best, bd = None, 1e9
        for did in cand[i]:
            if did in used:
                continue
            d = press_dist(i, did)
            if d < bd:
                best, bd = did, d
        if best is not None and bd < 0.08:
            mapping[i] = best
            used.add(best)
            n_verified += 1
        else:
            report.append((i, cand[i], round(bd, 3) if best else None))
    print(f"verified pairings: {n_verified}/{n_ep}")
    if report:
        print("UNRESOLVED:", report[:10])

# per-episode video segments (v3: per-video-key chunk/file index + timestamps)
vid_cols = [c for c in cols if "video" in c.lower() or "chunk" in c.lower() or
            "file" in c.lower() or "from" in c.lower() or "to" in c.lower()]
print("video/segment columns:", vid_cols)
segments = {}
for k, i in enumerate(em["episode_index"]):
    segments[int(i)] = {c: (em[c][k] if not hasattr(em[c][k], "tolist") else em[c][k].tolist())
                        for c in vid_cols}

out = {"mapping": mapping, "segments": segments,
       "lengths": {int(i): int(L) for i, L in zip(em["episode_index"], lengths)}}
with open("/root/episode_map.json", "w") as f:
    json.dump(out, f)
print(f"WROTE /root/episode_map.json  ({len(mapping)} verified pairs)")
# show the naive-assumption breakage
bad = [i for i, d in mapping.items() if d != 10 * (i + 1)]
print(f"episodes where naive i->10(i+1) was WRONG: {len(bad)}  e.g. {sorted(bad)[:10]}")
