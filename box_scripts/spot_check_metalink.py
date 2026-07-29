"""Physical validation of the composed metalink labels against the training parquet.

In every demo the task SUCCEEDS, and ToggledOn fires on finger overlap with a ~2.2 cm sphere at
the metalink held 5 steps. So near the end of each episode the EE (proprio, base frame) must
come close to our composed metalink_base label. Gross errors (axis flip, frame misalignment,
wrong offset) would show up as min-distances of 0.3 m+.

Checks per sampled episode:
  - frame-count parity: parquet episode length == pose-JSON/label length
  - min over the last 40% of frames of |ee - meta_base| for EACH arm (closest arm reported)
Expected: closest-arm min ~0.05-0.15 m (EE point vs fingertip geometry), NOT 0.3+.
"""

import glob
import json

import numpy as np
import pyarrow.parquet as pq

ROOT = "/root/b1k_radio2"
EE_L, EE_R = slice(17, 20), slice(42, 45)

info = json.load(open(f"{ROOT}/meta/info.json"))
print("dataset:", info.get("total_episodes"), "episodes,", info.get("total_frames"), "frames")

# ---- map episode_index -> raw demo id ------------------------------------------------------
raw_key = None
ep_meta_files = sorted(glob.glob(f"{ROOT}/meta/episodes/*.parquet")) or sorted(
    glob.glob(f"{ROOT}/meta/episodes*.parquet"))
ep_map = {}
if ep_meta_files:
    emt = pq.read_table(ep_meta_files[0])
    cols = emt.column_names
    print("episode-meta cols:", cols[:12])
    for c in cols:
        if "raw" in c.lower() or "demo" in c.lower():
            raw_key = c
            break
    if raw_key:
        em = emt.to_pydict()
        ep_map = dict(zip(em["episode_index"], em[raw_key]))
        print(f"raw-id column: {raw_key}; sample: {list(ep_map.items())[:3]}")
if not ep_map:
    print("NO raw-id mapping found — assuming episode_index i -> demo_id 10*(i+1)")

data_files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
print(f"{len(data_files)} data files")

# ---- pull proprio per episode for a sample -------------------------------------------------
state_col = None
tbl0 = pq.read_table(data_files[0])
for c in tbl0.column_names:
    if "state" in c or "proprio" in c:
        state_col = c
        break
print("state col:", state_col, "| all cols:", tbl0.column_names[:10])

want = 8  # episodes to check
found = {}
for df in data_files:
    t = pq.read_table(df, columns=["episode_index", state_col])
    eps = t["episode_index"].to_numpy()
    st = np.stack(t[state_col].to_numpy())
    for e in np.unique(eps):
        if len(found) >= want and e not in found:
            continue
        found.setdefault(int(e), []).append(st[eps == e])
    if len(found) >= want:
        break

print(f"\n{'ep_idx':>6} {'demo':>5} {'T_parq':>6} {'T_lbl':>6} {'minL':>7} {'minR':>7} {'closest':>7}")
mins = []
for e in sorted(found)[:want]:
    prop = np.concatenate(found[e])
    demo = ep_map.get(e, 10 * (e + 1))
    try:
        lbl = np.load(f"/root/metalink_labels/ep{demo}.npz")["meta_base"]
    except FileNotFoundError:
        print(f"{e:>6} {demo:>5}  NO LABEL FILE")
        continue
    T = min(len(prop), len(lbl))
    s = int(T * 0.6)
    dl = np.linalg.norm(prop[s:T, EE_L] - lbl[s:T], axis=1).min()
    dr = np.linalg.norm(prop[s:T, EE_R] - lbl[s:T], axis=1).min()
    m = min(dl, dr)
    mins.append(m)
    flag = "" if m < 0.2 else "  <-- SUSPECT"
    print(f"{e:>6} {demo:>5} {len(prop):>6} {len(lbl):>6} {dl:>7.3f} {dr:>7.3f} {m:>7.3f}{flag}")

if mins:
    print(f"\nclosest-arm min over {len(mins)} eps: median {np.median(mins):.3f} m, "
          f"max {max(mins):.3f} m  (expect ~0.05-0.15; >0.3 = label bug)")
