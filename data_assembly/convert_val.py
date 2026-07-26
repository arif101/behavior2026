"""Finish the held-out val set: sample_weight column + feature registration + strata split.

Mirrors the training conversion so the two are directly comparable:
  * sample_weight from the SAME commit definition (tag_contact_commit.py, validated 418/418
    identical to the training tagger)
  * target_points / target_points_mask REGISTERED in info.json but left absent from the
    parquets -- the loader then fills zeros/mask=False, exactly as it does for the 95% of
    training frames that have no labels. Verified safe earlier: zeros, no NaN.
  * only video-complete episodes kept (the failure that cost hours on the training set)
"""
import json, pathlib, collections
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

VAL = pathlib.Path("/root/valA/demos")
TRAIN_INFO = json.load(open("/root/phaseA/b1k_phaseA/meta/info.json"))
COMMIT = json.load(open("/root/valA/contact_commit.json"))
W = COMMIT["meta"]["weight_25pct"]
plan = json.load(open("/root/valA/val_plan.json"))
want = {"heldout_episode": set(plan["heldout_episode"]), "heldout_task": set(plan["heldout_task"])}
print(f"val commit frac {COMMIT['meta']['commit_frac']*100:.2f}% -> weight {W}x", flush=True)

# ---- 1) sample_weight into the parquets -----------------------------------
tot = up = 0
for f in sorted(VAL.glob("data/chunk-*/file-*.parquet")):
    t = pq.read_table(f)
    ep = np.array(t.column("episode_index").to_pylist())
    w = np.ones(len(ep), dtype=np.float32)
    for e in set(ep.tolist()):
        info = COMMIT["by_episode"].get(str(int(e)))
        if not info:
            continue
        rows = np.flatnonzero(ep == e)
        for (a, b) in info["ranges"]:
            sel = rows[a:b] if b <= len(rows) else rows[a:]
            w[sel] = W
    if "sample_weight" in t.column_names:
        t = t.drop_columns(["sample_weight"])
    t = t.append_column("sample_weight", pa.array(w, type=pa.float32()))
    pq.write_table(t, f)
    tot += len(w); up += int((w > 1).sum())
print(f"sample_weight written: {up/max(1,tot)*100:.2f}% of {tot:,} frames upweighted", flush=True)

# ---- 2) register features so the loader fills points for unlabelled frames --
info = json.load(open(VAL / "meta/info.json"))
for k in ("target_points", "target_points_mask", "sample_weight"):
    if k in TRAIN_INFO["features"]:
        info["features"][k] = TRAIN_INFO["features"][k]
json.dump(info, open(VAL / "meta/info.json", "w"), indent=1)
print("registered features:", [k for k in ("target_points","target_points_mask","sample_weight")
                               if k in info["features"]], flush=True)

# ---- 3) keep only video-complete episodes, split by stratum ----------------
vids = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
present = set()
for f in sorted(VAL.glob("data/chunk-*/file-*.parquet")):
    present.update(pq.read_table(f, columns=["episode_index"]).column("episode_index").to_pylist())
import pandas as pd, glob as _g
eps_meta = pd.concat([pd.read_parquet(p) for p in sorted(_g.glob(f"{VAL}/meta/episodes/*/*.parquet"))])
ok = collections.defaultdict(list)
for stratum, ids in want.items():
    for e in sorted(ids & present):
        row = eps_meta[eps_meta.episode_index == e]
        if not len(row):
            continue
        r = row.iloc[0]
        good = True
        for vk in vids:
            p = VAL / f"videos/{vk}/chunk-{int(r[f'videos/{vk}/chunk_index']):03d}/file-{int(r[f'videos/{vk}/file_index']):03d}.mp4"
            if not p.exists():
                good = False; break
        if good:
            ok[stratum].append(int(e))
json.dump(dict(ok), open("/root/valA/val_episodes_ok.json", "w"), indent=1)
for k, v in ok.items():
    print(f"{k}: {len(v)} video-complete episodes", flush=True)
print("VAL_CONVERT_DONE", flush=True)
