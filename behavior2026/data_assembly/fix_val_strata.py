"""Drop declared-but-absent depth video features from the val info.json, then recompute
video-complete episodes per stratum. Mirrors the fix already applied to the training set:
LeRobot rejects a load if a declared video stream has no file, and reports it as an HF 401."""
import collections, glob, json, pathlib

import pandas as pd
import pyarrow.parquet as pq

VAL = pathlib.Path("/root/valA/demos")
info = json.load(open(VAL / "meta/info.json"))
drop = [k for k, v in info["features"].items() if v.get("dtype") == "video" and "depth" in k]
for k in drop:
    info["features"].pop(k)
json.dump(info, open(VAL / "meta/info.json", "w"), indent=1)
vids = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
print("dropped depth video features:", len(drop))
print("remaining video keys:", vids)

plan = json.load(open("/root/valA/val_plan.json"))
want = {"heldout_episode": set(plan["heldout_episode"]), "heldout_task": set(plan["heldout_task"])}
present = set()
for f in sorted(VAL.glob("data/chunk-*/file-*.parquet")):
    present.update(pq.read_table(f, columns=["episode_index"]).column("episode_index").to_pylist())
em = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{VAL}/meta/episodes/*/*.parquet"))])

ok = collections.defaultdict(list)
for stratum, ids in want.items():
    for e in sorted(ids & present):
        row = em[em.episode_index == e]
        if not len(row):
            continue
        r = row.iloc[0]
        good = True
        for vk in vids:
            ci = int(r["videos/" + vk + "/chunk_index"])
            fi = int(r["videos/" + vk + "/file_index"])
            if not (VAL / "videos" / vk / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4").exists():
                good = False
                break
        if good:
            ok[stratum].append(int(e))

json.dump(dict(ok), open("/root/valA/val_episodes_ok.json", "w"), indent=1)
for k, v in ok.items():
    print(f"{k}: {len(v)} video-complete episodes")
print("total val episodes:", sum(len(v) for v in ok.values()))
