"""Plan the held-out val set. Two strata, mirroring the grounding head's convention:
  heldout-EPISODE : unseen episodes from tasks the model HAS trained on  -> instance generalization
  heldout-TASK    : episodes from tasks never trained on                 -> task generalization
Prefers episodes whose data parquets are ALREADY local (the 204 excluded for missing video),
so the fetch only needs their video chunks."""
import glob, json
import pandas as pd

md = "/root/phaseA/demos_meta"
eps = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{md}/meta/episodes/*/*.parquet"))])
tdf = pd.read_parquet(f"{md}/meta/tasks.parquet")
name2idx = {str(k): int(v) for k, v in tdf["task_index"].items()}
idx2name = {v: k for k, v in name2idx.items()}
trained = set(json.load(open("/root/phaseA/tasklist.json")))
plan = json.load(open("/root/phaseA/phaseA_plan.json"))
used = set()
for t, d in plan.items():
    used.update(d["episode_indices"])

# episodes already sitting in local parquets but never trained on (data present, video absent)
local_eps = set()
import pyarrow.parquet as pq, pathlib
for f in pathlib.Path("/root/phaseA/b1k_phaseA/data").rglob("*.parquet"):
    local_eps.update(pq.read_table(f, columns=["episode_index"]).column("episode_index").to_pylist())
free = sorted((local_eps - used))
print(f"episodes with LOCAL data but never trained: {len(free)}")

ho_ep, ho_task = [], []
for e in free:
    row = eps[eps.episode_index == e]
    if len(row) and idx2name.get(int(row.iloc[0].task_index)) in trained:
        ho_ep.append(int(e))
ho_ep = ho_ep[:150]

# held-out TASKS: 6 unseen tasks x 20 contiguous episodes (contiguous => fewest distinct files)
unseen = sorted(t for t in name2idx if t not in trained)[:6]
for t in unseen:
    b = eps[eps.task_index == name2idx[t]].sort_values("episode_index")
    ho_task += [int(e) for e in b.head(20).episode_index.tolist()]

out = {"heldout_episode": ho_ep, "heldout_task": ho_task, "heldout_task_names": unseen}
json.dump(out, open("/root/phaseA/val_plan.json", "w"), indent=1)
print(f"heldout-EPISODE: {len(ho_ep)} eps (videos need fetching, data already local)")
print(f"heldout-TASK   : {len(ho_task)} eps across {len(unseen)} unseen tasks: {unseen}")
