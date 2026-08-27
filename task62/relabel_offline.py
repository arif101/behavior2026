"""Offline CONTEXT labels for task-62 — no simulator (2026-08-27, replaces relabel_v1's sim passes).
  (a) AG-held object per arm per frame, decoded from the recorded state vector: OmniGibson appends
      [_AG_MAGIC, arm_idx, obj_uuid, link_idx, 14 frame floats, joint_type] per grasping arm
      (robot.py serialize()); uuid = float32(md5(name) % 1e8) resolves against the scene template.
      Objects absent from the template are the egg halves spawned at the slice.
  (b) progress q(t): the organizers' scoring segments (chop +0.4, place-in knife +0.2, place-on half
      +0.2 x2) in annotation order, each matched to the nearest POSITIVE reward spike of the right
      size; q steps at the spike frame. Negative/flicker rewards are ignored. Demos without a chop
      spike or without task_success are flagged (exclude from training).
Output: {out}/ep{raw}.npz with ag_L, ag_R (0 none, 1 egg/half, 2 knife, 3 other), q, uuid_L, uuid_R,
spike frames; summary_offline.json in relabel_v1's summary format (seg_ok) so build_t62_dataset.py
--v1 consumes it unchanged.
Usage: python task62/relabel_offline.py [--out /root/step0/relabel_offline] [--compare /root/step0/relabel_v1]
"""
import os, json, glob, argparse, collections
import numpy as np, h5py
from hashlib import md5

RAW = "/root/rawdemos/task-0062"; ANN = "/root/t62_annotations/annotations/task-0062"
SCENE = "/root/bw/BEHAVIOR-1K/datasets/2026-challenge-task-instances/scenes/house_single_floor/json/house_single_floor_task_halve_an_egg_0_0_template.json"
MAGIC = 1e8 + 123456
def get_uuid(name, n=8): return int(np.float32(int(md5(name.encode()).hexdigest(), 16) % (10 ** n)))

def load(ep):
    f = h5py.File(f"{RAW}/episode_{ep:08d}.hdf5", "r"); g = f["data"][max(f["data"], key=lambda k: f["data"][k]["action"].shape[0])]
    return dict(reward=g["reward"][:], state=g["state"][:], ss=g["state_size"][:], T=g["action"].shape[0],
                ok=str(g.attrs.get("completion_reason")) == "task_success",
                ann=json.load(open(f"{ANN}/episode_{ep:08d}.json"))["skill_annotation"])

def grasp_labels(d, uuid2name):
    T = d["T"]; uid = np.zeros((T, 2), np.int64); cls = np.zeros((T, 2), np.int8)
    for t in range(T):
        row = d["state"][t, :int(d["ss"][t])]; pos = np.nonzero(np.abs(row - MAGIC) < 0.5)[0]
        for p in pos:
            arm = int(round(row[p + 1])); u = int(round(row[p + 2]))
            if arm not in (0, 1): continue
            uid[t, arm] = u; name = uuid2name.get(u)
            cls[t, arm] = 2 if (name and "knife" in name) else 1 if (name and "egg" in name) else 3   # 3 = fixture/other (fridge door) or unresolved
    return uid, cls

def seg_value(s):
    d = s["skill_description"][0]; o = s["manipulating_object_id"][0] if s["manipulating_object_id"] else ""
    if d == "chop": return 0.4
    if d == "place in" and o == "carving_knife": return 0.2
    if d == "place on" and o == "half_hard_boiled_egg": return 0.2
    return None

def progress_labels(d):
    r = d["reward"]; T = d["T"]; q = np.zeros(T, np.float32); seg_ok = {}; spikes = []
    pos = [(int(t), round(float(r[t]), 1)) for t in np.nonzero(r > 0)[0]]; used = set(); cur = 0.0
    for s in d["ann"]:
        v = seg_value(s)
        if v is None: continue
        a, b = s["frame_duration"]; best = None
        for i, (t, val) in enumerate(pos):
            # a place-on segment may also be satisfied by a +0.4 spike: both halves' predicates flipping in
            # the same frame (2 demos); the cap at 1.0 absorbs the double credit
            if i in used or not (abs(val - v) < 0.01 or (v == 0.2 and s["skill_description"][0] == "place on" and abs(val - 0.4) < 0.01)): continue
            dist = 0 if a <= t <= b else min(abs(t - a), abs(t - b))
            if dist <= 600 and (best is None or dist < best[0]): best = (dist, i, t)
        key = f"{s['skill_description'][0].replace(' ', '_')}#{s['skill_idx']}"
        if best is None: seg_ok[key] = dict(ok=False, why="no matching +spike within 600 frames"); continue
        used.add(best[1]); t = best[2]; v = pos[best[1]][1]; cur = min(1.0, cur + v); q[t:] = cur; spikes.append((t, v))
        seg_ok[key] = dict(ok=True, spike=t, offset_from_seg=int(best[0]), q1=cur)
    return q, seg_ok, spikes

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/root/step0/relabel_offline"); ap.add_argument("--compare", default="/root/step0/relabel_v1")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    scene = json.load(open(SCENE)); objs = scene["objects_info"]["init_info"] if "objects_info" in scene else scene["objects"]
    uuid2name = {get_uuid(n): n for n in objs}
    for n in list(objs):                              # halves spawned by the SlicingRule: half_<name>_<i>
        if "egg" in n: uuid2name.update({get_uuid(f"half_{n}_{i}"): f"half_{n}_{i}" for i in range(2)})
    summary = {}; stats = collections.Counter(); unknown = collections.Counter(); cmp = []
    for p in sorted(glob.glob(f"{RAW}/episode_*.hdf5")):
        ep = int(p.split("_")[-1].split(".")[0]); d = load(ep)
        uid, cls = grasp_labels(d, uuid2name); q, seg_ok, spikes = progress_labels(d)
        for u in np.unique(uid[uid > 0]):
            if u not in uuid2name: unknown[int(u)] += 1
        chop = [t for t, v in spikes if v == 0.4]
        usable = d["ok"] and bool(chop) and all(v["ok"] for v in seg_ok.values())
        np.savez(f"{a.out}/ep{ep}.npz", ag_L=cls[:, 0], ag_R=cls[:, 1], q=q, uuid_L=uid[:, 0], uuid_R=uid[:, 1],
                 spike_frames=np.array([t for t, _ in spikes]), spike_vals=np.array([v for _, v in spikes]), usable=usable)
        summary[ep] = dict(q_final=float(q[-1]), usable=usable, task_success=d["ok"], chop_frame=chop[0] if chop else None,
                           agL_frac=float((cls[:, 0] > 0).mean()), agR_frac=float((cls[:, 1] > 0).mean()),
                           held_names=sorted({uuid2name.get(int(u), f"unknown_{int(u)}") for u in np.unique(uid[uid > 0])}), seg_ok=seg_ok)
        stats["usable" if usable else "flagged"] += 1; stats[f"q={q[-1]:.1f}"] += 1
        sp = f"{a.compare}/ep{ep}.npz"
        if os.path.exists(sp):
            v1 = np.load(sp); n = min(len(q), len(v1["q"]))
            agree = [float(((cls[:n, j] > 0) == (v1[k][:n] > 0)).mean()) for j, k in enumerate(("ag_L", "ag_R"))]
            cmp.append((ep, agree, [int(t) for t in np.nonzero(np.diff(v1["q"][:n]))[0] + 1], [t for t, _ in spikes], float(v1["q"][n - 1]), float(q[n - 1])))
    json.dump(summary, open(f"{a.out}/summary_offline.json", "w"), indent=1)
    print("demos:", dict(stats)); print("uuids not in template (egg halves expected):", dict(unknown))
    for ep, agree, sim_steps, off_steps, sq, oq in cmp:
        print(f"  vs sim ep{ep}: held-agreement L={agree[0]:.3f} R={agree[1]:.3f} | q steps sim={sim_steps} offline={off_steps} | q_final sim={sq:.1f} offline={oq:.1f}")
    print("OFFLINE_DONE")

if __name__ == "__main__": main()
