#!/usr/bin/env python3
"""Phase-A1 conversion: 1800 episodes -> trainable LeRobot dataset.

Adds, per frame:
  * target_points / target_points_mask  (AdaLN 3D conditioning; real for labelled episodes,
    mask=False elsewhere — mixed supervision, verified supported by add_target_points)
  * sample_weight                        (10.36x on COMMIT frames = gripper open<->close
    transitions, measured at 3.12% of frames; this is what rebalances transit:contact, which
    scaling episodes provably does NOT)
"""
import glob, json, os, subprocess, sys
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

ROOT = "/root/phaseA/demos"
COMMIT = json.load(open("/root/phaseA/contact_commit.json"))
W = COMMIT["meta"]["weight_25pct"]
print(f"commit fraction {COMMIT['meta']['commit_frac']*100:.2f}% -> weight {W}x", flush=True)

# ---------- 1) label symlinks per task (episode-index keyed), where labels exist ----------
maps = {}
for m in glob.glob("/root/labels_dl/g3_pipeline/*_map.json"):
    t = os.path.basename(m).replace("_map.json", "")
    maps[t] = json.load(open(m))
n_lab = 0
for t, mp in maps.items():
    src_dir = f"/root/labels_dl/g3_deep_labels/{t}"
    if not os.path.isdir(src_dir):
        continue
    dst = f"/root/phaseA/labels_sym/{t}"; os.makedirs(dst, exist_ok=True)
    for lid, ei in mp.items():
        s = f"{src_dir}/labels_{lid}.jsonl"
        if os.path.exists(s):
            for suf in ("", ".done.json"):
                d = f"{dst}/labels_{ei}.jsonl{suf}"
                if os.path.lexists(d): os.remove(d)
                if os.path.exists(s + suf): os.symlink(s + suf, d)
            n_lab += 1
print(f"label symlinks: {n_lab} episodes across {len(maps)} gate tasks", flush=True)

# ---------- 2) add_target_points per labelled task (chained; videos hardlinked) ----------
src = ROOT
outs = []
tt = "/root/labels_dl/g3_pipeline/task_targets.json"
for i, t in enumerate(sorted(maps)):
    ldir = f"/root/phaseA/labels_sym/{t}"
    if not os.path.isdir(ldir): continue
    out = f"/root/phaseA/conv_{i}" if i < len(maps)-1 else "/root/phaseA/b1k_phaseA"
    r = subprocess.run([f"/root/openpi/.venv/bin/python", "/root/openpi/scripts/b1k/add_target_points.py",
                        "--dataset", src, "--labels-dir", ldir, "--task", t,
                        "--task-targets", tt, "--out", out, "--overwrite"],
                       capture_output=True, text=True)
    tail = "\n".join(r.stdout.splitlines()[-4:])
    print(f"--- {t} exit={r.returncode}\n{tail}", flush=True)
    if r.returncode != 0:
        print(r.stderr[-1500:], flush=True); sys.exit(1)
    if src.startswith("/root/phaseA/conv_"): subprocess.run(["rm","-rf",src])
    src = out; outs.append(out)
FINAL = src
print("converted dataset:", FINAL, flush=True)

# ---------- 3) sample_weight column from measured commit ranges ----------
files = sorted(glob.glob(f"{FINAL}/data/chunk-*/file-*.parquet"))
tot_w = tot_n = 0
for k, f in enumerate(files):
    t = pq.read_table(f)
    ep = np.array(t.column("episode_index").to_pylist())
    w = np.ones(len(ep), dtype=np.float32)
    for e in set(ep.tolist()):
        info = COMMIT["by_episode"].get(str(int(e)))
        if not info: continue
        rows = np.flatnonzero(ep == e)
        for (a, b) in info["ranges"]:
            sel = rows[a:b] if b <= len(rows) else rows[a:]
            w[sel] = W
    if "sample_weight" in t.column_names:
        t = t.drop_columns(["sample_weight"])
    t = t.append_column("sample_weight", pa.array(w, type=pa.float32()))
    pq.write_table(t, f)
    tot_w += float((w > 1).sum()); tot_n += len(w)
    if (k+1) % 20 == 0: print(f"  weights {k+1}/{len(files)}", flush=True)
print(f"sample_weight written: {tot_w/tot_n*100:.2f}% of frames upweighted {W}x", flush=True)
print("PHASEA_CONVERT_DONE", FINAL, flush=True)
