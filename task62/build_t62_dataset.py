"""Build the task-62 LeRobot training dataset with v0 CONTEXT labels (CONTEXT_STATES_SPEC_62 §2).

Input : /root/t62_lerobot_src   (behavior-1k/2026-challenge-demos, task-62 shards + videos + meta)
        /root/t62_annotations   (organizers' skill annotations + meta/episodes)
Output: /root/b1k_t62           (data shards with `context`/`context_weight`/`stage` columns,
                                 trimmed meta, videos symlinked)

v0 labels are computed OFFLINE from annotations + the parquet action/reward columns:
  context[0:8]   skill family one-hot {navigate, open_door, pick_up_from, place_on, chop, place_in,
                 close_door, other} (also written as int `stage` for the stage head, stage_classes=8)
  context[8]     progress q(t): annotation-derived (chop end +0.4, place_in knife end +0.2,
                 each place_on half end +0.2, capped at 1.0) — v1 replaces with sim goal_status
  context[9]     post_slice flag (t >= chop segment end)
  context[10:12] AG-held per arm [L, R], from pick_up_from closure -> matching place release
  context[12:16] held-object class per arm [L:egg/half, L:knife, R:egg/half, R:knife]
  context_weight 1.0 (v1: 0.5 on segments whose near-anchor replay fails to reproduce the human)

Frame alignment: meta/episodes gives episode_index -> raw_episode_id; LeRobot length == raw T for
198/200 demos. Episodes with |length - annotation_end| > --max_skew are DROPPED loudly.
"""
import os, json, glob, shutil, argparse, pathlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

FAMILIES = ["navigate", "open_door", "pick_up_from", "place_on", "chop", "place_in", "close_door", "other"]
FAM_OF = {"move to": 0, "open door": 1, "pick up from": 2, "place on": 3, "chop": 4, "place in": 5, "close door": 6, "push to": 7}
GL, GR = 14, 22
CTX_DIM = 16


def closures(a, lo, hi, ch, closing=True):
    s = np.sign(a[lo:hi, ch])
    if closing: c = np.nonzero((s[1:] < 0) & (s[:-1] >= 0))[0] + 1 + lo
    else:       c = np.nonzero((s[1:] > 0) & (s[:-1] <= 0))[0] + 1 + lo
    return c.tolist()


def labels_for_episode(T, ann, act):
    fam = np.full(T, 7, np.int32)
    prog = np.zeros(T, np.float32); post = np.zeros(T, np.float32)
    ag = np.zeros((T, 2), np.float32); cls = np.zeros((T, 2, 2), np.float32)   # [arm][eggish, knife]
    holding = {0: None, 1: None}    # arm -> (obj_class_idx)
    for s in ann:
        k = s["skill_description"][0]; a, b = s["frame_duration"]; a, b = max(0, a), min(T, b)
        if b <= a: continue
        fam[a:b] = FAM_OF.get(k, 7)
        obj = s["object_id"][0][0] if s["object_id"] else ""
        if k == "chop":
            prog[b:] += 0.4; post[b:] = 1.0
        elif k == "place in" and "knife" in obj:
            prog[b:] += 0.2
        elif k == "place on" and "half" in obj:
            prog[b:] += 0.2
        if k == "pick up from":
            cL, cR = closures(act, a, b, GL), closures(act, a, b, GR)
            if cL or cR:
                arm, c = (0, cL[0]) if (cL and (not cR or cL[0] <= cR[0])) else (1, cR[0])
            else:
                arm, c = (1 if "egg" in obj and "half" not in obj else 0), b - 1     # census default
            holding[arm] = (c, 1 if "knife" in obj else 0)
        if k in ("place on", "place in"):
            # release: the held arm whose gripper opens inside the segment, else at segment end
            for arm in (0, 1):
                if holding[arm] is None: continue
                ch = GL if arm == 0 else GR
                op = closures(act, a, b, ch, closing=False)
                r = op[0] if op else b
                c0, kind = holding[arm]
                ag[c0:r, arm] = 1.0; cls[c0:r, arm, kind] = 1.0
                holding[arm] = None
    for arm in (0, 1):                      # still holding at episode end
        if holding[arm] is not None:
            c0, kind = holding[arm]; ag[c0:, arm] = 1.0; cls[c0:, arm, kind] = 1.0
    prog = np.clip(prog, 0.0, 1.0)
    onehot = np.eye(8, dtype=np.float32)[fam]
    ctx = np.concatenate([onehot, prog[:, None], post[:, None], ag, cls.reshape(T, 4)], axis=1).astype(np.float32)
    assert ctx.shape == (T, CTX_DIM)
    return ctx, fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/t62_lerobot_src")
    ap.add_argument("--ann", default="/root/t62_annotations")
    ap.add_argument("--out", default="/root/b1k_t62")
    ap.add_argument("--max_skew", type=int, default=4)
    ap.add_argument("--v1", default=None, help="dir of relabel_v1.py outputs (ep{raw}.npz); overrides progress/AG/"
                                                "held-class channels and sets context_weight from seg_ok")
    a = ap.parse_args()
    src, ann, out = pathlib.Path(a.src), pathlib.Path(a.ann), pathlib.Path(a.out)
    meta = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{ann}/meta/episodes/chunk-*/file-*.parquet"))], ignore_index=True)
    t62 = meta[meta.task_index == 62].set_index("episode_index")
    print(f"task-62 episodes in meta: {len(t62)}")

    (out / "data").mkdir(parents=True, exist_ok=True)
    report = dict(episodes=[], dropped=[], skew={}, frames=0, family_frac={})
    fam_counts = np.zeros(8, np.int64)
    kept_eps = []            # ORIGINAL episode_index values kept, in order of appearance (sorted)
    # LeRobot's cache check requires episode_index == range(total_episodes) and a contiguous global
    # `index`; the source keeps global ids (12400..). Re-index densely, sorted by original index.
    new_idx = {}; frame_base = 0
    for sf in sorted(glob.glob(f"{src}/data/chunk-*/file-*.parquet")):
        t = pq.read_table(sf); df = t.to_pandas()
        N = len(df); ctx_all = np.zeros((N, CTX_DIM), np.float32); stage_all = np.full(N, 7, np.int32); w_all = np.ones(N, np.float32)
        keep = np.zeros(N, bool); ep_new = np.zeros(N, np.int64)
        act_all = np.stack(df["action"].values).astype(np.float32)
        for e in sorted(df["episode_index"].unique()):
            if int(e) not in t62.index:
                continue
            row = t62.loc[int(e)]; sel = np.where(df["episode_index"].values == e)[0]
            T = len(sel); raw = int(row.raw_episode_id)
            skills = json.load(open(f"{ann}/{row.annotation_path}"))["skill_annotation"]
            ann_end = max(s["frame_duration"][1] for s in skills)
            skew = T - ann_end; report["skew"][str(raw)] = int(skew)
            if skew < -a.max_skew:      # annotation extends beyond the LeRobot episode (truncated video)
                report["dropped"].append(dict(episode=int(e), raw=raw, why=f"skew {skew}")); continue
            act = act_all[sel]
            ctx, fam = labels_for_episode(T, skills, act)
            if a.v1 and os.path.exists(f"{a.v1}/ep{raw}.npz"):
                v1 = np.load(f"{a.v1}/ep{raw}.npz"); n = min(T, len(v1["q"]))
                ctx[:n, 8] = np.clip(v1["q"][:n], 0, 1)                       # progress from sim goal_status
                ctx[:n, 9] = (v1["q"][:n] >= 0.4).astype(np.float32)          # post-slice = both real(half) satisfied
                for j, key in enumerate(("ag_L", "ag_R")):
                    c = v1[key][:n].astype(np.int64)
                    ctx[:n, 10 + j] = (c > 0).astype(np.float32)
                    ctx[:n, 12 + 2 * j] = (c == 1).astype(np.float32); ctx[:n, 13 + 2 * j] = (c == 2).astype(np.float32)
                summ = glob.glob(f"{a.v1}/summary_*.json"); ok_map = {}
                for sp_ in summ:
                    ok_map.update({k: v for k, v in json.load(open(sp_)).get(str(raw), {}).get("seg_ok", {}).items()})
                for s in skills:                                              # segment weight: 0.5 where replay failed
                    key = None
                    for tag_key, v in ok_map.items():
                        if tag_key.endswith(f"#{s['skill_idx']}"): key = tag_key
                    if key is not None and not ok_map[key].get("ok", True):
                        aa, bb = s["frame_duration"]; w_all[sel[max(0, aa):min(T, bb)]] = 0.5
                report.setdefault("v1_episodes", []).append(raw)
            ctx_all[sel] = ctx; stage_all[sel] = fam; keep[sel] = True
            new_idx[int(e)] = len(kept_eps); ep_new[sel] = new_idx[int(e)]
            fam_counts += np.bincount(fam, minlength=8); report["frames"] += T; report["episodes"].append(raw); kept_eps.append(int(e))
        t = t.filter(pa.array(keep))
        n = t.num_rows
        t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index", pa.array(ep_new[keep], type=pa.int64()))
        t = t.set_column(t.schema.get_field_index("index"), "index", pa.array(np.arange(frame_base, frame_base + n), type=pa.int64()))
        frame_base += n
        t = t.append_column("context", pa.array(list(ctx_all[keep]), type=pa.list_(pa.float32(), CTX_DIM)))
        t = t.append_column("context_weight", pa.array(w_all[keep], type=pa.float32()))
        t = t.append_column("stage", pa.array(stage_all[keep], type=pa.int32()))
        dst = out / pathlib.Path(sf).relative_to(src); dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, dst); print(f"  wrote {dst} rows={t.num_rows}")

    # ---- meta: trimmed info.json, filtered episodes, tasks, videos symlink ----
    (out / "meta").mkdir(exist_ok=True)
    info = json.load(open(src / "meta" / "info.json"))
    info["total_episodes"] = len(kept_eps); info["total_frames"] = int(report["frames"]); info["splits"] = {"train": f"0:{len(kept_eps)}"}
    info["features"]["context"] = {"dtype": "float32", "shape": [CTX_DIM], "names": None}
    info["features"]["context_weight"] = {"dtype": "float32", "shape": [1], "names": None}
    info["features"]["stage"] = {"dtype": "int32", "shape": [1], "names": None}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    for f in ("tasks.parquet", "tasks.jsonl", "stats.json"):
        if (src / "meta" / f).exists(): shutil.copy(src / "meta" / f, out / "meta" / f)
    # meta/episodes: one shard, rows re-indexed to the dense episode ids + contiguous frame ranges
    ep_dir = out / "meta" / "episodes"; shutil.rmtree(ep_dir, ignore_errors=True)
    rows = []
    for p in sorted(glob.glob(f"{ann}/meta/episodes/chunk-*/file-*.parquet")):
        d = pd.read_parquet(p); rows.append(d[d["episode_index"].isin(kept_eps)])
    epdf = pd.concat(rows, ignore_index=True)
    epdf["episode_index"] = epdf["episode_index"].map(new_idx).astype("int64")
    epdf = epdf.sort_values("episode_index").reset_index(drop=True)
    starts = np.concatenate([[0], np.cumsum(epdf["length"].values)[:-1]])
    epdf["dataset_from_index"] = starts.astype("int64"); epdf["dataset_to_index"] = (starts + epdf["length"].values).astype("int64")
    epdf["meta/episodes/chunk_index"] = 0; epdf["meta/episodes/file_index"] = 0
    (ep_dir / "chunk-000").mkdir(parents=True, exist_ok=True)
    epdf.to_parquet(ep_dir / "chunk-000" / "file-000.parquet", index=False)
    assert int(epdf["dataset_to_index"].iloc[-1]) == report["frames"], "frame count mismatch between data and meta"
    if not (out / "videos").exists(): os.symlink(src / "videos", out / "videos")
    report["family_frac"] = {FAMILIES[i]: round(float(fam_counts[i] / max(1, fam_counts.sum())), 4) for i in range(8)}
    json.dump(report, open(out / "build_report.json", "w"), indent=1)
    print(f"episodes kept {len(kept_eps)}, dropped {len(report['dropped'])}, frames {report['frames']:,}")
    print("family frac:", report["family_frac"]); print("skew:", {k: v for k, v in report["skew"].items() if v != 0})
    print("T62_DATASET_BUILT")


if __name__ == "__main__":
    main()
