"""Pod-side data prep: download a task subset of behavior-1k/2026-challenge-demos
(LeRobot v3) + annotations, resolve per-episode spans from meta, write the
build_cache.py manifest.

Usage (on the GPU box):
  python prep_data.py --tasks turning_on_radio picking_up_trash \
      --eps_per_task 5 --root /root/data/demos --manifest /root/data/manifest.json

Episode meta schema (verified 2026-07-17 on meta/episodes/chunk-000/file-000):
  tasks=[underscored_name], task_index, episode_index, length, annotation_path,
  data/{chunk_index,file_index} + dataset_{from,to}_index (global rows; the
  data parquet also carries episode_index per row -- build_cache filters on it),
  videos/<key>/{chunk_index,file_index} + {from,to}_timestamp (seconds @30fps
  -> within-file frame span = round(ts*30)).
"""

import argparse
import json
import os
import subprocess

REPO = "behavior-1k/2026-challenge-demos"
ZED_RGB = "observation.rgb.zed_link_camera_0"
ZED_DEP = "observation.depth_linear.zed_link_camera_0"
FPS = 30


def hf_get(path, root):
    dst = os.path.join(root, path)
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    subprocess.run(["curl", "-sfL", "--retry", "3", url, "-o", dst + ".part"],
                   check=True)
    os.rename(dst + ".part", dst)
    return dst


def hf_ls(path):
    out = subprocess.run(
        ["curl", "-sfL",
         f"https://huggingface.co/api/datasets/{REPO}/tree/main/{path}"],
        capture_output=True, check=True)
    return json.loads(out.stdout)


def load_episode_meta(root, tasks_wanted):
    """Scan all meta/episodes files (~100 x 60KB), keep rows for wanted tasks."""
    import pandas as pd
    frames = []
    for chunk in hf_ls("meta/episodes"):
        for f in hf_ls(chunk["path"]):
            df = pd.read_parquet(hf_get(f["path"], root))
            df["task_name"] = df["tasks"].map(lambda t: t[0])
            hit = df[df["task_name"].isin(tasks_wanted)]
            if len(hit):
                frames.append(hit)
    return pd.concat(frames, ignore_index=True) if frames else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True,
                    help="underscored task names (turning_on_radio)")
    ap.add_argument("--eps_per_task", type=int, default=5)
    ap.add_argument("--root", default="/root/data/demos")
    ap.add_argument("--manifest", default="/root/data/manifest.json")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)

    ep = load_episode_meta(args.root, set(args.tasks))
    assert ep is not None, f"no episodes found for {args.tasks}"
    print(f"episode meta: {len(ep)} rows for {sorted(set(ep['task_name']))}")

    manifest = []
    for task in args.tasks:
        rows = ep[ep["task_name"] == task].sort_values("episode_index")
        rows = rows.head(args.eps_per_task)
        for _, r in rows.iterrows():
            vc = int(r[f"videos/{ZED_RGB}/chunk_index"])
            vf = int(r[f"videos/{ZED_RGB}/file_index"])
            v0 = round(float(r[f"videos/{ZED_RGB}/from_timestamp"]) * FPS)
            v1 = round(float(r[f"videos/{ZED_RGB}/to_timestamp"]) * FPS)
            vsub = f"chunk-{vc:03d}/file-{vf:03d}.mp4"
            dc = int(r[f"videos/{ZED_DEP}/chunk_index"])
            dfi = int(r[f"videos/{ZED_DEP}/file_index"])
            dsub = f"chunk-{dc:03d}/file-{dfi:03d}.mp4"
            # depth files are chunked independently of rgb -- their within-file
            # frame span comes from the DEPTH stream's own timestamps
            d0 = round(float(r[f"videos/{ZED_DEP}/from_timestamp"]) * FPS)
            d1 = round(float(r[f"videos/{ZED_DEP}/to_timestamp"]) * FPS)
            dpath = (f"data/chunk-{int(r['data/chunk_index']):03d}/"
                     f"file-{int(r['data/file_index']):03d}.parquet")
            manifest.append(dict(
                task=task, file_idx=int(r["episode_index"]),
                episode_index=int(r["episode_index"]),
                length=int(r["length"]),
                rgb=hf_get(f"videos/{ZED_RGB}/{vsub}", args.root),
                depth=hf_get(f"videos/{ZED_DEP}/{dsub}", args.root),
                parquet=hf_get(dpath, args.root),
                annot=hf_get(str(r["annotation_path"]), args.root)
                    if "annotation_path" in r and r["annotation_path"]
                    else None,
                vrange=[v0, v1],
                drange=[d0, d1],
                prange=[int(r["dataset_from_index"]), int(r["dataset_to_index"])],
            ))
        print(f"{task}: {len(rows)} episodes queued")

    json.dump(manifest, open(args.manifest, "w"), indent=1)
    print(f"{len(manifest)} episodes -> {args.manifest}")


if __name__ == "__main__":
    main()
