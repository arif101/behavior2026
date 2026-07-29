"""Augment the radio LeRobot dataset with FOVEATED MEMORY columns (Run-1 data assembly).

Follows the add_target_points.py pattern (copy dataset, append parquet columns, register
features in meta/info.json, norm-stats untouched — raw values bypass normalization). Videos are
HARDLINKED (cp -al), not copied.

New columns, joined via the VERIFIED episode map (raw_episode_id 10..3000 — the naive id
assumption was wrong for 192/200 episodes):

  target_points       f32 flat[6]  [meta_base - ee_l, meta_base - ee_r]  (METALINK-corrected
                                   AdaLN labels — the trigger volume, not the object center;
                                   same contract as add_target_points.py: base frame, per arm)
  target_points_mask  bool[2]      always [True, True] (metalink defined every frame)
  map_tokens_full     f32 flat[576]  FoveatedMap.query() with target writes, (8,72) row-major
  map_tokens_blind    f32 flat[576]  target-derived fields zeroed (anti-shortcut stream)

Alignment: map tokens were built FROM the parquet cam poses (already parquet-aligned);
metalink labels have T_parquet + 1 frames (pose JSONs carry the reset frame) -> use [:T].

Usage:  python add_map_labels.py --src /root/b1k_radio2 --out /root/b1k_radio_map
"""

import argparse
import glob
import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EE_L, EE_R = slice(17, 20), slice(42, 45)
K, D = 8, 72


def copy_dataset(src: pathlib.Path, dst: pathlib.Path, overwrite: bool):
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists (pass --overwrite)")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    shutil.copytree(src / "meta", dst / "meta")
    shutil.copytree(src / "data", dst / "data")
    subprocess.run(["cp", "-al", str(src / "videos"), str(dst / "videos")], check=True)


def register_features(root: pathlib.Path):
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text())
    info["features"]["target_points"] = {"dtype": "float32", "shape": [6], "names": None}
    info["features"]["target_points_mask"] = {"dtype": "bool", "shape": [2], "names": None}
    info["features"]["map_tokens_full"] = {"dtype": "float32", "shape": [K * D], "names": None}
    info["features"]["map_tokens_blind"] = {"dtype": "float32", "shape": [K * D], "names": None}
    p.write_text(json.dumps(info, indent=4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/b1k_radio2")
    ap.add_argument("--out", default="/root/b1k_radio_map")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    src, out = pathlib.Path(a.src), pathlib.Path(a.out)

    emap = json.load(open("/root/episode_map.json"))
    ep2demo = {int(i): int(d) for i, d in emap["mapping"].items()}

    print("copying dataset (videos hardlinked)...", flush=True)
    copy_dataset(src, out, a.overwrite)

    cache = {}

    def demo_arrays(demo):
        if demo not in cache:
            mt = np.load(f"/root/map_tokens/ep{demo}.npz")
            ml = np.load(f"/root/metalink_labels/ep{demo}.npz")["meta_base"]
            cache[demo] = (mt["tokens_full"].astype(np.float32),
                           mt["tokens_notarget"].astype(np.float32), ml)
        return cache[demo]

    n_rows = 0
    for fp in sorted(glob.glob(str(out / "data" / "**" / "*.parquet"), recursive=True)):
        t = pq.read_table(fp)
        eps = t["episode_index"].to_numpy()
        fis = t["frame_index"].to_numpy()
        state = np.stack(t["observation.state"].to_numpy())
        N = len(eps)
        tp = np.zeros((N, 6), np.float32)
        tpm = np.ones((N, 2), bool)
        mtf = np.zeros((N, K * D), np.float32)
        mtb = np.zeros((N, K * D), np.float32)
        for r in range(N):
            demo = ep2demo[int(eps[r])]
            full, blind, meta = demo_arrays(demo)
            fi = int(fis[r])
            assert fi < len(full) and fi < len(meta), (demo, fi, len(full), len(meta))
            mtf[r] = full[fi].reshape(-1)
            mtb[r] = blind[fi].reshape(-1)
            tp[r, :3] = meta[fi] - state[r, EE_L]
            tp[r, 3:] = meta[fi] - state[r, EE_R]
        table = t
        for name, arr in (("target_points", pa.array(list(tp), pa.list_(pa.float32(), 6))),
                          ("target_points_mask", pa.array(list(tpm), pa.list_(pa.bool_(), 2))),
                          ("map_tokens_full", pa.array(list(mtf), pa.list_(pa.float32(), K * D))),
                          ("map_tokens_blind", pa.array(list(mtb), pa.list_(pa.float32(), K * D)))):
            if name in table.column_names:
                table = table.drop_columns([name])
            table = table.append_column(name, arr)
        pq.write_table(table, fp)
        n_rows += N
        print(f"  {os.path.basename(fp)}: {N} rows", flush=True)

    register_features(out)
    # spot validation on a random row of the last file
    v = pq.read_table(fp).slice(N // 2, 1)
    demo = ep2demo[int(v["episode_index"][0].as_py())]
    fi = int(v["frame_index"][0].as_py())
    full, _, meta = demo_arrays(demo)
    got = np.asarray(v["map_tokens_full"][0].as_py(), np.float32)
    assert np.allclose(got, full[fi].reshape(-1), atol=1e-6), "join validation FAILED"
    print(f"DONE: {n_rows} rows augmented; spot join (demo {demo} f{fi}) VERIFIED -> {out}")


if __name__ == "__main__":
    main()
