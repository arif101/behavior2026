"""Extract per-instance world displacement per episode from the label jsonl
(subsampled rows) -> CACHE/<task>/epXXX/disp.npy float32 [K] (meters, norm of
the per-axis position range). Feeds the static/moving occlusion rule."""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mt_common import CACHE, all_tasks, labels_path

ROW_STRIDE = 10   # labels have 2 rows/frame; every 10th row = every 5th frame


def one(task, file_idx, demo):
    out = os.path.join(CACHE, task, "ep%03d" % file_idx, "disp.npy")
    lab = np.load(os.path.join(CACHE, task, "ep%03d" % file_idx, "lab.npz"))
    names = [str(n) for n in lab["obj_names"]]
    lo = {o: None for o in names}
    hi = {o: None for o in names}
    with open(labels_path(task, demo)) as f:
        for li, line in enumerate(f):
            if li % ROW_STRIDE:
                continue
            r = json.loads(line)
            for o in names:
                p = np.asarray(r["objs"][o], dtype=np.float64)
                lo[o] = p if lo[o] is None else np.minimum(lo[o], p)
                hi[o] = p if hi[o] is None else np.maximum(hi[o], p)
    disp = np.array([float(np.linalg.norm(hi[o] - lo[o])) for o in names],
                    dtype=np.float32)
    np.save(out, disp)
    return "%s/ep%03d " % (task, file_idx) + " ".join(
        "%s=%.2f" % (n.rsplit("_", 1)[0][:14], d) for n, d in zip(names, disp))


def main():
    meta = json.load(open(os.path.join(CACHE, "meta.json")))
    jobs = [(t, r["file"], r["demo"])
            for t in all_tasks() for r in meta["episodes"][t]]
    with ProcessPoolExecutor(max_workers=20) as ex:
        futs = [ex.submit(one, *j) for j in jobs]
        for fu in as_completed(futs):
            print(fu.result(), flush=True)


if __name__ == "__main__":
    main()
