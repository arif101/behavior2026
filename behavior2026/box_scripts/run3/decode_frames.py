"""Decode-only worker for the gist precompute: one LeRobot root, one head-camera video file -> the resized 224x224 uint8
frames written into a shared memmap at their dataset row indices (precompute_gists.py --from-cache does the tower pass).
Run one process per video file in parallel (CPU-bound; JAX on CPU for the exact resize_with_pad the model uses).
  JAX_PLATFORMS=cpu python decode_frames.py --root ROOT --cache /root/frame_cache/<name>.npy --file-index 0
  (create the memmap first: python decode_frames.py --root ROOT --cache ... --init)
"""
import argparse, glob, json, pathlib, time
import numpy as np, pyarrow.parquet as pq

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--file-index", type=int, default=-1); ap.add_argument("--init", action="store_true"); ap.add_argument("--batch", type=int, default=64)
a = ap.parse_args(); root = pathlib.Path(a.root); t0 = time.time()
info = json.loads((root / "meta/info.json").read_text()); fps = float(info["fps"]); vkey = "observation.rgb.zed_link_camera_0"
eps = pq.read_table(sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True))[0]).to_pandas().sort_values("episode_index")
n_rows = int(eps["length"].sum())
if a.init:
    mm = np.lib.format.open_memmap(a.cache, mode="w+", dtype=np.uint8, shape=(n_rows, 224, 224, 3)); mm.flush(); print(f"INIT {a.cache} {mm.shape}"); raise SystemExit
import jax
from openpi.shared import image_tools
_resize = jax.jit(lambda x: image_tools.resize_with_pad(x, 224, 224))
mm = np.load(a.cache, mmap_mode="r+"); assert mm.shape[0] == n_rows
import av
grp = eps[eps[f"videos/{vkey}/file_index"] == a.file_index]; ci = int(grp[f"videos/{vkey}/chunk_index"].iloc[0])
vpath = root / "videos" / vkey / f"chunk-{ci:03d}" / f"file-{a.file_index:03d}.mp4"
need = {}
for _, r in grp.iterrows():
    s = int(round(r[f"videos/{vkey}/from_timestamp"] * fps)); r0 = int(r["dataset_from_index"])
    for i in range(int(r["length"])): need[s + i] = r0 + i
buf = []; bidx = []; k = 0; last = max(need); n_done = 0
def flush():
    global n_done
    if not buf: return
    mm[bidx] = np.asarray(_resize(np.stack(buf))); n_done += len(buf); buf.clear(); bidx.clear()
with av.open(str(vpath)) as cont:
    cont.streams.video[0].thread_type = "AUTO"
    for frame in cont.decode(video=0):
        if k in need:
            buf.append(frame.to_ndarray(format="rgb24")); bidx.append(need[k])
            if len(buf) >= a.batch: flush()
        k += 1
        if k > last: break
flush(); mm.flush()
print(f"DECODE_OK {root.name} file-{a.file_index:03d}: {n_done}/{len(need)} frames in {time.time()-t0:.0f}s ({n_done/max(time.time()-t0,1):.0f} fps)")
