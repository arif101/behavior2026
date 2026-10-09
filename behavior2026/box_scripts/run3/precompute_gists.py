"""Temporal forcing, data side (TEMPORAL_4D_SWEEP §4.1): per-frame HEAD-camera gist from the FROZEN SigLIP tower of the
warm-start checkpoint (mean of the 256 projected image tokens, 2048-D, images preprocessed exactly as the model does:
resize_with_pad to 224 then [-1, 1]) + `hist_geo` = [EE_L(3), EE_R(3), button_base(3)] for the change-prediction targets.
Adds two columns to every data parquet of a LeRobot root and registers them. Runs the tower alone on the GPU with
on-demand allocation so it can sit beside a training job.
  XLA_PYTHON_CLIENT_PREALLOCATE=false python precompute_gists.py --root ROOT --params /root/ckpt_a4/params
"""
import argparse, glob, json, pathlib, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

def build_tower(params_dir, config_name):
    import jax, jax.numpy as jnp
    from flax import nnx
    from flax.nnx import bridge as nnx_bridge
    import openpi.models.siglip as _siglip, openpi.models.gemma as _gemma, openpi.training.config as _c
    from openpi.models import model as _m
    from openpi.shared import image_tools
    cfg = _c.get_config(config_name).model; pg = _gemma.get_config(cfg.paligemma_variant)
    img = nnx_bridge.ToNNX(_siglip.Module(num_classes=pg.width, variant="So400m/14", pool_type="none", scan=True, dtype_mm=cfg.dtype))
    img.lazy_init(jnp.zeros((1, 224, 224, 3), jnp.float32), train=False, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(img)
    ck = _m.restore_params(params_dir, restore_type=np.ndarray)["PaliGemma"]["img"]
    ck = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16) if x.dtype == np.float32 else jnp.asarray(x), ck)
    state.replace_by_pure_dict(ck); tower = nnx.merge(graphdef, state)
    @nnx.jit
    def gist(model, imgs_u8):
        x = image_tools.resize_with_pad(imgs_u8, 224, 224).astype(jnp.float32) / 255.0 * 2.0 - 1.0
        toks, _ = model(x, train=False)
        return toks.astype(jnp.float32).mean(axis=1)
    return tower, gist

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--params", required=True)
    ap.add_argument("--config", default="pi05_radio_run3_a4"); ap.add_argument("--batch", type=int, default=48); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cache-frames", help="write the decoded+resized 224x224 uint8 frames to this memmap file (decode once, re-tower later)")
    ap.add_argument("--from-cache", help="skip decoding: read frames from this memmap (written by --cache-frames)")
    a = ap.parse_args(); root = pathlib.Path(a.root); t0 = time.time()
    info = json.loads((root / "meta/info.json").read_text()); fps = float(info["fps"]); vkey = "observation.rgb.zed_link_camera_0"
    eps = pq.read_table(sorted(glob.glob(str(root / "meta/episodes/**/*.parquet"), recursive=True))[0]).to_pandas().sort_values("episode_index")
    n_rows = int(eps["length"].sum()); gists = np.zeros((n_rows, 2048), np.float32); filled = np.zeros(n_rows, bool)
    tower, gist_fn = build_tower(a.params, a.config); import av
    from openpi.shared import image_tools
    import jax
    _resize = jax.jit(lambda x: image_tools.resize_with_pad(x, 224, 224))
    if a.from_cache:
        frames = np.load(a.from_cache, mmap_mode="r"); assert frames.shape[0] == n_rows, (frames.shape, n_rows)
        for s0 in range(0, n_rows, a.batch):
            gists[s0:s0 + a.batch] = np.asarray(gist_fn(tower, np.asarray(frames[s0:s0 + a.batch])))
        filled[:] = True
        print(f"tower pass from cache: {n_rows} rows, {time.time()-t0:.0f}s", flush=True)
    cache = None
    if a.cache_frames and not a.from_cache:
        cache = np.lib.format.open_memmap(a.cache_frames, mode="w+", dtype=np.uint8, shape=(n_rows, 224, 224, 3))
    for fi, grp in (eps.groupby(f"videos/{vkey}/file_index") if not a.from_cache else []):
        ci = int(grp[f"videos/{vkey}/chunk_index"].iloc[0]); vpath = root / "videos" / vkey / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        plan = []   # (file frame start, length, dataset_from_index)
        for _, r in grp.sort_values(f"videos/{vkey}/from_timestamp").iterrows():
            plan.append((int(round(r[f"videos/{vkey}/from_timestamp"] * fps)), int(r["length"]), int(r["dataset_from_index"])))
        need = {}
        for s, L, r0 in plan:
            for i in range(L): need[s + i] = r0 + i
        buf = []; bidx = []; k = 0; last = max(need) if need else -1
        def flush():
            if not buf: return
            arr = np.stack(buf)
            if cache is not None: cache[bidx] = np.asarray(_resize(arr))
            g = np.asarray(gist_fn(tower, arr)); gists[bidx] = g; filled[bidx] = True; buf.clear(); bidx.clear()
        with av.open(str(vpath)) as cont:
            for frame in cont.decode(video=0):
                if k in need:
                    buf.append(frame.to_ndarray(format="rgb24")); bidx.append(need[k])
                    if len(buf) >= a.batch: flush()
                k += 1
                if k > last: break
        flush(); print(f"  file {vpath.name}: {len(need)} frames, {k} decoded, {time.time()-t0:.0f}s", flush=True)
    if cache is not None: cache.flush(); print(f"frame cache written: {a.cache_frames} {cache.shape}", flush=True)
    assert filled.all(), f"unfilled rows: {(~filled).sum()}"
    print(f"gists done: {n_rows} rows, mean|g| {np.linalg.norm(gists, axis=1).mean():.3f}, {time.time()-t0:.0f}s", flush=True)
    files = sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)); wrote = 0
    for f in files:
        t = pq.read_table(f); idx = np.asarray(t.column("index").to_pylist())
        st = np.asarray(t.column("observation.state").to_pylist(), np.float32); tp = np.asarray(t.column("target_points").to_pylist(), np.float32).reshape(len(t), 6)
        geo = np.concatenate([st[:, 17:20], st[:, 42:45], tp[:, 3:6] + st[:, 42:45]], axis=1).astype(np.float32)
        for name in ("gist_head", "hist_geo"):
            if name in t.schema.names: t = t.drop([name])
        if not a.dry_run:
            t = t.append_column("gist_head", pa.array(gists[idx].tolist(), type=pa.list_(pa.float32(), 2048)))
            t = t.append_column("hist_geo", pa.array(geo.tolist(), type=pa.list_(pa.float32(), 9)))
            pq.write_table(t, f)
        wrote += len(t)
    if not a.dry_run:
        info["features"]["gist_head"] = {"dtype": "float32", "shape": [2048], "names": None}
        info["features"]["hist_geo"] = {"dtype": "float32", "shape": [9], "names": None}
        (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"PRECOMPUTE_GISTS_OK {root.name}: {wrote} rows written, {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
