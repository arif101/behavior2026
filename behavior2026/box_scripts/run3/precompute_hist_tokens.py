"""4D HISTORY TOKENS, data side (ARCH_4D_ATTENTION_SPEC A2). Per frame, from the FROZEN SigLIP tower of the warm-start
checkpoint: the 256 projected head-camera tokens average-pooled to a 4x4 grid -> `hist_tok` (16 x 2048, float16, flat
32768); `hist_cellxyz` (16 x 3, flat 48): the mean base-frame 3D point of each cell's valid patches, from gt_depth_ds
(head camera, 16x16 patch-mean metres) and the FK camera pose (`cam_pose` column, 3 x 7, fk_cam_poses.py);
`hist_cellvalid` (16 bool); `odom_xyyaw` (3): dead-reckoned base pose in the episode frame from proprio[0:3] =
body-frame [vx, vy, wz] at fps (the serving wrapper integrates the same way). Adds the columns to every data parquet
of a LeRobot root and registers them. Reuses the frame cache written by precompute_gists.py (--from-cache).
  XLA_PYTHON_CLIENT_PREALLOCATE=false python precompute_hist_tokens.py --root ROOT --params /root/ckpt_a4/params --from-cache /root/frame_cache/mix.npy
"""
import argparse, os, glob, json, pathlib, time
import numpy as np, pyarrow as pa, pyarrow.parquet as pq

_PATCH_RAY = 1080.0 / 874.5   # R1 cameras: fx = fy = 874.5 @ 1080 px, principal point at the centre (camera_intrinsics.json)


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
    def pooled(model, imgs_u8):
        x = image_tools.resize_with_pad(imgs_u8, 224, 224).astype(jnp.float32) / 255.0 * 2.0 - 1.0
        toks, _ = model(x, train=False)                                     # [b, 256, 2048] (16x16 patch grid, row-major)
        t = toks.astype(jnp.float32).reshape(-1, 4, 4, 4, 4, toks.shape[-1])  # [b, gi, pi, gj, pj, d]
        return t.mean(axis=(2, 4)).reshape(-1, 16, toks.shape[-1])            # [b, 16 cells, d]
    return tower, pooled


def quat_to_rot(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cell_points(depth16, cam_pose7):
    """head-camera 16x16 patch depth + camera pose (base frame) -> [16, 3] cell means (base frame), [16] validity."""
    jj, ii = np.meshgrid(np.arange(16), np.arange(16))
    ax = ((jj + 0.5) / 16.0 - 0.5) * _PATCH_RAY; ay = ((ii + 0.5) / 16.0 - 0.5) * _PATCH_RAY
    z = depth16; ok = np.isfinite(z) & (z > 0.05)
    p_cam = np.stack([ax * z, -ay * z, -z], axis=-1)                       # USD: -Z forward, +Y up
    R = quat_to_rot(cam_pose7[3:7]); t = cam_pose7[:3]
    p_base = p_cam @ R.T + t[None, None]
    out = np.zeros((16, 3), np.float32); val = np.zeros(16, bool)
    for gi in range(4):
        for gj in range(4):
            m = ok[gi * 4:(gi + 1) * 4, gj * 4:(gj + 1) * 4]; pts = p_base[gi * 4:(gi + 1) * 4, gj * 4:(gj + 1) * 4][m]
            if len(pts): out[gi * 4 + gj] = pts.mean(0); val[gi * 4 + gj] = True
    return out, val


def odometry(state_ep, fps):
    """[T, 61] proprio -> [T, 3] (x, y, yaw) by integrating the body-frame twist proprio[0:3] at 1/fps (wrapper convention)."""
    T = len(state_ep); out = np.zeros((T, 3), np.float32); x = y = yaw = 0.0; dt = 1.0 / fps
    for t in range(T):
        vx, vy, wz = [float(v) for v in state_ep[t, :3]]
        yaw += wz * dt   # wrapper order: yaw first, then position with the updated yaw
        x += (vx * np.cos(yaw) - vy * np.sin(yaw)) * dt; y += (vx * np.sin(yaw) + vy * np.cos(yaw)) * dt
        out[t] = (x, y, yaw)
    return out


def col_np(t, name, dtype=None):
    """arrow column -> numpy without Python lists (list / fixed-size-list columns come back flat: reshape at the call site)."""
    a = t.column(name).combine_chunks()
    if pa.types.is_fixed_size_list(a.type) or pa.types.is_list(a.type): a = a.flatten()
    x = a.to_numpy(zero_copy_only=False)
    return x.astype(dtype, copy=False) if dtype is not None else x


def fsl(np2d, pa_type):
    """[n, k] numpy -> fixed_size_list<pa_type>[k] arrow array, zero-copy (no .tolist())."""
    np2d = np.ascontiguousarray(np2d); n, k = np2d.shape
    return pa.FixedSizeListArray.from_arrays(pa.array(np2d.reshape(-1), type=pa_type), k)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--params", required=True)
    ap.add_argument("--toks-cache", help="float16 memmap [n_rows, 16, 2048] for the pooled tokens; reused (tower pass skipped) when PATH.done exists")
    ap.add_argument("--config", default="pi05_radio_run3_a4"); ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--from-cache", required=True, help="224x224 uint8 frame memmap written by precompute_gists.py --cache-frames")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); root = pathlib.Path(a.root); t0 = time.time()
    info = json.loads((root / "meta/info.json").read_text()); fps = float(info["fps"])
    frames = np.load(a.from_cache, mmap_mode="r"); n_rows = frames.shape[0]
    tc = pathlib.Path(a.toks_cache) if a.toks_cache else None; done = tc.with_suffix(".done") if tc else None
    if tc is not None and tc.exists() and done.exists():
        toks = np.load(tc, mmap_mode="r"); assert toks.shape == (n_rows, 16, 2048), toks.shape
        print(f"tower pass SKIPPED: reusing {tc} {toks.shape}", flush=True)
    else:
        tower, pooled = build_tower(a.params, a.config)
        toks = np.lib.format.open_memmap(tc, mode="w+", dtype=np.float16, shape=(n_rows, 16, 2048)) if tc else np.zeros((n_rows, 16, 2048), np.float16)
        for s0 in range(0, n_rows, a.batch):
            toks[s0:s0 + a.batch] = np.asarray(pooled(tower, np.asarray(frames[s0:s0 + a.batch]))).astype(np.float16)
            if (s0 // a.batch) % 500 == 0: print(f"  tower {s0}/{n_rows} {time.time()-t0:.0f}s", flush=True)
        if tc: toks.flush(); done.touch()
        print(f"tower pass: {n_rows} rows, {time.time()-t0:.0f}s", flush=True)
    files = sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True)); wrote = 0
    for f in files:
        t = pq.read_table(f); n = len(t); idx = col_np(t, "index", np.int64)
        st = col_np(t, "observation.state", np.float32).reshape(n, -1)
        dep = col_np(t, "gt_depth_ds", np.float32).reshape(n, 3, 16, 16)
        cp = col_np(t, "cam_pose", np.float32).reshape(n, 3, 7)
        ep = col_np(t, "episode_index", np.int64); fr = col_np(t, "frame_index", np.int64)
        cxyz = np.zeros((n, 16, 3), np.float32); cval = np.zeros((n, 16), bool); odom = np.zeros((n, 3), np.float32)
        for i in range(n):
            cxyz[i], cval[i] = cell_points(dep[i, 0], cp[i, 0])
        for e in np.unique(ep):
            m = np.where(ep == e)[0]; order = m[np.argsort(fr[m])]
            odom[order] = odometry(st[order], fps)
        for name in ("hist_tok", "hist_cellxyz", "hist_cellvalid", "odom_xyyaw"):
            if name in t.schema.names: t = t.drop([name])
        if not a.dry_run:
            t = t.append_column("hist_tok", fsl(np.asarray(toks[idx]).reshape(n, -1).astype(np.float16, copy=False), pa.float16()))
            t = t.append_column("hist_cellxyz", fsl(cxyz.reshape(n, -1), pa.float32()))
            t = t.append_column("hist_cellvalid", fsl(cval, pa.bool_()))
            t = t.append_column("odom_xyyaw", fsl(odom, pa.float32()))
            pq.write_table(t, f + ".tmp"); os.replace(f + ".tmp", f)          # never leave a half-written parquet
        wrote += n; print(f"  {pathlib.Path(f).name}: {n} rows, valid cells {cval.mean():.2f}, {time.time()-t0:.0f}s", flush=True)
    if not a.dry_run:
        info["features"]["hist_tok"] = {"dtype": "float16", "shape": [16 * 2048], "names": None}
        info["features"]["hist_cellxyz"] = {"dtype": "float32", "shape": [48], "names": None}
        info["features"]["hist_cellvalid"] = {"dtype": "bool", "shape": [16], "names": None}
        info["features"]["odom_xyyaw"] = {"dtype": "float32", "shape": [3], "names": None}
        (root / "meta/info.json").write_text(json.dumps(info, indent=4))
    print(f"PRECOMPUTE_HIST_TOKENS_OK {root.name}: {wrote} rows, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
