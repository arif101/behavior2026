"""PAIRED-LOSS READOUT (2026-09-28) — does the policy read the image? Runs against a checkpoint + an assembled mix that contains
the factory twins and the ODART clips (the full mix on the trainer, or the 4 GB readout mix built by build_readout_mix.sh on the sim box). Every object-perturbed (ODART) episode has a twin: the OPPOSITE perturbation of the same demo (ol5/olm5, od5/odm5,
oy15/oym15, omix1/omix2), restored to the same demo frame and driven by the same script, so the two clips have identical
proprio for the first k0 = 27..173 frames (settle + orient) while the radio sits 10 cm / 30 deg apart in the picture; after
k0 the goals diverge and the required actions differ. (The factory root is NOT a twin: RaC-v2 collector, other restore.) For each pair, on the W frames after the divergence point:
  L_true  = loss(ODART obs, ODART actions)
  L_swap  = loss(ODART obs with the TWIN's images (all cams) swapped in, ODART actions)   <- same proprio, wrong picture
  L_fact  = loss(twin obs, twin actions)
  L_cross = loss(twin obs, ODART actions)                                                <- what a proprio-only policy scores
A policy that acts from proprioception gives L_swap ~= L_true (~= L_cross); a policy that reads the picture gives
L_swap >> L_true. Reports means and per-perturbation-tag means. train=False, one fixed rng per pair shared by all four.
  cd /root/openpi_fork && XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python /root/run3/paired_loss_readout.py \
      --config pi05_radio_4d_all --params /root/openpi_fork/outputs/checkpoints/pi05_radio_4d_all/radio_4d_all/2500/params \
      --mix /root/b1k_radio_mix_all --map /root/odart_episode_map.json [--window 32] [--tol 2e-3] [--out /root/run3_logs/paired_2500.json]
"""
import os, sys, json, glob, argparse, time; os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, jax, jax.numpy as jnp, pyarrow.parquet as pq
from flax import nnx
from openpi.training import config as _c, data_loader as _dl
from openpi.models import model as _m


def load_model(cfg, params_dir, bf16=True):
    # init on the CPU, then ONE device copy per leaf: init-on-GPU + a second GPU copy of the checkpoint OOMed the 32 GB sim-box
    # card (RESOURCE_EXHAUSTED at 20.8 GB used, 2026-09-29); host RAM holds the init + the numpy checkpoint instead.
    gpu = jax.devices()[0]
    with jax.default_device(jax.devices("cpu")[0]):
        model = cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model); pure = state.to_pure_dict()
    params = _m.restore_params(params_dir, restore_type=np.ndarray); missing = []
    def dt(v):   # bf16 params (the serve-time dtype; 6.6 GB instead of 13 GB) unless --fp32: fp32 + batch-32 activations OOM a 32 GB card
        d = np.asarray(v).dtype; return jnp.bfloat16 if (bf16 and np.issubdtype(d, np.floating)) else d
    def merge(dst, src, path=""):
        for k, v in dst.items():
            if k in src and isinstance(v, dict) and isinstance(src[k], dict): merge(v, src[k], path + "/" + k)
            elif k in src and not isinstance(v, dict): dst[k] = jax.device_put(np.asarray(src[k]).astype(dt(v)), gpu)
            else: missing.append(path + "/" + k); dst[k] = jax.device_put(np.asarray(v).astype(dt(v)), gpu) if hasattr(v, "shape") else v
    merge(pure, params); del params; state.replace_by_pure_dict(pure)
    print(f"model loaded; params not in ckpt (kept at init): {len(missing)} {sorted(set(m.split('/')[1] for m in missing))[:6]}", flush=True)
    return nnx.merge(graphdef, state)


# INPUT channels only (labels such as gt_depth / aux_pixels / rail_now / hist_flow / stage / progress stay with the scored clip;
# the loss is flow-only anyway, see model.flow_only)
SWAP_IMG = {"image", "image_mask", "history_gists", "history_mask", "history_tokens", "history_xyz", "history_valid", "patch_xyz", "patch_valid"}
SWAP_ALL = SWAP_IMG | {"target_points", "target_points_mask", "map_tokens", "anchors"}


def collate(items):
    def rec(vals):
        if isinstance(vals[0], dict): return {k: rec([v[k] for v in vals]) for k in vals[0]}
        return np.stack([np.asarray(v) for v in vals])
    return rec(items)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="pi05_radio_4d_all"); ap.add_argument("--params", required=True)
    ap.add_argument("--mix", default="/root/b1k_radio_mix_all"); ap.add_argument("--map", default="/root/odart_episode_map.json")
    ap.add_argument("--window", type=int, default=32); ap.add_argument("--tol", type=float, default=2e-3); ap.add_argument("--max-pairs", type=int, default=999)
    ap.add_argument("--out", default=None); ap.add_argument("--chunk", type=int, default=8, help="frames per forward (activation memory)")
    ap.add_argument("--fp32", action="store_true", help="keep fp32 params (needs an 80 GB card)"); a = ap.parse_args(); t0 = time.time()
    import dataclasses
    cfg = _c.get_config(a.config); data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    # --mix is BOTH the bookkeeping root and the dataset root (2026-09-29: the readout runs on the sim box against the
    # 137-episode readout mix = factory twins + ODART, not against the config's dataset_root, which lives on the trainer)
    data_cfg = dataclasses.replace(data_cfg, dataset_root=a.mix)
    # DETERMINISTIC fetch: drop the train-time ProprioNoise transform (np.random per __getitem__: 10-15% loss jitter between two
    # fetches of the same frames, 2026-09-29) and pin the map tokens to the full map (B1KInputs.map_blind_prob draws per fetch)
    from openpi import transforms as _tf
    def _det(g):
        ins = []
        for t in g.inputs:
            if isinstance(t, _tf.ProprioNoise): continue
            if hasattr(t, "map_blind_prob") and dataclasses.is_dataclass(t): t = dataclasses.replace(t, map_blind_prob=0.0)
            ins.append(t)
        return dataclasses.replace(g, inputs=tuple(ins))
    data_cfg = dataclasses.replace(data_cfg, data_transforms=_det(data_cfg.data_transforms), model_transforms=_det(data_cfg.model_transforms))
    print(f"dataset root: {data_cfg.dataset_root} (repo_id {data_cfg.repo_id}); assets {cfg.assets_dirs}; transforms {[type(t).__name__ for t in data_cfg.data_transforms.inputs]}", flush=True)
    # create_b1k_dataset (the trainer's path): honors dataset_root and the history_frame_offsets (gists/history tokens);
    # the generic create_torch_dataset ignores both (it looks in ~/.cache/huggingface/lerobot/<repo_id> -> the 401 liar)
    ds = _dl.transform_dataset(_dl.create_b1k_dataset(data_cfg, cfg.model.action_horizon), data_cfg)
    model = load_model(cfg, a.params, bf16=not a.fp32); model.flow_only = True   # flow loss only (aux terms off), see pi0.py
    # ---- mix bookkeeping: source -> mix episode offset; episode -> global row range; proprio per episode -----------------
    # assemble_run3_mix.py writes {"sources": [{"source": "<root path>", "episodes": [first, last], ...}, ...]}
    srcs = json.load(open(f"{a.mix}/meta/run3_sources.json"))["sources"]
    off = {os.path.basename(str(s["source"]).rstrip("/")): int(s["episodes"][0]) for s in srcs}
    em = sorted(glob.glob(f"{a.mix}/meta/episodes/**/*.parquet", recursive=True)); ep = pq.read_table(em[0]).to_pandas().sort_values("episode_index")
    ep_from = dict(zip(ep["episode_index"].tolist(), ep["dataset_from_index"].tolist())); ep_len = dict(zip(ep["episode_index"].tolist(), ep["length"].tolist()))
    files = sorted(glob.glob(f"{a.mix}/data/**/*.parquet", recursive=True))
    st_tab = {}
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"])
        e = t.column("episode_index").to_numpy(); fr = t.column("frame_index").to_numpy(); st = np.stack(t.column("observation.state").to_pylist()).astype(np.float32)
        for eid in np.unique(e):
            m = e == eid; st_tab[int(eid)] = st[m][np.argsort(fr[m])]
    tmap = json.load(open(a.map))
    # PAIRS (2026-09-29): the b1k_radio_factory clips are NOT twins (RaC-v2 collector, other restore state: |dstate| 0.3-0.6 from
    # frame 0, zero shared prefix on all 99). The ODART clips of the SAME demo are: same restore + settle, proprio identical
    # (max|dstate| = 0.0) for k0 = 27..173 frames, then the goals diverge. So a pair = two opposite perturbations of one demo
    # (ol5/olm5, od5/odm5, oy15/oym15, omix1/omix2), evaluated in both directions: same proprio prefix, radio 10 cm / 30 deg
    # apart in the picture, different required actions. 43 unordered pairs over 15 demos -> 86 directed pairs.
    import collections
    by_demo = collections.defaultdict(dict)
    for root, eps in tmap["odart"].items():
        if root not in off: print(f"  {root} not in mix sources; skipped", flush=True); continue
        for r in eps: by_demo[int(r["demo"])][r["tag"]] = off[root] + int(r["ep"])
    OPP = [("ol5", "olm5"), ("od5", "odm5"), ("oy15", "oym15"), ("omix1", "omix2")]
    directed = []
    for demo in sorted(by_demo):
        for ta, tb in OPP:
            if ta in by_demo[demo] and tb in by_demo[demo]:
                directed.append((demo, ta, tb, by_demo[demo][ta], by_demo[demo][tb])); directed.append((demo, tb, ta, by_demo[demo][tb], by_demo[demo][ta]))
    print(f"pairs: {len(directed)} directed over {len(by_demo)} demos", flush=True)
    # ---- pairs -------------------------------------------------------------------------------------------------------
    rng = jax.random.key(7); res = []; n_done = 0; cache = {}
    @nnx.jit
    def _loss(m, r, obs, act): return m.compute_loss(r, obs, act, train=False)   # jitted once per window length (eager = ~15 s/call)
    def _sl(v, i, j):   # slice the batch axis through nested dicts (image / image_mask are per-camera dicts)
        if isinstance(v, dict): return {k: _sl(x, i, j) for k, x in v.items()}
        return v[i:j] if hasattr(v, "shape") and getattr(v, "ndim", 0) > 0 else v
    def loss_of(obs_dict, act):   # chunked over the window (a.chunk frames per forward); per-frame mean
        B = len(act); tot = 0.0
        for i in range(0, B, a.chunk):
            sub = _sl(obs_dict, i, i + a.chunk)
            obs = _m.Observation.from_dict(sub); l = np.asarray(_loss(model, rng, obs, jnp.asarray(act[i:i + a.chunk]))); tot += float(l.mean()) * len(l)
        return tot / B
    for demo, ta, tb, mo, mf in directed:
            r = dict(demo=demo, tag=f"{ta}/{tb}", ep=mo); root = f"d{demo}"
            if n_done >= a.max_pairs: break
            so, sf = st_tab[mo], st_tab[mf]; n = min(len(so), len(sf))
            d = np.abs(so[:n] - sf[:n]).max(1); same = np.where(d < a.tol)[0]
            k0 = int(same.max()) if len(same) else -1
            if k0 < 5 or k0 + 2 >= n:
                print(f"  d{r['demo']} {r['tag']}: no shared prefix (k0={k0}, n={n}) -> pairing failed, skipped", flush=True); continue
            w = list(range(k0 + 1, min(k0 + 1 + a.window, n)))
            def fetch(ep):   # both directions of a pair reuse the same fetched items (deterministic loader; halves the decode time)
                key = (ep, w[0], w[-1])
                if key not in cache:
                    if len(cache) >= 4: cache.pop(next(iter(cache)))
                    b = collate([ds[ep_from[ep] + k] for k in w]); cache[key] = (b, b.pop("actions"))
                return cache[key]
            bo, ao = fetch(mo); bf, af = fetch(mf); bo = dict(bo); bf = dict(bf)
            if n_done == 0:
                _l1, _l2 = loss_of(bo, ao), loss_of(bo, ao); print(f"determinism check: {_l1:.6f} vs {_l2:.6f} (same batch twice)", flush=True)
            if n_done == 0: print(f"batch keys: {sorted(bo)}", flush=True)
            # swap sets: IMG = every image-derived channel (current pictures, history tokens/gists/cells, depth PE, depth/pixel
            # labels) -> isolates "does it read the pictures" with the pointer/map/proprio kept; ALL = IMG + pointer + map + anchors
            swap = {k: (bf[k] if k in SWAP_IMG else v) for k, v in bo.items()}; swap_all = {k: (bf[k] if k in SWAP_ALL else v) for k, v in bo.items()}
            l_true, l_swap, l_swap_all, l_fact, l_cross = loss_of(bo, ao), loss_of(swap, ao), loss_of(swap_all, ao), loss_of(bf, af), loss_of(bf, ao)
            act_gap = float(np.abs(np.asarray(ao) - np.asarray(af)).mean()); prop_gap = float(d[w].mean())
            res.append(dict(root=root, ep=r["ep"], demo=r["demo"], tag=r["tag"], twin_ep=mf, k0=k0, n_frames=len(w), L_true=float(l_true), L_swap=float(l_swap), L_swap_all=float(l_swap_all), L_fact=float(l_fact), L_cross=float(l_cross), action_gap=act_gap, proprio_gap=prop_gap))
            n_done += 1
            print(f"  d{r['demo']:>3} {r['tag']:<11} k0={k0:>4} n={len(w):>2} L_true={l_true:.4f} L_swap={l_swap:.4f} L_swap_all={l_swap_all:.4f} L_fact={l_fact:.4f} L_cross={l_cross:.4f} |da|={act_gap:.3f} |dq|={prop_gap:.4f}", flush=True)
    if not res: print("PAIRED_READOUT_FAILED: no pairs"); sys.exit(1)
    R = {k: float(np.mean([x[k] for x in res])) for k in ("L_true", "L_swap", "L_swap_all", "L_fact", "L_cross", "action_gap", "proprio_gap")}
    gap = R["L_swap"] - R["L_true"]; rel = gap / max(R["L_true"], 1e-9); gap_all = R["L_swap_all"] - R["L_true"]
    print(f"\nPAIRED_RESULT pairs={len(res)} L_true={R['L_true']:.4f} L_swap={R['L_swap']:.4f} L_swap_all={R['L_swap_all']:.4f} L_fact={R['L_fact']:.4f} L_cross={R['L_cross']:.4f} "
          f"swap_gap={gap:+.4f} ({rel:+.1%}) swap_all_gap={gap_all:+.4f} ({gap_all / max(R['L_true'], 1e-9):+.1%}) frac_pairs_swap_worse={np.mean([x['L_swap'] > x['L_true'] for x in res]):.2f} "
          f"frac_pairs_swap_all_worse={np.mean([x['L_swap_all'] > x['L_true'] for x in res]):.2f} |da|={R['action_gap']:.3f} |dq|={R['proprio_gap']:.4f}", flush=True)
    for tag in sorted(set(x["tag"] for x in res)):
        xs = [x for x in res if x["tag"] == tag]; print(f"  {tag:<6} n={len(xs):>2} L_true={np.mean([x['L_true'] for x in xs]):.4f} L_swap={np.mean([x['L_swap'] for x in xs]):.4f} gap={np.mean([x['L_swap']-x['L_true'] for x in xs]):+.4f}", flush=True)
    print("VERDICT_ALL (pointer+map+images swapped):", "PERCEPTION USED" if gap_all / max(R["L_true"], 1e-9) > 0.2 else "PROPRIO-ONLY/WEAK")
    print("VERDICT:", "VISION USED (swap gap > 20% of L_true and > 0 on most pairs)" if rel > 0.2 and np.mean([x['L_swap'] > x['L_true'] for x in res]) > 0.7 else "PROPRIO-ONLY (swap changes nothing)" if abs(rel) < 0.05 else "WEAK / MIXED", flush=True)
    if a.out: json.dump(dict(summary=R, swap_gap=gap, rel=rel, pairs=res, params=a.params, config=a.config), open(a.out, "w"), indent=1)
    print(f"PAIRED_READOUT_DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
