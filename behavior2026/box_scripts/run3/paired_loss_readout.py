"""PAIRED-LOSS READOUT (2026-09-28) — does the policy read the image? Runs against a checkpoint + an assembled mix that contains
the factory twins and the ODART clips (the full mix on the trainer, or the 4 GB readout mix built by build_readout_mix.sh on the sim box). Every object-perturbed (ODART) episode has a twin: the unperturbed factory clip of the same demo, restored to the same
demo frame and driven/staged identically, so the two clips share their recorded prefix (base approach) frame for frame
until the radio is teleported; from that frame on, proprio stays ~identical for a while while the pictures and the
required actions differ. For each pair, on the W frames after the divergence point:
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


def load_model(cfg, params_dir):
    model = cfg.model.create(jax.random.key(0)); graphdef, state = nnx.split(model); pure = state.to_pure_dict()
    params = _m.restore_params(params_dir, restore_type=np.ndarray); missing = []
    def merge(dst, src, path=""):
        for k, v in dst.items():
            if k in src and isinstance(v, dict) and isinstance(src[k], dict): merge(v, src[k], path + "/" + k)
            elif k in src and not isinstance(v, dict): dst[k] = jnp.asarray(src[k], dtype=jnp.asarray(v).dtype)
            else: missing.append(path + "/" + k)
    merge(pure, params); state.replace_by_pure_dict(pure)
    print(f"model loaded; params not in ckpt (kept at init): {len(missing)} {sorted(set(m.split('/')[1] for m in missing))[:6]}", flush=True)
    return nnx.merge(graphdef, state)


def collate(items):
    def rec(vals):
        if isinstance(vals[0], dict): return {k: rec([v[k] for v in vals]) for k in vals[0]}
        return np.stack([np.asarray(v) for v in vals])
    return rec(items)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="pi05_radio_4d_all"); ap.add_argument("--params", required=True)
    ap.add_argument("--mix", default="/root/b1k_radio_mix_all"); ap.add_argument("--map", default="/root/odart_episode_map.json")
    ap.add_argument("--window", type=int, default=32); ap.add_argument("--tol", type=float, default=2e-3); ap.add_argument("--max-pairs", type=int, default=999)
    ap.add_argument("--out", default=None); a = ap.parse_args(); t0 = time.time()
    import dataclasses
    cfg = _c.get_config(a.config); data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    # --mix is BOTH the bookkeeping root and the dataset root (2026-09-29: the readout runs on the sim box against the
    # 137-episode readout mix = factory twins + ODART, not against the config's dataset_root, which lives on the trainer)
    data_cfg = dataclasses.replace(data_cfg, dataset_root=a.mix)
    print(f"dataset root: {data_cfg.dataset_root} (repo_id {data_cfg.repo_id}); assets {cfg.assets_dirs}", flush=True)
    # create_b1k_dataset (the trainer's path): honors dataset_root and the history_frame_offsets (gists/history tokens);
    # the generic create_torch_dataset ignores both (it looks in ~/.cache/huggingface/lerobot/<repo_id> -> the 401 liar)
    ds = _dl.transform_dataset(_dl.create_b1k_dataset(data_cfg, cfg.model.action_horizon), data_cfg)
    model = load_model(cfg, a.params)
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
    tmap = json.load(open(a.map)); fac_off = off["b1k_radio_factory"]; fac_ep_of_demo = {int(d): int(e) for e, d in tmap["factory"].items()}
    # ---- pairs -------------------------------------------------------------------------------------------------------
    rng = jax.random.key(7); res = []; n_done = 0
    def loss_of(obs_dict, act):
        obs = _m.Observation.from_dict(obs_dict); return np.asarray(model.compute_loss(rng, obs, jnp.asarray(act), train=False)).mean()
    for root, eps in tmap["odart"].items():
        if root not in off: print(f"  {root} not in mix sources; skipped", flush=True); continue
        for r in eps:
            if n_done >= a.max_pairs: break
            mo = off[root] + r["ep"]; mf = fac_off + fac_ep_of_demo[r["demo"]]
            so, sf = st_tab[mo], st_tab[mf]; n = min(len(so), len(sf))
            d = np.abs(so[:n] - sf[:n]).max(1); same = np.where(d < a.tol)[0]
            k0 = int(same.max()) if len(same) else -1
            if k0 < 5 or k0 + 2 >= n:
                print(f"  d{r['demo']} {r['tag']}: no shared prefix (k0={k0}, n={n}) -> pairing failed, skipped", flush=True); continue
            w = list(range(k0 + 1, min(k0 + 1 + a.window, n)))
            io = [ds[ep_from[mo] + k] for k in w]; if_ = [ds[ep_from[mf] + k] for k in w]
            bo, bf = collate(io), collate(if_)
            ao, af = bo.pop("actions"), bf.pop("actions")
            swap = dict(bo); swap["image"] = bf["image"]; swap["image_mask"] = bf["image_mask"]
            l_true, l_swap, l_fact, l_cross = loss_of(bo, ao), loss_of(swap, ao), loss_of(bf, af), loss_of(bf, ao)
            act_gap = float(np.abs(np.asarray(ao) - np.asarray(af)).mean()); prop_gap = float(d[w].mean())
            res.append(dict(root=root, ep=r["ep"], demo=r["demo"], tag=r["tag"], k0=k0, n_frames=len(w), L_true=float(l_true), L_swap=float(l_swap), L_fact=float(l_fact), L_cross=float(l_cross), action_gap=act_gap, proprio_gap=prop_gap))
            n_done += 1
            print(f"  d{r['demo']:>3} {r['tag']:<6} k0={k0:>4} n={len(w):>2} L_true={l_true:.4f} L_swap={l_swap:.4f} L_fact={l_fact:.4f} L_cross={l_cross:.4f} |da|={act_gap:.3f} |dq|={prop_gap:.4f}", flush=True)
    if not res: print("PAIRED_READOUT_FAILED: no pairs"); sys.exit(1)
    R = {k: float(np.mean([x[k] for x in res])) for k in ("L_true", "L_swap", "L_fact", "L_cross", "action_gap", "proprio_gap")}
    gap = R["L_swap"] - R["L_true"]; rel = gap / max(R["L_true"], 1e-9)
    print(f"\nPAIRED_RESULT pairs={len(res)} L_true={R['L_true']:.4f} L_swap={R['L_swap']:.4f} L_fact={R['L_fact']:.4f} L_cross={R['L_cross']:.4f} "
          f"swap_gap={gap:+.4f} ({rel:+.1%}) frac_pairs_swap_worse={np.mean([x['L_swap'] > x['L_true'] for x in res]):.2f} |da|={R['action_gap']:.3f} |dq|={R['proprio_gap']:.4f}", flush=True)
    for tag in sorted(set(x["tag"] for x in res)):
        xs = [x for x in res if x["tag"] == tag]; print(f"  {tag:<6} n={len(xs):>2} L_true={np.mean([x['L_true'] for x in xs]):.4f} L_swap={np.mean([x['L_swap'] for x in xs]):.4f} gap={np.mean([x['L_swap']-x['L_true'] for x in xs]):+.4f}", flush=True)
    print("VERDICT:", "VISION USED (swap gap > 20% of L_true and > 0 on most pairs)" if rel > 0.2 and np.mean([x['L_swap'] > x['L_true'] for x in res]) > 0.7 else "PROPRIO-ONLY (swap changes nothing)" if abs(rel) < 0.05 else "WEAK / MIXED", flush=True)
    if a.out: json.dump(dict(summary=R, swap_gap=gap, rel=rel, pairs=res, params=a.params, config=a.config), open(a.out, "w"), indent=1)
    print(f"PAIRED_READOUT_DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
