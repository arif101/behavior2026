"""Task-62 stage prediction head (STAGE_HEAD_SPEC_v2 §4 reference architecture, single task, no literals).
Input per frame (6 Hz): 3 cameras x DINOv2 [CLS ⊕ mean-patch] (3x1536, frozen, cached) ⊕ proprio 61-D
  -> fuse MLP -> d=256 token;  window of W tokens (5 s) -> causal transformer (4L) -> per-token heads:
  family 8-way (soft boundary labels)  ·  global progress q  ·  within-skill progress  ·  held-object L/R 4-way
Trains on cached features (task62/stage_head/extract_feats.py). Held-out = every 10th demo (20 episodes).
Metrics (spec §6, single-task subset): per-frame family accuracy + per-family recall, boundary timing error,
q MAE + monotonicity violations, held-object accuracy, entropy at boundaries vs mid-segment, plus two
"does it beat a clock" baselines: family-from-normalized-time and proprio-only.
Usage (openpi env): python task62/stage_head/train_head.py --out /root/step0/stage_head_v1 [--no_vision] [--epochs 25]
"""
import os, json, glob, argparse, time, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

FEATS = "/root/step0/stage_feats"; FAMILIES = ["navigate", "open_door", "pick_up_from", "place_on", "chop", "place_in", "close_door", "other"]
W, STRIDE, SOFT_TOK = 30, 5, 6          # 30 tokens = 5 s at 6 Hz; soft labels within ±6 tokens (±30 raw frames) of a boundary

def load_all(no_vision):
    eps = sorted(int(os.path.basename(p)[2:-4]) for p in glob.glob(f"{FEATS}/ep*.npz"))
    val_ids = set(eps[::10]); data = {}
    for ep in eps:
        d = np.load(f"{FEATS}/ep{ep}.npz")
        x = d["proprio"].astype(np.float32) if no_vision else np.concatenate([d["feats"].reshape(len(d["frame"]), -1).astype(np.float32), d["proprio"].astype(np.float32)], 1)
        n = len(d["frame"]); fam = d["fam"]; soft = np.zeros((n, 8), np.float32); soft[np.arange(n), fam] = 1.0
        b = np.nonzero(np.diff(fam))[0] + 1                                   # token index where a new segment starts
        for t in b:                                                            # cosine blend across ±SOFT_TOK tokens
            for k in range(-SOFT_TOK, SOFT_TOK):
                i = t + k
                if 0 <= i < n:
                    w_new = 0.5 * (1 + math.cos(math.pi * (k + 0.5) / SOFT_TOK * 0.5 - math.pi / 2)) if False else 0.5 * (1 + math.sin(math.pi * (k + 0.5) / (2 * SOFT_TOK)))
                    soft[i] = 0; soft[i, fam[t]] += w_new; soft[i, fam[t - 1]] += 1 - w_new
        seg_len = np.maximum(1, d["seg_end"] - d["seg_start"]); wp = np.clip((d["frame"] - d["seg_start"]) / seg_len, 0, 1).astype(np.float32)
        data[ep] = dict(x=x, fam=fam, soft=soft, q=d["q"].astype(np.float32), wp=wp, agL=np.clip(d["agL"], 0, 3), agR=np.clip(d["agR"], 0, 3),
                        frame=d["frame"], tnorm=(d["frame"] / d["frame"][-1]).astype(np.float32), val=ep in val_ids)
    return data

class Head(nn.Module):
    def __init__(self, d_in, d=256, layers=4, nhead=4):
        super().__init__()
        self.fuse = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, d))
        self.pos = nn.Parameter(torch.zeros(1, W, d)); nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.core = nn.TransformerEncoder(enc, layers); self.norm = nn.LayerNorm(d)
        self.fam = nn.Linear(d, 8); self.q = nn.Linear(d, 1); self.wp = nn.Linear(d, 1); self.agL = nn.Linear(d, 4); self.agR = nn.Linear(d, 4)
        self.register_buffer("mask", torch.triu(torch.full((W, W), float("-inf")), 1))
    def forward(self, x):                                   # x [B,W,d_in] -> per-token outputs
        h = self.fuse(x) + self.pos[:, : x.shape[1]]; h = self.norm(self.core(h, mask=self.mask[: x.shape[1], : x.shape[1]]))
        return dict(fam=self.fam(h), q=self.q(h).squeeze(-1), wp=self.wp(h).squeeze(-1), agL=self.agL(h), agR=self.agR(h))

def windows(ep, i_end):
    """tokens [i_end-W+1 .. i_end], left-padded by repeating the first frame (window-only perception, no episode memory)."""
    idx = np.arange(i_end - W + 1, i_end + 1); return np.maximum(idx, 0)

def batch_from(data, eps, keys_idx, mu, sd, dev):
    xs, ys = [], {k: [] for k in ("soft", "q", "wp", "agL", "agR")}
    for ep, i_end in keys_idx:
        d = data[ep]; idx = windows(ep, i_end); xs.append((d["x"][idx] - mu) / sd)
        for k in ys: ys[k].append(d[k][idx])
    x = torch.from_numpy(np.stack(xs)).to(dev)
    return x, {k: torch.from_numpy(np.stack(v)).to(dev) for k, v in ys.items()}

def losses(out, y):
    l_fam = -(y["soft"] * F.log_softmax(out["fam"], -1)).sum(-1).mean()
    l_q = F.mse_loss(torch.sigmoid(out["q"]), y["q"]); l_wp = F.mse_loss(torch.sigmoid(out["wp"]), y["wp"])
    l_ag = F.cross_entropy(out["agL"].reshape(-1, 4), y["agL"].reshape(-1)) + F.cross_entropy(out["agR"].reshape(-1, 4), y["agR"].reshape(-1))
    return l_fam + 2.0 * l_q + l_wp + 0.5 * l_ag, dict(fam=l_fam.item(), q=l_q.item(), wp=l_wp.item(), ag=l_ag.item())

@torch.no_grad()
def predict_episode(model, d, mu, sd, dev, bs=256):
    n = len(d["frame"]); outs = {k: [] for k in ("fam", "q", "wp", "agL", "agR")}
    for s in range(0, n, bs):
        idx = np.stack([windows(None, i) for i in range(s, min(n, s + bs))]); x = torch.from_numpy((d["x"][idx] - mu) / sd).to(dev)
        o = model(x)
        for k in outs: outs[k].append(o[k][:, -1].float().cpu().numpy())     # causal: read the last token of each window
    o = {k: np.concatenate(v) for k, v in outs.items()}
    return dict(fam_p=torch.softmax(torch.from_numpy(o["fam"]), -1).numpy(), q=1 / (1 + np.exp(-o["q"])), wp=1 / (1 + np.exp(-o["wp"])),
                agL=o["agL"].argmax(-1), agR=o["agR"].argmax(-1))

def boundary_errors(fam_true, fam_pred, frame):
    """for each annotated transition, |first frame the prediction switches to the new family (within ±60 tokens) - true frame|."""
    errs = []; b = np.nonzero(np.diff(fam_true))[0] + 1
    for t in b:
        lo, hi = max(0, t - 60), min(len(fam_true), t + 60); new = fam_true[t]
        hit = [i for i in range(lo, hi) if fam_pred[i] == new and (i == 0 or fam_pred[i - 1] != new or i == lo)]
        cand = [i for i in range(lo, hi) if fam_pred[i] == new]
        errs.append(abs(frame[min(cand, key=lambda i: abs(i - t))] - frame[t]) if cand else np.nan)
    return np.array(errs, np.float32)

def evaluate(model, data, mu, sd, dev, out_dir, tag):
    model.eval(); rows = []; per_fam = np.zeros((8, 2)); conf = np.zeros((8, 8), int); ents = {"boundary": [], "mid": []}
    berr = []; q_mae = []; mono = []; ag_acc = []; preds = {}; chop_lead = []; chop_false = []
    for ep, d in data.items():
        if not d["val"]: continue
        p = predict_episode(model, d, mu, sd, dev); fp = p["fam_p"].argmax(-1); preds[ep] = p
        acc = (fp == d["fam"]).mean(); rows.append((ep, acc))
        for c in range(8): m = d["fam"] == c; per_fam[c] += [(fp[m] == c).sum(), m.sum()]
        np.add.at(conf, (d["fam"], fp), 1)
        ent = -(p["fam_p"] * np.log(p["fam_p"] + 1e-9)).sum(-1); near = np.zeros(len(fp), bool)
        for t in np.nonzero(np.diff(d["fam"]))[0] + 1: near[max(0, t - SOFT_TOK): t + SOFT_TOK] = True
        ents["boundary"] += ent[near].tolist(); ents["mid"] += ent[~near].tolist()
        berr += boundary_errors(d["fam"], fp, d["frame"]).tolist()
        q_mae.append(np.abs(p["q"] - d["q"]).mean()); dq = np.diff(p["q"]); mono.append((dq < -0.05).mean())
        ag_acc.append([(p["agL"] == d["agL"]).mean(), (p["agR"] == d["agR"]).mean()])
        # chop anticipation: does predicted q rise before the true slice? (knife-near-egg shortcut check)
        spike = np.nonzero(d["q"] >= 0.4)[0]
        if len(spike):
            ts = spike[0]; seg0 = np.nonzero(d["fam"] == 4)[0]; pre = np.arange(seg0[0] if len(seg0) else max(0, ts - 36), ts)
            first = np.nonzero(p["q"] >= 0.2)[0]; lead = (first[0] - ts) * STRIDE if len(first) else np.nan
            chop_lead.append(lead); chop_false.append(float((p["q"][pre] >= 0.2).mean()) if len(pre) else np.nan)
    berr = np.array(berr); ag = np.array(ag_acc)
    rep = dict(tag=tag, n_val=len(rows), family_acc=float(np.mean([r[1] for r in rows])),
               per_family_recall={FAMILIES[c]: (round(float(per_fam[c, 0] / per_fam[c, 1]), 3) if per_fam[c, 1] else None) for c in range(8)},
               boundary_timing_frames=dict(median=float(np.nanmedian(berr)), p90=float(np.nanpercentile(berr, 90)), missed=int(np.isnan(berr).sum()), n=int(len(berr))),
               q_mae=float(np.mean(q_mae)), q_monotonicity_violation_rate=float(np.mean(mono)),
               held_object_acc=dict(L=float(ag[:, 0].mean()), R=float(ag[:, 1].mean())),
               entropy=dict(boundary=float(np.mean(ents["boundary"])), mid_segment=float(np.mean(ents["mid"]))),
               chop_anticipation=dict(lead_frames_median=float(np.nanmedian(chop_lead)), lead_frames_min=float(np.nanmin(chop_lead)),
                                      episodes_predicting_cut_before_spike=int(np.sum(np.array(chop_lead) < 0)),
                                      false_cut_rate_during_approach=float(np.nanmean(chop_false))),
               confusion_rows_true_cols_pred=conf.tolist(), per_episode_acc={str(e): round(float(a), 3) for e, a in rows})
    np.savez(f"{out_dir}/val_predictions_{tag}.npz", **{f"ep{e}_{k}": v for e, p in preds.items() for k, v in p.items()})
    return rep

def clock_baseline(data):
    """family predicted from normalized episode time alone (train majority per 2%-bin) — the single-task 'kill' bar."""
    bins = np.zeros((50, 8)); 
    for d in data.values():
        if d["val"]: continue
        np.add.at(bins, (np.minimum(49, (d["tnorm"] * 50).astype(int)), d["fam"]), 1)
    maj = bins.argmax(1); accs = [(maj[np.minimum(49, (d["tnorm"] * 50).astype(int))] == d["fam"]).mean() for d in data.values() if d["val"]]
    return float(np.mean(accs))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/root/step0/stage_head_v1"); ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--no_vision", action="store_true"); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
    data = load_all(a.no_vision); tr = [e for e, d in data.items() if not d["val"]]; va = [e for e, d in data.items() if d["val"]]
    print(f"episodes: train {len(tr)} val {len(va)}; d_in={data[tr[0]]['x'].shape[1]}; clock baseline family acc = {clock_baseline(data):.3f}", flush=True)
    xs = np.concatenate([data[e]["x"] for e in tr]); mu = xs.mean(0); sd = xs.std(0) + 1e-3; del xs
    model = Head(data[tr[0]]["x"].shape[1]).to(dev); print("params: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    keys = [(e, i) for e in tr for i in range(len(data[e]["frame"]))]; steps = len(keys) // a.bs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps, pct_start=0.1)
    best = None
    for epoch in range(a.epochs):
        model.train(); perm = np.random.permutation(len(keys)); t0 = time.time(); agg = {}
        for s in range(steps):
            x, y = batch_from(data, tr, [keys[j] for j in perm[s * a.bs:(s + 1) * a.bs]], mu, sd, dev)
            loss, parts = losses(model(x), y); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            for k, v in parts.items(): agg[k] = agg.get(k, 0) + v / steps
        rep = evaluate(model, data, mu, sd, dev, a.out, "last")
        print(f"epoch {epoch+1}/{a.epochs} {time.time()-t0:.0f}s train {agg} | val fam_acc {rep['family_acc']:.3f} q_mae {rep['q_mae']:.3f} agL/R {rep['held_object_acc']['L']:.3f}/{rep['held_object_acc']['R']:.3f} boundary med {rep['boundary_timing_frames']['median']:.0f}f", flush=True)
        if best is None or rep["family_acc"] > best["family_acc"]:
            best = rep; torch.save(dict(model=model.state_dict(), mu=mu, sd=sd, d_in=data[tr[0]]["x"].shape[1], no_vision=a.no_vision, W=W, STRIDE=STRIDE), f"{a.out}/head_best.pt")
            os.replace(f"{a.out}/val_predictions_last.npz", f"{a.out}/val_predictions_best.npz")
    best["clock_baseline_family_acc"] = clock_baseline(data); best["train_episodes"] = tr; best["val_episodes"] = va
    json.dump(best, open(f"{a.out}/report.json", "w"), indent=1); print("BEST", json.dumps({k: best[k] for k in ("family_acc", "per_family_recall", "boundary_timing_frames", "q_mae", "q_monotonicity_violation_rate", "held_object_acc", "entropy", "chop_anticipation", "clock_baseline_family_acc")}), flush=True)
    print("TRAIN_DONE", flush=True)

if __name__ == "__main__": main()
