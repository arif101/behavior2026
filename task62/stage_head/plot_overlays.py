"""Overlay plots for held-out episodes (STAGE_HEAD_SPEC_v2 §6.1 style): annotated vs predicted family,
q (true staircase vs predicted), within-skill progress, held-object per arm, and family-entropy.
Usage (behavior env, has matplotlib): python task62/stage_head/plot_overlays.py --run /root/step0/stage_head_v1 [--eps 620010,620180]
"""
import argparse, json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FEATS = "/root/step0/stage_feats"; FAM = ["navigate", "open_door", "pick_up_from", "place_on", "chop", "place_in", "close_door", "other"]
HELD = ["none", "egg/half", "knife", "other"]
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--eps", default=""); ap.add_argument("--tag", default="best"); a = ap.parse_args()
P = np.load(f"{a.run}/val_predictions_{a.tag}.npz"); eps = [int(x) for x in a.eps.split(",")] if a.eps else sorted({int(k.split("_")[0][2:]) for k in P.files})
os.makedirs(f"{a.run}/plots", exist_ok=True)
for ep in eps:
    d = np.load(f"{FEATS}/ep{ep}.npz"); t = d["frame"] / 30.0; fp = P[f"ep{ep}_fam_p"]; ent = -(fp * np.log(fp + 1e-9)).sum(-1)
    fig, ax = plt.subplots(5, 1, figsize=(16, 12), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.5, 1, 1.2, 1]))
    ax[0].imshow(fp.T, aspect="auto", extent=[t[0], t[-1], 7.5, -0.5], cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax[0].step(t, d["fam"], where="post", color="red", lw=1.2, label="annotation"); ax[0].set_yticks(range(8)); ax[0].set_yticklabels(FAM); ax[0].legend(loc="upper right"); ax[0].set_title(f"ep{ep}: family — predicted distribution (blue) vs annotation (red)")
    ax[1].step(t, d["q"], where="post", color="red", label="q (reward spikes)"); ax[1].plot(t, P[f"ep{ep}_q"], color="tab:blue", label="q predicted"); ax[1].set_ylim(-0.05, 1.05); ax[1].legend(loc="upper left"); ax[1].set_ylabel("progress")
    seg_len = np.maximum(1, d["seg_end"] - d["seg_start"]); wp = np.clip((d["frame"] - d["seg_start"]) / seg_len, 0, 1)
    ax[2].plot(t, wp, color="red", lw=0.8, label="within-skill"); ax[2].plot(t, P[f"ep{ep}_wp"], color="tab:blue", lw=0.8, label="predicted"); ax[2].set_ylim(-0.05, 1.05); ax[2].legend(loc="upper left")
    ax[3].step(t, d["agL"] + 0.05, where="post", color="red", lw=1, label="L true"); ax[3].step(t, P[f"ep{ep}_agL"] - 0.05, where="post", color="tab:blue", lw=1, label="L pred")
    ax[3].step(t, d["agR"] + 4.05, where="post", color="darkred", lw=1, label="R true"); ax[3].step(t, P[f"ep{ep}_agR"] + 3.95, where="post", color="tab:cyan", lw=1, label="R pred")
    ax[3].set_yticks(list(range(4)) + [4 + i for i in range(4)]); ax[3].set_yticklabels([f"L:{h}" for h in HELD] + [f"R:{h}" for h in HELD]); ax[3].legend(loc="upper right", ncol=4, fontsize=8)
    ax[4].plot(t, ent, color="gray"); ax[4].set_ylabel("family entropy"); ax[4].set_xlabel("time (s)")
    for b in np.nonzero(np.diff(d["fam"]))[0] + 1:
        for x in ax: x.axvline(t[b], color="k", alpha=0.15, lw=0.8)
    fig.tight_layout(); fig.savefig(f"{a.run}/plots/ep{ep}.png", dpi=80); plt.close(fig)
print("plots:", sorted(glob.glob(f"{a.run}/plots/*.png")))
