"""Stage-head window dataset over the extended v0.5 cache.

A sample = one anchor cache-frame t of one episode:
  rgb [518,518,3] u8 (frames/f_%05d.jpg, current frame only)
  depth [148,148] f32 m
  hist_glob [T,768] mean DINO tokens, frames t-T+1..t, edge-padded at start
  hist_prop [T,61]
  targets per arm: soft stage distribution (v2 3.2 boundary blending: frames
  within BLEND_SEC of a transition get cosine-blended two-hot targets --
  the softness in the live demos is the intended output), phase id (-1 =
  masked), sincos progress + validity, active-literal id (-1 = masked),
  ledger [L] + per-literal tail-boost weights + per-episode base rates
  (v2 5: near-flip frames upweighted, smoothing target for p_sat).

Task balancing at the sampler level: uniform-per-task weights 1/n_items(task)
(same documented choice as mt_dataset).
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from common import (BLEND_SEC, FPS_CACHE, L_MAX, PSAT_TAIL_BOOST,
                    PSAT_TAIL_SEC, T_HIST, cache_dir)
from taxonomy import N_STAGES, TRANSITION, task_stage_mask

BLEND_F = max(1, int(round(BLEND_SEC * FPS_CACHE)))   # cache frames (~2 @ 6Hz)
TAIL_F = max(1, int(round(PSAT_TAIL_SEC * FPS_CACHE)))
TRANSITION_W = 0.3   # low-confidence transition labels (spec enrichment 1)


def soft_stage_targets(stage, seg_id, n_stages):
    """[N] hard ids -> [N, n_stages] soft targets with boundary blending.
    Within BLEND_F frames of a seg_id change, cosine-blend old/new (v2 3.2)."""
    N = len(stage)
    out = np.zeros((N, n_stages), dtype=np.float32)
    out[np.arange(N), stage] = 1.0
    bounds = np.flatnonzero(np.diff(seg_id) != 0) + 1     # first frame of new seg
    for b in bounds:
        s_old, s_new = stage[b - 1], stage[b]
        if s_old == s_new:
            continue
        for off in range(-BLEND_F, BLEND_F):
            i = b + off
            if 0 <= i < N:
                # w_new: cosine ramp 0->1 across the blend window centred on b
                u = (off + BLEND_F + 0.5) / (2 * BLEND_F)
                w = 0.5 * (1.0 - np.cos(np.pi * u))
                out[i] = 0.0
                out[i, s_new] = w
                out[i, s_old] = 1.0 - w
    return out


def psat_weights_and_base(ledger):
    """ledger [N, L] 0/1 targets -> (weights [N, L], base [L]).
    Frames within TAIL_F cache frames of a satisfaction flip get
    PSAT_TAIL_BOOST weight (near-miss states dominate otherwise); base is
    the per-episode positive rate, the smoothing anchor (v2 5)."""
    w = np.ones_like(ledger, dtype=np.float32)
    N, L = ledger.shape
    for j in range(L):
        flips = np.flatnonzero(np.diff(ledger[:, j]) != 0) + 1
        for f in flips:
            w[max(0, f - TAIL_F):min(N, f + TAIL_F + 1), j] = PSAT_TAIL_BOOST
    return w, ledger.mean(0).astype(np.float32)


class StageWindowDataset(Dataset):
    def __init__(self, episodes, t_hist=T_HIST, stride=2,
                 label_file="stage_labels.npz"):
        """episodes: [(task, file_idx)]; stride subsamples anchors (labels are
        near-constant at 6 Hz; stride 2 halves epoch cost losslessly).
        label_file (v4): stage_labels_posbins.npz trains the kill-ablation head
        on temporal-position bins -- every stage class live (bins are task-
        agnostic, no task mask)."""
        self.t_hist = t_hist
        self.posbins = "posbins" in label_file
        self.eps = []
        self.items = []
        self.item_task = []
        self._is_w = None
        for task, fi in episodes:
            d = cache_dir(task, fi)
            need = ["glob.npy", "proprio.npy", label_file, "depth.npy"]
            if not all(os.path.exists(os.path.join(d, f)) for f in need):
                raise FileNotFoundError(f"{d}: stage cache incomplete (run build_cache)")
            ei = len(self.eps)
            lab = dict(np.load(os.path.join(d, label_file), allow_pickle=False))
            if "ledger_lit_valid" not in lab:      # pre-v4 caches: all-valid
                lab["ledger_lit_valid"] = np.ones(L_MAX, dtype=np.uint8)
            lits = json.load(open(os.path.join(d, "literals.json")))
            N = len(lab["stage_left"])
            soft = {arm: soft_stage_targets(lab[f"stage_{arm}"],
                                            lab[f"seg_{arm}"], N_STAGES)
                    for arm in ("left", "right")}
            # low-confidence transition frames get TRANSITION_W in the stage CE
            stage_w = {arm: np.where(lab[f"stage_{arm}"] == TRANSITION,
                                     TRANSITION_W, 1.0).astype(np.float32)
                       for arm in ("left", "right")}
            led_w, led_base = psat_weights_and_base(
                np.asarray(lab["ledger"], dtype=np.float32))
            mask = (np.ones(N_STAGES, dtype=bool) if self.posbins
                    else np.array(task_stage_mask(task)))
            self.eps.append(dict(
                task=task, dir=d, lab=lab, soft=soft, stage_w=stage_w,
                led_w=led_w, led_base=led_base,
                glob=np.load(os.path.join(d, "glob.npy"), mmap_mode="r"),
                prop=np.load(os.path.join(d, "proprio.npy"), mmap_mode="r"),
                depth=np.load(os.path.join(d, "depth.npy"), mmap_mode="r"),
                lits=lits, stage_mask=mask))
            for i in range(0, N, stride):
                self.items.append((ei, i))
                self.item_task.append(task)

    def sampler_weights(self):
        from collections import Counter
        n = Counter(self.item_task)
        return torch.tensor([1.0 / n[t] for t in self.item_task], dtype=torch.double)

    @property
    def is_w(self):
        """Per-item IS-debias weights (mean 1 under the uniform-per-task
        sampler): w_i = n_task_i * Z / N with Z = sum_j 1/n_task_j (v2 5)."""
        if self._is_w is None:
            from collections import Counter
            n = Counter(self.item_task)
            ni = np.array([n[t] for t in self.item_task], dtype=np.float64)
            self._is_w = (ni * (1.0 / ni).sum() / len(ni)).astype(np.float32)
        return self._is_w

    def __len__(self):
        return len(self.items)

    def _rgb(self, ep, i):
        from PIL import Image
        p = os.path.join(ep["dir"], "frames", f"f_{i:05d}.jpg")
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)

    def __getitem__(self, idx):
        ei, i = self.items[idx]
        ep = self.eps[ei]
        lab = ep["lab"]
        t0 = i - self.t_hist + 1
        pad = max(0, -t0)
        sl = slice(max(0, t0), i + 1)
        hg = np.asarray(ep["glob"][sl], dtype=np.float32)
        hp = np.asarray(ep["prop"][sl], dtype=np.float32)
        if pad:                                          # edge-pad episode start
            hg = np.concatenate([np.repeat(hg[:1], pad, 0), hg])
            hp = np.concatenate([np.repeat(hp[:1], pad, 0), hp])

        prog = np.stack([lab["progress_left"][i], lab["progress_right"][i]])
        ang = 2 * np.pi * prog
        out = dict(
            rgb=torch.from_numpy(self._rgb(ep, i)),
            depth=torch.from_numpy(np.asarray(ep["depth"][i], dtype=np.float32)),
            hist_glob=torch.from_numpy(hg),
            hist_prop=torch.from_numpy(hp),
            stage_soft=torch.from_numpy(np.stack(
                [ep["soft"]["left"][i], ep["soft"]["right"][i]])),
            stage_w=torch.tensor([ep["stage_w"]["left"][i],
                                  ep["stage_w"]["right"][i]]),
            phase=torch.tensor([lab["phase_left"][i], lab["phase_right"][i]],
                               dtype=torch.long),
            progress=torch.from_numpy(
                np.stack([np.sin(ang), np.cos(ang)], -1).astype(np.float32)),
            progress_valid=torch.tensor(
                [bool(lab["seg_left"][i] >= 0), bool(lab["seg_right"][i] >= 0)]),
            active_lit=torch.tensor([lab["active_lit_left"][i],
                                     lab["active_lit_right"][i]], dtype=torch.long),
            ledger=torch.from_numpy(np.asarray(lab["ledger"][i], dtype=np.float32)),
            ledger_w=torch.from_numpy(ep["led_w"][i]),
            ledger_base=torch.from_numpy(ep["led_base"]),
            ledger_valid=torch.tensor(bool(lab["ledger_valid"])),
            ledger_lit_valid=torch.from_numpy(
                np.asarray(lab["ledger_lit_valid"], dtype=bool)),
            lit_pred=torch.tensor(ep["lits"]["pred"], dtype=torch.long),
            lit_tgt=torch.tensor(ep["lits"]["tgt"], dtype=torch.long),
            lit_ref=torch.tensor(ep["lits"]["ref"], dtype=torch.long),
            lit_mask=torch.tensor(ep["lits"]["mask"], dtype=torch.bool),
            stage_mask=torch.from_numpy(ep["stage_mask"]),
            is_w=torch.tensor(self.is_w[idx]),
        )
        return out


def split_episodes(tasks, heldout_files=(), heldout_tasks=()):
    """(train, val) episode lists. Held-out EPISODES gate memorization;
    held-out TASKS gate transfer (report both, spec metrics section)."""
    tr, va = [], []
    for task, fis in tasks.items():
        for fi in fis:
            ep = (task, fi)
            if task in heldout_tasks or (task, fi) in heldout_files:
                va.append(ep)
            else:
                tr.append(ep)
    return tr, va
