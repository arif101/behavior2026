"""v0.5 multi-task grounding dataset over the JPEG/npy cache.

A training sample = (frame RGB-D, ONE queried category present in that frame,
mixture-of-gaussians target with one mode per VISIBLE instance of the category;
occluded instances contribute NO gaussian -- adaptive occlusion gate, see
mt_common). If the queried category is present but ALL instances are occluded,
the target is EMPTY (trained toward a uniform map = negative supervision
matching serve-time validity-mask semantics); such negative queries are drawn
with probability NEG_P when available. Query category is drawn uniformly among
eligible categories in the frame.

Task balancing: per-batch UNIFORM-PER-TASK via WeightedRandomSampler weights
1/n_items(task) (documented choice; inverse-frequency over tasks, uniform over
frames within a task, uniform over categories within a frame).
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from mt_common import CACHE, HM, W, instance_thresholds, visible_mask

MAX_MODES = 24   # hard cap on instances per (frame, category); >20 never happens
NEG_P = 0.25     # prob of drawing an all-occluded (empty-target) query when present


class MTGroundingDataset(Dataset):
    """episodes: list of (task, file_idx). Items are FRAMES with >=1 visible
    in-vocab category; the queried category is sampled in __getitem__ unless
    fixed_queries=True (eval mode: __getitem__ takes (frame_item, cat) pairs).
    """

    def __init__(self, episodes, rng_seed=0, with_negatives=True):
        self.eps = []           # [(task, fi, dir)]
        self.dep = {}
        self.lab = {}
        self.items = []         # (ep_idx, local_frame_idx)
        self.frame_cats = []    # per item: np.array of visible cat ids
        self.frame_negcats = [] # per item: cat ids present but ALL occluded
        self.item_task = []     # per item: task name
        self.with_negatives = with_negatives
        self.rng = np.random.RandomState(rng_seed)
        for task, fi in episodes:
            d = os.path.join(CACHE, task, f"ep{fi:03d}")
            if not os.path.exists(os.path.join(d, ".done")):
                raise FileNotFoundError(d)
            ei = len(self.eps)
            self.eps.append((task, fi, d))
            self.dep[ei] = np.load(os.path.join(d, "depth.npy"), mmap_mode="r")
            lab = dict(np.load(os.path.join(d, "lab.npz"), allow_pickle=False))
            self.lab[ei] = lab
            margin = lab["margin"].astype(np.float32)
            disp = np.load(os.path.join(d, "disp.npy"))
            thr = instance_thresholds(margin, lab["inframe"], disp)
            vis = visible_mask(lab["inframe"], margin, thr)      # [N,K]
            invocab = (lab["obj_cats"] >= 0)[None, :]
            vis &= invocab
            present = lab["inframe"] & invocab
            lab["vis"] = vis
            lab["thr"] = thr
            cats = lab["obj_cats"]
            for i in range(vis.shape[0]):
                cs = np.unique(cats[vis[i]])
                ncs = np.setdiff1d(np.unique(cats[present[i]]), cs)
                if len(cs) or (with_negatives and len(ncs)):
                    self.items.append((ei, i))
                    self.frame_cats.append(cs)
                    self.frame_negcats.append(ncs)
                    self.item_task.append(task)

    def task_weights(self):
        """Uniform-per-task sampling weights (one per item)."""
        from collections import Counter
        cnt = Counter(self.item_task)
        return np.array([1.0 / cnt[t] for t in self.item_task], dtype=np.float64)

    def __len__(self):
        return len(self.items)

    def n_modes(self, idx, cat):
        ei, i = self.items[idx]
        lab = self.lab[ei]
        return int(((lab["obj_cats"] == cat) & lab["vis"][i]).sum())

    def get(self, idx, cat=None):
        ei, i = self.items[idx]
        task, fi, d = self.eps[ei]
        lab = self.lab[ei]
        if cat is None:
            cs = self.frame_cats[idx]
            ncs = self.frame_negcats[idx]
            use_neg = self.with_negatives and len(ncs) and \
                (len(cs) == 0 or self.rng.rand() < NEG_P)
            pool = ncs if use_neg else cs
            cat = int(pool[self.rng.randint(len(pool))])
        sel = (lab["obj_cats"] == cat) & lab["vis"][i]
        uvz = lab["uvz"][i][sel]                              # [k,3]
        k = min(len(uvz), MAX_MODES)
        uvs = np.zeros((MAX_MODES, 2), dtype=np.float32)
        zs = np.zeros((MAX_MODES,), dtype=np.float32)
        mm = np.zeros((MAX_MODES,), dtype=bool)
        uvs[:k] = uvz[:k, :2]
        zs[:k] = uvz[:k, 2]
        mm[:k] = True

        from PIL import Image
        rgb = np.asarray(Image.open(os.path.join(d, "frames", f"f_{i:05d}.jpg")))
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb)),
            "depth": torch.from_numpy(self.dep[ei][i].astype(np.float32)),
            "cat": torch.tensor(cat, dtype=torch.long),
            "uvs": torch.from_numpy(uvs),
            "zs": torch.from_numpy(zs),
            "mode_mask": torch.from_numpy(mm),
            "ep": torch.tensor(ei, dtype=torch.long),
            "frame": torch.tensor(int(lab["frame_k"][i]), dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }

    def __getitem__(self, idx):
        return self.get(idx)


def mixture_gaussian_target(uvs, mode_mask, hm=HM, sigma=3.0, device="cpu"):
    """Mixture-of-gaussians heatmap, one mode per VISIBLE instance, normalized
    to sum 1. Rows with NO visible mode (all-occluded negative queries) get a
    UNIFORM target (max-entropy = 'nothing to point at'; serve-time validity
    comes from peakiness). uvs: [B,M,2] (720p), mode_mask: [B,M] bool."""
    scale = hm / float(W)
    g = torch.arange(hm, device=device, dtype=torch.float32) + 0.5
    gx = g.view(1, 1, 1, hm)
    gy = g.view(1, 1, hm, 1)
    cx = (uvs[:, :, 0] * scale).view(*uvs.shape[:2], 1, 1)
    cy = (uvs[:, :, 1] * scale).view(*uvs.shape[:2], 1, 1)
    t = torch.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * sigma ** 2))
    t = (t * mode_mask.view(*uvs.shape[:2], 1, 1)).sum(1)
    s = t.sum(dim=(1, 2), keepdim=True)
    uniform = torch.full_like(t, 1.0 / (hm * hm))
    return torch.where(s < 1e-8, uniform, t / s.clamp_min(1e-8))


def sample_depth_multi(depth_map, uvs, mode_mask):
    """Sample [B,g,g] depth maps at per-mode 720p pixels. Returns flat tensors
    (pred, batch_index) over valid modes."""
    import torch.nn.functional as F
    B, M, _ = uvs.shape
    g = uvs / W * 2 - 1
    d = F.grid_sample(depth_map.unsqueeze(1), g.view(B, M, 1, 2),
                      mode="bilinear", padding_mode="border",
                      align_corners=False).view(B, M)
    return d[mode_mask]


def load_vocab():
    j = json.load(open(os.path.join(CACHE, "vocab.json")))
    return j["vocab"], j["n_base"]
