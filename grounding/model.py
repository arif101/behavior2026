"""M1 pilot grounding head on a frozen DINOv2 ViT-B/14 backbone.

Pathways:
  RGB 518x518 -> frozen DINOv2 -> 37x37 patch tokens (768) -> Linear -> d_model
  Depth 148x148 (meters) -> per-pixel unprojection to (x,y,z)_cam in the ORIGINAL
      720p intrinsics frame -> 4x4 avg-pool to 37x37x3 -> MLP -> d_model (added)
  Category id -> learned embedding query -> cross-attention over fused tokens ->
      coarse 37x37 query-token similarity map -> light conv decoder -> 180x180 logits
  Depth head: per-patch linear regression map (37x37, meters), bilinearly sampled
      at the (predicted or GT) pixel.

Trainable parameters ~10M; backbone frozen (no grads, bf16 autocast).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import FX, FY, CX, CY, W, DGRID, HM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CrossAttnBlock(nn.Module):
    def __init__(self, d, nhead):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv):
        a, _ = self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv))
        q = q + a
        return q + self.mlp(self.ln2(q))


class GroundingModel(nn.Module):
    def __init__(self, n_categories=1, d_model=384, n_enc=3, n_dec=2, nhead=6,
                 grid=37, backbone="dinov2_vitb14"):
        super().__init__()
        self.grid = grid
        self.backbone_name = backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone)
        self.backbone.eval().requires_grad_(False)

        self.vis_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d_model))
        self.depth_mlp = nn.Sequential(nn.Linear(3, 128), nn.GELU(),
                                       nn.Linear(128, d_model), nn.LayerNorm(d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, grid * grid, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc)

        self.cat_emb = nn.Embedding(n_categories, d_model)
        self.cross = nn.ModuleList(CrossAttnBlock(d_model, nhead) for _ in range(n_dec))
        self.ln_tok = nn.LayerNorm(d_model)
        self.ln_q = nn.LayerNorm(d_model)

        self.dec = nn.Sequential(
            nn.Conv2d(d_model + 1, 128, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Upsample(size=(HM, HM), mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, 3, padding=1))
        self.depth_head = nn.Linear(d_model, 1)

        # Unprojection ray grid at DGRID res, in ORIGINAL 720p intrinsics frame:
        # cell (i,j) center is 720p pixel u=(j+0.5)*720/DGRID, v=(i+0.5)*720/DGRID.
        s = W / DGRID
        u = (np.arange(DGRID) + 0.5) * s
        v = (np.arange(DGRID) + 0.5) * s
        uu, vv = np.meshgrid(u, v)
        dirs = np.stack([(uu - CX) / FX, (CY - vv) / FY, np.ones_like(uu)], 0)
        self.register_buffer("ray", torch.from_numpy(dirs.astype(np.float32)), persistent=False)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("im_mean", mean, persistent=False)
        self.register_buffer("im_std", std, persistent=False)

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def head_state_dict(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone.")}

    def normalize_rgb(self, rgb_u8):
        x = rgb_u8.permute(0, 3, 1, 2).float() / 255.0
        return (x - self.im_mean) / self.im_std

    def forward(self, rgb_u8, depth, cat):
        """rgb_u8 [B,518,518,3] uint8; depth [B,148,148] meters; cat [B] long.
        Returns logits [B,HM,HM], depth_map [B,grid,grid] (meters)."""
        B = rgb_u8.shape[0]
        with torch.no_grad():
            feats = self.backbone.forward_features(self.normalize_rgb(rgb_u8))
            tok = feats["x_norm_patchtokens"].float()          # [B,1369,768]

        # depth lift: (x,y,z)_cam per cell, avg-pooled to patch grid
        xyz = self.ray.unsqueeze(0) * depth.unsqueeze(1)        # [B,3,148,148]
        xyz = F.avg_pool2d(xyz, DGRID // self.grid)             # [B,3,37,37]
        d_tok = self.depth_mlp(xyz.flatten(2).transpose(1, 2))  # [B,1369,d]

        t = self.vis_proj(tok) + d_tok + self.pos_emb
        t = self.encoder(t)

        q = self.cat_emb(cat).unsqueeze(1)                      # [B,1,d]
        for blk in self.cross:
            q = blk(q, t)

        tn = self.ln_tok(t)
        sim = (tn @ self.ln_q(q).transpose(1, 2)) / (tn.shape[-1] ** 0.5)  # [B,1369,1]
        fmap = tn.transpose(1, 2).reshape(B, -1, self.grid, self.grid)
        smap = sim.transpose(1, 2).reshape(B, 1, self.grid, self.grid)
        logits = self.dec(torch.cat([fmap, smap], 1)).squeeze(1)           # [B,HM,HM]

        depth_map = self.depth_head(tn).squeeze(-1).reshape(B, self.grid, self.grid)
        return logits, depth_map

    @staticmethod
    def sample_depth(depth_map, uv):
        """Bilinearly sample [B,g,g] depth map at 720p pixel coords uv [B,2]."""
        g = uv / W * 2 - 1                                      # align_corners=False
        g = g.view(-1, 1, 1, 2)
        d = F.grid_sample(depth_map.unsqueeze(1), g, mode="bilinear",
                          padding_mode="border", align_corners=False)
        return d.view(-1)

    @staticmethod
    def argmax_uv(logits):
        """Heatmap argmax -> 720p pixel coords (cell centers). logits [B,HM,HM]."""
        B = logits.shape[0]
        idx = logits.flatten(1).argmax(1)
        v = (idx // HM).float()
        u = (idx % HM).float()
        s = W / HM
        return torch.stack([(u + 0.5) * s, (v + 0.5) * s], 1)

    @staticmethod
    def soft_argmax_uv(logits, k=2):
        """Sub-cell refinement: softmax-weighted centroid in a (2k+1)^2 window
        around the argmax. Removes the heatmap-grid quantization floor."""
        B = logits.shape[0]
        idx = logits.flatten(1).argmax(1)
        vi = (idx // HM).long()
        ui = (idx % HM).long()
        off = torch.arange(-k, k + 1, device=logits.device)
        vv = (vi.view(-1, 1, 1) + off.view(1, -1, 1)).clamp(0, HM - 1)
        uu = (ui.view(-1, 1, 1) + off.view(1, 1, -1)).clamp(0, HM - 1)
        win = logits[torch.arange(B).view(-1, 1, 1), vv, uu]
        w = torch.softmax(win.flatten(1), 1).view(B, 2 * k + 1, 2 * k + 1)
        vc = (w.sum(2) * vv[:, :, 0].float()).sum(1)
        uc = (w.sum(1) * uu[:, 0, :].float()).sum(1)
        s = W / HM
        return torch.stack([(uc + 0.5) * s, (vc + 0.5) * s], 1)
