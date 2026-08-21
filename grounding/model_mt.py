"""v0.5 multi-task grounding model: frozen ViT backbone (DINOv2 ViT-B/14 or
DINOv3 ViT-B/16) + per-patch 3D depth lifting + category-query cross-attention
-> 180x180 heatmap + per-patch depth regression. Same head architecture as the
validated pilot; the only backbone-dependent parts are input size / patch grid.

  dinov2_vitb14 : input 518 -> 37x37 patches, torch.hub, tokens via
                  forward_features()["x_norm_patchtokens"]
  dinov3_vitb16 : input 512 -> 32x32 patches, HF transformers, tokens =
                  last_hidden_state[:, 1+4 registers:, :]

The dataset always serves RGB 518 / depth 148 (pilot cache layout); this model
resizes both on-GPU when the backbone wants 512/128 (bilinear-antialias for RGB,
area for depth -- a 1% rescale, documented).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mt_common import CX, CY, FX, FY, HM, W

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES = {
    "dinov2_vitb14": dict(img=518, grid=37, feat=768, kind="hub"),
    "dinov3_vitb16": dict(img=512, grid=32, feat=768, kind="hf",
                          hf_id="facebook/dinov3-vitb16-pretrain-lvd1689m"),
}


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


class GroundingModelMT(nn.Module):
    def __init__(self, n_categories, backbone="dinov2_vitb14", d_model=384,
                 n_enc=3, n_dec=2, nhead=6, hf_token=None):
        super().__init__()
        spec = BACKBONES[backbone]
        self.backbone_name = backbone
        self.img = spec["img"]
        self.grid = spec["grid"]
        self.dgrid = 4 * self.grid
        self.n_special = 0

        if spec["kind"] == "hub":
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone)
        else:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(spec["hf_id"], token=hf_token)
            self.n_special = 1 + self.backbone.config.num_register_tokens  # CLS+reg
        self.backbone.eval().requires_grad_(False)

        feat = spec["feat"]
        g2 = self.grid * self.grid
        self.vis_proj = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, d_model))
        self.depth_mlp = nn.Sequential(nn.Linear(3, 128), nn.GELU(),
                                       nn.Linear(128, d_model), nn.LayerNorm(d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, g2, d_model))
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

        # Unprojection rays at dgrid res in the ORIGINAL 720p intrinsics frame.
        s = W / self.dgrid
        u = (np.arange(self.dgrid) + 0.5) * s
        v = (np.arange(self.dgrid) + 0.5) * s
        uu, vv = np.meshgrid(u, v)
        dirs = np.stack([(uu - CX) / FX, (CY - vv) / FY, np.ones_like(uu)], 0)
        self.register_buffer("ray", torch.from_numpy(dirs.astype(np.float32)),
                             persistent=False)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("im_mean", mean, persistent=False)
        self.register_buffer("im_std", std, persistent=False)

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def head_state_dict(self):
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith("backbone.")}

    def normalize_rgb(self, rgb_u8):
        x = rgb_u8.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != self.img:
            x = F.interpolate(x, size=(self.img, self.img), mode="bilinear",
                              align_corners=False, antialias=True)
        return (x - self.im_mean) / self.im_std

    def backbone_tokens(self, x):
        if self.n_special == 0:                       # dinov2 hub
            return self.backbone.forward_features(x)["x_norm_patchtokens"].float()
        out = self.backbone(pixel_values=x).last_hidden_state
        tok = out[:, self.n_special:, :].float()      # drop CLS + registers
        assert tok.shape[1] == self.grid * self.grid, tok.shape
        return tok

    def forward(self, rgb_u8, depth, cat):
        """rgb_u8 [B,518,518,3] uint8 (resized on-GPU if backbone wants 512);
        depth [B,148,148] m (area-resized to dgrid if needed); cat [B] long.
        Returns logits [B,HM,HM], depth_map [B,grid,grid] (m)."""
        B = rgb_u8.shape[0]
        with torch.no_grad():
            tok = self.backbone_tokens(self.normalize_rgb(rgb_u8))

        if depth.shape[-1] != self.dgrid:
            depth = F.interpolate(depth.unsqueeze(1), size=(self.dgrid, self.dgrid),
                                  mode="area").squeeze(1)
        xyz = self.ray.unsqueeze(0) * depth.unsqueeze(1)          # [B,3,dg,dg]
        xyz = F.avg_pool2d(xyz, self.dgrid // self.grid)          # [B,3,g,g]
        d_tok = self.depth_mlp(xyz.flatten(2).transpose(1, 2))    # [B,g2,d]

        t = self.vis_proj(tok) + d_tok + self.pos_emb
        t = self.encoder(t)

        q = self.cat_emb(cat).unsqueeze(1)
        for blk in self.cross:
            q = blk(q, t)

        tn = self.ln_tok(t)
        sim = (tn @ self.ln_q(q).transpose(1, 2)) / (tn.shape[-1] ** 0.5)
        fmap = tn.transpose(1, 2).reshape(B, -1, self.grid, self.grid)
        smap = sim.transpose(1, 2).reshape(B, 1, self.grid, self.grid)
        logits = self.dec(torch.cat([fmap, smap], 1)).squeeze(1)

        depth_map = self.depth_head(tn).squeeze(-1).reshape(B, self.grid, self.grid)
        return logits, depth_map

    @staticmethod
    def sample_depth(depth_map, uv):
        g = uv / W * 2 - 1
        g = g.view(-1, 1, 1, 2)
        d = F.grid_sample(depth_map.unsqueeze(1), g, mode="bilinear",
                          padding_mode="border", align_corners=False)
        return d.view(-1)

    @staticmethod
    def argmax_uv(logits):
        B = logits.shape[0]
        idx = logits.flatten(1).argmax(1)
        v = (idx // HM).float()
        u = (idx % HM).float()
        s = W / HM
        return torch.stack([(u + 0.5) * s, (v + 0.5) * s], 1)

    @staticmethod
    def soft_argmax_uv(logits, k=2):
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
