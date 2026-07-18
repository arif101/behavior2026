"""Stage / subtask head on the shared frozen DINO backbone (Phase 2).

Owns NO backbone: consumes the same 37x37x768 patch tokens the grounding head
reads (one backbone pass per step, bus (O)). Memory badge WINDOW: the only
temporal input is a 2-5 s rolling window; "what's done" lives in the director's
ledger, not in here.

Two-branch design (the serve-time economics drive it):
  CURRENT frame, full spatial detail: patch tokens + depth-unprojected xyz
    (grounding's depth-lift, same ray convention) -> d_model -> 2-layer encoder
    -> 4 learned queries attention-pool -> spatial summary tokens.
  HISTORY, cheap: per past frame only the MEAN patch token (768-d, free at
    serve -- each was computed at its own step) + 61-d proprio -> GRU. No
    backbone re-runs on history, training caches are 768 floats/frame.

Task interface = goal-literal tokens (predicate + hashed target/reference
category embeddings). Per-arm queries cross-attend over [spatial, temporal,
literal] tokens; the literal attention IS the active-literal output; literal
tokens + temporal state -> instantaneous P(satisfied) per literal (the
director latches -- head never accumulates).

Outputs per arm: stage distribution over the OFFICIAL skill taxonomy
(task-masked logits), 6-phase distribution, sincos progress, active-literal
distribution; shared: ledger logits; exports: z_stage soft mixture
(v1.1: sum_i p_i E_i, never argmax) and distribution entropy (metacog).

~12M trainable params (< 20M budget), bf16-safe.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (CX, CY, D_MODEL, D_ZSTAGE, DGRID, FX, FY, GRID, L_MAX,
                    N_HEAD, N_PHASES, PROPRIO_DIM, W)
from taxonomy import CAT_BUCKETS, N_PREDICATES, N_STAGES

NEG_INF = -1e4  # mask value; bf16-safe


class CrossAttnBlock(nn.Module):
    """Same block as grounding/model.py (pre-LN cross-attn + MLP)."""

    def __init__(self, d, nhead):
        super().__init__()
        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, q, kv, kv_mask=None):
        a, _ = self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv),
                         key_padding_mask=kv_mask)
        q = q + a
        return q + self.mlp(self.ln2(q))


class StageHead(nn.Module):
    def __init__(self, d_model=D_MODEL, nhead=N_HEAD, n_enc=2, n_fuse=2,
                 n_pool_q=4, n_stages=N_STAGES, l_max=L_MAX):
        super().__init__()
        self.n_stages = n_stages
        self.l_max = l_max

        # ---- current-frame spatial branch (mirrors grounding's front end)
        self.vis_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d_model))
        self.depth_mlp = nn.Sequential(nn.Linear(3, 128), nn.GELU(),
                                       nn.Linear(128, d_model), nn.LayerNorm(d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, GRID * GRID, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4 * d_model, batch_first=True,
            norm_first=True, activation="gelu", dropout=0.0)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc)
        self.pool_q = nn.Parameter(torch.zeros(1, n_pool_q, d_model))
        nn.init.trunc_normal_(self.pool_q, std=0.02)
        self.pool = CrossAttnBlock(d_model, nhead)

        # ---- history branch (mean DINO token + proprio -> GRU)
        self.hist_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, d_model))
        self.prop_mlp = nn.Sequential(nn.Linear(PROPRIO_DIM, 128), nn.GELU(),
                                      nn.Linear(128, d_model), nn.LayerNorm(d_model))
        self.gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)

        # ---- goal-literal tokens
        self.emb_pred = nn.Embedding(N_PREDICATES, d_model)
        self.emb_tgt = nn.Embedding(CAT_BUCKETS, d_model)
        self.emb_ref = nn.Embedding(CAT_BUCKETS, d_model)
        self.lit_ln = nn.LayerNorm(d_model)

        # ---- fusion trunk: [spatial(4) | temporal(1) | literal(L)] tokens
        self.type_emb = nn.Embedding(3, d_model)
        self.arm_q = nn.Parameter(torch.zeros(1, 2, d_model))   # left, right
        nn.init.trunc_normal_(self.arm_q, std=0.02)
        self.fuse = nn.ModuleList(CrossAttnBlock(d_model, nhead) for _ in range(n_fuse))

        # ---- heads
        self.stage_head = nn.Linear(d_model, n_stages)
        self.phase_head = nn.Linear(d_model, N_PHASES)
        self.prog_head = nn.Linear(d_model, 2)                  # sincos
        self.lit_q_proj = nn.Linear(d_model, d_model)           # active-literal dot
        self.lit_k_proj = nn.Linear(d_model, d_model)
        self.ledger_mlp = nn.Sequential(nn.Linear(2 * d_model, 256), nn.GELU(),
                                        nn.Linear(256, 1))
        self.stage_emb = nn.Embedding(n_stages, D_ZSTAGE)       # z_stage table

        # unprojection ray grid, identical convention to grounding/model.py
        s = W / DGRID
        u = (np.arange(DGRID) + 0.5) * s
        v = (np.arange(DGRID) + 0.5) * s
        uu, vv = np.meshgrid(u, v)
        dirs = np.stack([(uu - CX) / FX, (CY - vv) / FY, np.ones_like(uu)], 0)
        self.register_buffer("ray", torch.from_numpy(dirs.astype(np.float32)),
                             persistent=False)

    def trainable_parameters(self):
        return list(self.parameters())      # backbone lives elsewhere

    def forward(self, tok, depth, hist_glob, hist_prop,
                lit_pred, lit_tgt, lit_ref, lit_mask, stage_mask):
        """tok [B,1369,768] frozen DINO patch tokens (shared bus, no grad);
        depth [B,148,148] m; hist_glob [B,T,768] mean patch token per past
        frame (oldest first, current last); hist_prop [B,T,61];
        lit_* [B,L] long; lit_mask [B,L] bool (True = real literal);
        stage_mask [B,S] bool (True = skill live in this task).
        """
        B = tok.shape[0]

        # current-frame spatial branch
        xyz = self.ray.unsqueeze(0) * depth.unsqueeze(1)          # [B,3,148,148]
        xyz = F.avg_pool2d(xyz, DGRID // GRID)                    # [B,3,37,37]
        d_tok = self.depth_mlp(xyz.flatten(2).transpose(1, 2))    # [B,1369,d]
        t = self.vis_proj(tok.float()) + d_tok + self.pos_emb
        t = self.encoder(t)
        spatial = self.pool(self.pool_q.expand(B, -1, -1), t)     # [B,4,d]

        # history branch
        h = self.hist_proj(hist_glob.float()) + self.prop_mlp(hist_prop)
        _, hn = self.gru(h)
        temporal = hn[-1].unsqueeze(1)                            # [B,1,d]

        # literal tokens
        lit = self.lit_ln(self.emb_pred(lit_pred) + self.emb_tgt(lit_tgt)
                          + self.emb_ref(lit_ref))                # [B,L,d]

        # fusion trunk
        trunk = torch.cat([
            spatial + self.type_emb.weight[0],
            temporal + self.type_emb.weight[1],
            lit + self.type_emb.weight[2]], dim=1)                # [B,4+1+L,d]
        pad = torch.zeros(B, spatial.shape[1] + 1, dtype=torch.bool,
                          device=tok.device)
        trunk_mask = torch.cat([pad, ~lit_mask], dim=1)           # True = ignore

        q = self.arm_q.expand(B, -1, -1)
        for blk in self.fuse:
            q = blk(q, trunk, kv_mask=trunk_mask)                 # [B,2,d]

        # per-arm heads (task-masked stage logits)
        stage_logits = self.stage_head(q)
        stage_logits = stage_logits.masked_fill(~stage_mask.unsqueeze(1), NEG_INF)
        phase_logits = self.phase_head(q)
        progress = torch.tanh(self.prog_head(q))                  # sincos in [-1,1]

        # active literal = scaled dot attention of arm queries over literals
        lq = self.lit_q_proj(q)                                   # [B,2,d]
        lk = self.lit_k_proj(lit)                                 # [B,L,d]
        lit_logits = lq @ lk.transpose(1, 2) / (lq.shape[-1] ** 0.5)
        lit_logits = lit_logits.masked_fill(~lit_mask.unsqueeze(1), NEG_INF)

        # ledger: instantaneous P(satisfied) per literal
        led_in = torch.cat([lit, temporal.expand(-1, lit.shape[1], -1)], dim=-1)
        ledger_logits = self.ledger_mlp(led_in).squeeze(-1)       # [B,L]

        return dict(stage_logits=stage_logits, phase_logits=phase_logits,
                    progress=progress, lit_logits=lit_logits,
                    ledger_logits=ledger_logits)

    # ---- serve-time exports -------------------------------------------------
    def z_stage(self, stage_logits):
        """Soft mixture sum_i p_i E_i per arm [B,2,D_ZSTAGE] (v1.1: soft
        conditioning for policy AdaLN; hysteresis voting is director-side)."""
        p = torch.softmax(stage_logits, dim=-1)
        return p @ self.stage_emb.weight

    @staticmethod
    def entropy(logits):
        """Distribution entropy (nats) -- metacog feature."""
        logp = torch.log_softmax(logits, dim=-1)
        return -(logp.exp() * logp).sum(-1)


if __name__ == "__main__":
    m = StageHead()
    n = sum(p.numel() for p in m.trainable_parameters())
    print(f"trainable params: {n/1e6:.2f}M (budget < 20M)")
