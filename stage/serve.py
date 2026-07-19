"""Serve-time stage estimator -- the spec v2 API contract (section 2).

    est = StageEstimator("ckpt/best.pt", "ckpt/stage_medians.json", device)
    est.set_task(task_name, literals)        # director's BDDL parse
    out = est.update(dino_tokens, depth, proprio)
    # {"left"/"right": {"stage_dist":  [S] softmax (task-masked, calibrated),
    #                   "stage_id", "stage_name",       # argmax, informational
    #                   "progress": float 0-1,
    #                   "active_literal_dist": [n_lit] softmax},
    #  "p_sat":   [n_lit] sigmoid -- instantaneous, NOT latched,
    #  "entropy": {"left": nats, "right": nats},
    #  "stage_age_ratio": {"left": ..., "right": ...}}   # serve-side stopwatch

v2 contract: this head ships NUMBERS, never embeddings -- the z_stage soft
mixture (sum_i p_i E_i + sincos(progress) via AdaLN) is computed policy-side
from stage_dist/progress. Consumers: director reflex tier latches p_sat with
hysteresis; metacog reads entropy + stage_age_ratio; Opus deliberative tier
reads the whole dict as its situation briefing.

dino_tokens is the SHARED backbone pass (bus (O)) -- this module never touches
the backbone. History is a ring buffer of (mean token, proprio) pairs, i.e.
WINDOW memory only; stage_age_ratio is a stopwatch against training-label
median durations (no learning). Hard argmaxes here are informational -- the
director applies its own hysteresis to stage switches.
"""

import json
from collections import deque

import torch

from common import FPS_CACHE, L_MAX, T_HIST
from model import StageHead
from taxonomy import encode_literals, stage_name, task_stage_mask


class StageEstimator:
    def __init__(self, ckpt, medians_path, device="cuda", t_hist=T_HIST,
                 grid=37, feat=768):
        self.device = device
        self.model = StageHead(grid=grid, feat=feat).to(device).eval()
        self.model.load_state_dict(torch.load(ckpt, map_location=device))
        self.medians = json.load(open(medians_path))
        self.t_hist = t_hist
        self.task = None
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=self.t_hist)
        self.cur_stage = {"left": None, "right": None}
        self.stage_len = {"left": 0, "right": 0}

    def set_task(self, task, literals):
        """literals: [{'predicate','target','reference'}] from the director."""
        pred, tgt, ref, mask = encode_literals(literals, L_MAX)
        dev = self.device
        self.task = task
        self.lit = tuple(torch.tensor(x, device=dev).unsqueeze(0)
                         for x in (pred, tgt, ref))
        self.lit_mask = torch.tensor(mask, device=dev, dtype=torch.bool).unsqueeze(0)
        self.smask = torch.tensor(task_stage_mask(task), device=dev,
                                  dtype=torch.bool).unsqueeze(0)
        self.n_lit = sum(mask)
        self.reset()

    def _median(self, stage_id):
        return self.medians.get(f"{self.task}|{stage_id}") or \
            self.medians.get(f"__global__|{stage_id}", 10.0)

    @torch.no_grad()
    def update(self, dino_tokens, depth, proprio):
        """dino_tokens [1,g*g,feat] (shared pass); depth [1,148,148] m;
        proprio [1,61]. Call once per perception step (~6 Hz)."""
        assert self.task is not None, "set_task() first"
        self.hist.append((dino_tokens.mean(1), proprio))
        hg = torch.stack([h[0] for h in self.hist], 1)       # [1,T,feat]
        hp = torch.stack([h[1] for h in self.hist], 1)       # [1,T,61]

        out = self.model(dino_tokens, depth, hg, hp, *self.lit,
                         self.lit_mask, self.smask)
        ent = self.model.entropy(out["stage_logits"])        # [1,2]
        p_sat = torch.sigmoid(out["ledger_logits"])[0, :self.n_lit]

        res = {"p_sat": p_sat.tolist(), "entropy": {},
               "stage_age_ratio": {}}
        for a, arm in enumerate(("left", "right")):
            logits = out["stage_logits"][0, a]
            dist = torch.softmax(logits, -1)
            sid = int(logits.argmax())
            sin, cos = out["progress"][0, a].tolist()
            prog = (torch.atan2(torch.tensor(sin), torch.tensor(cos)).item()
                    / (2 * 3.141592653589793)) % 1.0
            lit_dist = (torch.softmax(out["lit_logits"][0, a, :self.n_lit], -1)
                        .tolist() if self.n_lit else [])
            res[arm] = dict(stage_dist=dist.tolist(), stage_id=sid,
                            stage_name=stage_name(sid), progress=prog,
                            active_literal_dist=lit_dist)
            res["entropy"][arm] = float(ent[0, a])
            # stage-age stopwatch (age of the *argmax* stage vs label median)
            if sid == self.cur_stage[arm]:
                self.stage_len[arm] += 1
            else:
                self.cur_stage[arm] = sid
                self.stage_len[arm] = 1
            age_s = self.stage_len[arm] / FPS_CACHE
            res["stage_age_ratio"][arm] = age_s / max(1e-6, self._median(sid))
        return res
