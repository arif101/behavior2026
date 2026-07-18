"""Serve-time stage estimator -- the spec's API contract.

    est = StageEstimator("ckpt/best.pt", "ckpt/stage_medians.json", device)
    est.set_task(task_name, literals)        # director's BDDL parse
    out = est.update(dino_tokens, depth, proprio)
    # {"left":  {"stage_id", "stage_name", "active_literal", "progress",
    #            "entropy", "z_stage"},
    #  "right": {...},
    #  "ledger": [p_satisfied_per_literal],
    #  "stage_age_ratio": {"left": ..., "right": ...}}

dino_tokens is the SHARED backbone pass (bus (O)) -- this module never touches
the backbone. History is a ring buffer of (mean token, proprio) pairs, i.e.
WINDOW memory only; stage_age_ratio is a serve-side stopwatch against the
training-label median durations (no learning; "grasp at 3x median" is a
metacog feature, not a model output). The hard argmax here is informational --
policy conditioning uses z_stage (soft mixture) and the director applies its
own hysteresis to stage switches (v1.1 rule 1).
"""

import json
from collections import deque

import torch

from common import FPS_CACHE, L_MAX, T_HIST
from model import StageHead
from taxonomy import encode_literals, stage_name, task_stage_mask


class StageEstimator:
    def __init__(self, ckpt, medians_path, device="cuda", t_hist=T_HIST):
        self.device = device
        self.model = StageHead().to(device).eval()
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
        """dino_tokens [1,1369,768] (shared pass); depth [1,148,148] m;
        proprio [1,61]. Call once per perception step (~6 Hz)."""
        assert self.task is not None, "set_task() first"
        self.hist.append((dino_tokens.mean(1), proprio))
        hg = torch.stack([h[0] for h in self.hist], 1)       # [1,T,768]
        hp = torch.stack([h[1] for h in self.hist], 1)       # [1,T,61]

        out = self.model(dino_tokens, depth, hg, hp, *self.lit,
                         self.lit_mask, self.smask)
        z = self.model.z_stage(out["stage_logits"])          # [1,2,Dz]
        ent = self.model.entropy(out["stage_logits"])        # [1,2]
        p_led = torch.sigmoid(out["ledger_logits"])[0, :self.n_lit]

        res = {"ledger": p_led.tolist(), "stage_age_ratio": {}}
        for a, arm in enumerate(("left", "right")):
            logits = out["stage_logits"][0, a]
            sid = int(logits.argmax())
            sin, cos = out["progress"][0, a].tolist()
            prog = (torch.atan2(torch.tensor(sin), torch.tensor(cos)).item()
                    / (2 * 3.141592653589793)) % 1.0
            lit_id = int(out["lit_logits"][0, a].argmax()) if self.n_lit else -1
            res[arm] = dict(stage_id=sid, stage_name=stage_name(sid),
                            active_literal=lit_id, progress=prog,
                            entropy=float(ent[0, a]), z_stage=z[0, a].tolist())
            # stage-age stopwatch (age of the *argmax* stage vs label median)
            if sid == self.cur_stage[arm]:
                self.stage_len[arm] += 1
            else:
                self.cur_stage[arm] = sid
                self.stage_len[arm] = 1
            age_s = self.stage_len[arm] / FPS_CACHE
            res["stage_age_ratio"][arm] = age_s / max(1e-6, self._median(sid))
        return res
