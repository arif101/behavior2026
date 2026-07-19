"""CPU-runnable smoke test: model forward/backward shapes, loss masking,
soft-boundary targets, serve loop, param budget. No data, no backbone."""

import numpy as np
import torch

from common import DGRID, L_MAX, PROPRIO_DIM, T_HIST
from dataset import soft_stage_targets
from model import StageHead
from taxonomy import N_STAGES, task_stage_mask


def fake_batch(B=2):
    g = torch.Generator().manual_seed(0)
    smask = torch.tensor([task_stage_mask("turning_on_radio")] * B)
    lit_mask = torch.zeros(B, L_MAX, dtype=torch.bool)
    lit_mask[:, :3] = True
    return dict(
        tok=torch.randn(B, 37 * 37, 768, generator=g),
        depth=torch.rand(B, DGRID, DGRID, generator=g) * 3,
        hist_glob=torch.randn(B, T_HIST, 768, generator=g),
        hist_prop=torch.randn(B, T_HIST, PROPRIO_DIM, generator=g),
        lit_pred=torch.randint(0, 10, (B, L_MAX), generator=g),
        lit_tgt=torch.randint(0, 256, (B, L_MAX), generator=g),
        lit_ref=torch.randint(0, 256, (B, L_MAX), generator=g),
        lit_mask=lit_mask, stage_mask=smask)


def test_model():
    m = StageHead()
    n = sum(p.numel() for p in m.parameters())
    assert n < 20e6, f"param budget blown: {n/1e6:.1f}M"
    b = fake_batch()
    out = m(**b)
    assert out["stage_logits"].shape == (2, 2, N_STAGES)
    assert out["phase_logits"].shape == (2, 2, 6)
    assert out["progress"].shape == (2, 2, 2)
    assert out["lit_logits"].shape == (2, 2, L_MAX)
    assert out["ledger_logits"].shape == (2, L_MAX)
    # task mask: dead skills must be -inf-ish
    dead = ~b["stage_mask"][0]
    assert out["stage_logits"][0, :, dead].max() < -1e3
    # literal mask on attention
    assert out["lit_logits"][:, :, 3:].max() < -1e3
    # backward
    loss = sum(v.float().pow(2).mean() for v in out.values())
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no grad: {missing}"
    ent = m.entropy(out["stage_logits"])
    assert (ent >= 0).all()
    # v2: every head param lands in exactly one optimizer group
    from train import param_groups
    pg = param_groups(m)
    assert sum(len(g["params"]) for g in pg) == len(list(m.parameters()))
    assert len(pg[1]["params"]) > 0, "no head params matched HEAD_PREFIXES"
    print(f"model ok ({n/1e6:.2f}M params)")


def test_soft_targets():
    stage = np.array([0] * 10 + [5] * 10)
    seg = np.array([-1] * 10 + [0] * 10)
    soft = soft_stage_targets(stage, seg, N_STAGES)
    assert np.allclose(soft.sum(1), 1.0)
    assert soft[0, 0] == 1.0 and soft[15, 5] == 1.0
    b = 10                       # transition frame
    assert 0 < soft[b - 1, 5] < soft[b, 5] < soft[b + 1, 5] <= 1.0
    assert np.isclose(soft[b - 1, 0] + soft[b - 1, 5], 1.0)
    print("soft boundary targets ok")


def test_psat_weights():
    from common import PSAT_TAIL_BOOST
    from dataset import TAIL_F, psat_weights_and_base
    led = np.zeros((40, 3), dtype=np.float32)
    led[20:, 0] = 1.0            # literal 0 flips at frame 20
    w, base = psat_weights_and_base(led)
    assert w[20, 0] == PSAT_TAIL_BOOST and w[20 - TAIL_F, 0] == PSAT_TAIL_BOOST
    assert w[0, 0] == 1.0 and (w[:, 1] == 1.0).all()
    assert np.isclose(base[0], 0.5) and base[1] == 0.0
    print("p_sat tail-boost weights ok")


def test_serve():
    import json
    import os
    import tempfile
    from serve import StageEstimator
    d = tempfile.mkdtemp()
    m = StageHead()
    torch.save(m.state_dict(), os.path.join(d, "ckpt.pt"))
    json.dump({"turning_on_radio|1": 4.0}, open(os.path.join(d, "med.json"), "w"))
    est = StageEstimator(os.path.join(d, "ckpt.pt"), os.path.join(d, "med.json"),
                         device="cpu")
    est.set_task("turning_on_radio",
                 [{"predicate": "toggled_on", "target": "radio", "reference": ""}])
    for _ in range(3):
        out = est.update(torch.randn(1, 1369, 768), torch.rand(1, 148, 148),
                         torch.randn(1, PROPRIO_DIM))
    for arm in ("left", "right"):
        assert set(out[arm]) >= {"stage_dist", "stage_id", "stage_name",
                                 "progress", "active_literal_dist"}
        assert 0.0 <= out[arm]["progress"] <= 1.0
        assert abs(sum(out[arm]["stage_dist"]) - 1.0) < 1e-4
        assert abs(sum(out[arm]["active_literal_dist"]) - 1.0) < 1e-4
        assert "z_stage" not in out[arm], "v2: numbers only, no embeddings"
    assert len(out["p_sat"]) == 1 and 0.0 <= out["p_sat"][0] <= 1.0
    assert out["entropy"]["left"] >= 0
    assert out["stage_age_ratio"]["left"] > 0
    print("serve loop ok")


if __name__ == "__main__":
    test_model()
    test_soft_targets()
    test_psat_weights()
    test_serve()
    print("ALL SMOKE TESTS PASSED")
