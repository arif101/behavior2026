"""Can the model reproduce gripper CLOSURE on training data it has already seen?

WHY THIS IS THE DECISIVE TEST
-----------------------------
Closed loop, the policy emits gripper ~ +1 on 100% of steps (l_grip std 0.0057, r_grip 0.0045)
against a training channel that is binary +-1 with p(open)=0.67 -- derived exactly from
norm_stats: mean 0.34 = 2p-1 -> p=0.67, and std sqrt(1-0.34^2) = 0.940, matching the stored 0.94.
So 33% of training frames close the gripper and the policy never does. That is task-fatal: no
grasp, no press.

Two very different causes, and they demand opposite responses:

  A) The model predicts CLOSED on training frames whose ground truth is closed.
     -> Training worked. The collapse is closed-loop covariate shift: at eval the model sees
        states its training distribution never contained and falls back to the majority class.
        Retraining the same way would NOT fix it; the fix is DAgger/corrective data or RL.

  B) The model predicts OPEN even on training frames whose ground truth is closed.
     -> Training itself failed to represent the minority mode. Retraining AS-IS reproduces the
        collapse. The fix is in the objective/representation for this channel.

This runs open-loop (teacher-forced observations straight from the dataset), so covariate shift
is removed by construction -- which is exactly what makes it discriminate A from B.

Reports, over frames whose TRUE gripper is closed:
  * predicted gripper distribution
  * fraction the model gets on the correct side of 0
  * the same for frames whose true gripper is OPEN, as a control (a model that predicts open
    everywhere scores 100% there and ~0% on closed -- that pattern IS the collapse)
"""

from __future__ import annotations

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/root/ckpt")
    ap.add_argument("--config", default="pi05_radio_gate")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    # The config carries the TRAINING batch size (15), which OOMs a 24GB card during sampling:
    # sample_actions runs the full flow-matching integration over a 32-step chunk with 3 camera
    # streams. We only need enough frames to estimate a per-channel distribution, so sample small
    # batches many times instead.
    ap.add_argument("--bs", type=int, default=2)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp

    from openpi.training import config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.models.model as _model

    import dataclasses

    cfg = _config.get_config(a.config)
    cfg = dataclasses.replace(cfg, batch_size=a.bs)
    model = cfg.model.load(_model.restore_params(f"{a.ckpt}/params", dtype=jnp.bfloat16))

    loader = _data_loader.create_b1k_data_loader(
        cfg, shuffle=True, num_batches=a.batches, skip_norm_stats=False
    )
    rng = jax.random.key(a.seed)

    # b1k.py action layout: base 0:3, torso 3:7, left_arm 7:14, l_grip 14, right_arm 15:22, r_grip 22
    GRIP = [14, 22]

    true_all, pred_all = [], []
    for obs, act in loader:
        rng, k = jax.random.split(rng)
        pred = model.sample_actions(k, obs)
        pred = np.asarray(pred, dtype=np.float32)
        true = np.asarray(act, dtype=np.float32)
        # (batch, horizon, dim) -> flatten the horizon; every predicted step counts
        pred_all.append(pred.reshape(-1, pred.shape[-1])[:, GRIP])
        true_all.append(true.reshape(-1, true.shape[-1])[:, GRIP])

    P = np.concatenate(pred_all)
    T = np.concatenate(true_all)
    print(f"samples {len(P)}  (batches {a.batches})")

    for gi, name in enumerate(("l_grip", "r_grip")):
        t, p = T[:, gi], P[:, gi]
        closed = t < 0
        opened = ~closed
        print(f"\n=== {name} ===")
        print(f"  TRUE  closed fraction {closed.mean():.3f}   (norm_stats implies ~0.33)")
        print(f"  PRED  overall  mean {p.mean():+.4f}  std {p.std():.4f}  "
              f"min {p.min():+.4f}  max {p.max():+.4f}")
        if closed.any():
            pc = p[closed]
            print(f"  on TRUE-CLOSED frames: pred mean {pc.mean():+.4f}  std {pc.std():.4f}  "
                  f"frac predicted closed {(pc < 0).mean():.3f}")
        if opened.any():
            po = p[opened]
            print(f"  on TRUE-OPEN   frames: pred mean {po.mean():+.4f}  std {po.std():.4f}  "
                  f"frac predicted open   {(po > 0).mean():.3f}")

    closed_mask = T < 0
    if closed_mask.any():
        acc_closed = (P[closed_mask] < 0).mean()
        print()
        print(f"OVERALL recall on closed frames: {acc_closed:.3f}")
        if acc_closed > 0.5:
            print("VERDICT (A): the model DOES reproduce closure open-loop on training data.")
            print("  -> training is fine; the closed-loop collapse is COVARIATE SHIFT.")
            print("  -> retraining the same way will NOT fix it. Corrective/DAgger data or RL.")
        elif acc_closed < 0.2:
            print("VERDICT (B): the model predicts OPEN even with training observations.")
            print("  -> TRAINING ITSELF failed to capture the minority mode.")
            print("  -> retraining AS-IS reproduces the collapse; change the objective for this channel.")
        else:
            print("VERDICT: partial. The mode exists but is under-expressed -- both effects present.")


if __name__ == "__main__":
    main()
