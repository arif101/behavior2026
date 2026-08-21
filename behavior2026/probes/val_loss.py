"""Held-out loss per checkpoint — does the gate run generalize, or memorize?

Why this matters
----------------
3B params over 180 demos of ONE task is ample capacity to memorize. Each episode is a distinct
radio placement, so the 20 held-out episodes test exactly the instance generalization the point
conditioning is supposed to buy. Two readings, and they answer different questions:

  val(25800) vs val(50000)     falling  -> still learning; 1.94 epochs is genuinely undertrained
                               flat     -> converged; more epochs will not rescue the gate
                               rising   -> memorizing; the gate result is about this dataset, not
                                           the architecture

  train-subset vs val (same ckpt)       the generalization gap in absolute terms

Norm stats are the TRAIN stats (skip_norm_stats=False loads the existing asset). That is correct
and deliberate: normalizing val with val-derived stats would leak.

Uses create_b1k_data_loader for the same reason prefix_probe does — create_data_loader ignores
dataset_root and 401s against the Hub.

Usage:
  python val_loss.py --ckpt /path/to/checkpoint/params --config pi05_radio_gate
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np


def _loader_for(cfg, episodes, n_batches):
    """Same config, different episode slice.

    dataset_kwargs lives on cfg.data.base_config (a DataConfig), NOT on the
    LeRobotB1KDataConfig factory itself — hence the nested replace.
    """
    import openpi.training.data_loader as _data_loader

    base = cfg.data.base_config
    kwargs = dict(base.dataset_kwargs)
    kwargs["episodes"] = list(episodes)
    cfg2 = dataclasses.replace(
        cfg,
        data=dataclasses.replace(
            cfg.data, base_config=dataclasses.replace(base, dataset_kwargs=kwargs)
        ),
    )
    return _data_loader.create_b1k_data_loader(
        cfg2, shuffle=False, num_batches=n_batches, skip_norm_stats=False
    )


def _mean_loss(model, loader, seed):
    import jax

    rng = jax.random.key(seed)
    tot, cnt = 0.0, 0
    for obs, act in loader:
        # Same rng every batch: flow matching samples a noise level per call, so a shared key keeps
        # the comparison between checkpoints and between splits apples-to-apples.
        loss = model.compute_loss(rng, obs, act, train=False)
        loss = np.asarray(loss, dtype=np.float32)
        tot += float(loss.sum())
        cnt += int(loss.size)
    return tot / max(cnt, 1), cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to the checkpoint's params/ dir")
    ap.add_argument("--config", default="pi05_radio_gate")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="val_loss.json")
    a = ap.parse_args()

    import jax.numpy as jnp

    from openpi.training import config as _config
    import openpi.models.model as _model

    cfg = _config.get_config(a.config)
    model = cfg.model.load(_model.restore_params(a.ckpt, dtype=jnp.bfloat16))

    val, n_val = _mean_loss(model, _loader_for(cfg, range(180, 200), a.batches), a.seed)
    # Equal-sized train slice so the two numbers are comparable.
    trn, n_trn = _mean_loss(model, _loader_for(cfg, range(0, 20), a.batches), a.seed)

    res = {
        "ckpt": a.ckpt,
        "val_loss_heldout_ep180_199": round(val, 6),
        "train_loss_ep0_19": round(trn, 6),
        "generalization_gap": round(val - trn, 6),
        "n_val_elems": n_val,
        "n_train_elems": n_trn,
    }
    print(json.dumps(res, indent=1))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
