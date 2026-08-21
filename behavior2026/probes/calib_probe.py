"""Calibrate the grouped ridge estimator before spending GPU on it.

Two things must be true for the probe's numbers to be readable:
  1. NO FALSE POSITIVES: features with targets that are unrelated must read ~0.
  2. KNOWN CEILING: we need to know what R2 a PERFECT linear relationship reports at this
     n and d, so the verdict thresholds mean something.

The first pass used iid Gaussian features (d=2048 > n=768) and a perfect linear map read only
0.35 -- the worst case for ridge, since there is no correlation structure to exploit. Real
transformer activations are strongly correlated and low effective rank, where ridge does much
better. This sweeps effective rank to bracket where our real features sit.
"""

import importlib.util

import numpy as np

spec = importlib.util.spec_from_file_location("g", "/root/prefix_probe_grouped.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

rs = np.random.default_rng(0)
D, K = 2048, 6


def lowrank(n, rank, B, noise=0.1):
    """Low effective rank + noise, standing in for real activations.

    B (the basis mapping latent -> feature space) MUST be shared between the fit and test sets.
    Drawing a fresh B per call puts the two sets in different random subspaces, so a decoder fit
    on one is meaningless on the other and even a perfect linear map reads R2 ~ 0. That was a bug
    in an earlier version of this harness, not in the probe.
    """
    Z = rs.normal(size=(n, rank))
    X = (Z @ B).astype(np.float32)
    X = X + rs.normal(size=(n, D)).astype(np.float32) * noise * X.std()
    return X, Z


print("PERFECT linear signal, by effective rank (the estimator's ceiling):")
for rank in (8, 16, 64, 256, 1024):
    basis = rs.normal(size=(rank, D))          # shared feature basis
    W = rs.normal(size=(rank, K))              # shared latent -> target map
    Xa, Za = lowrank(768, rank, basis)
    Xb, Zb = lowrank(384, rank, basis)
    r = g.cross_episode(Xa, (Za @ W).astype(np.float32), Xb, (Zb @ W).astype(np.float32),
                        np.random.default_rng(1))
    print("  rank {:>5}   R2 {:.3f}   ctrl {:+.3f}".format(rank, r["r2"], r["r2_shuffled_control"]))

print("\nFALSE-POSITIVE test (targets unrelated to features) -- must read ~0:")
for rank in (16, 64, 256):
    basis = rs.normal(size=(rank, D))
    W = rs.normal(size=(rank, K))
    Xa, Za = lowrank(768, rank, basis)
    Xc, _ = lowrank(384, rank, basis)
    r = g.cross_episode(Xa, (Za @ W).astype(np.float32), Xc,
                        rs.normal(size=(384, K)).astype(np.float32), np.random.default_rng(1))
    print("  rank {:>5}   R2 {:+.3f}   signal {:+.3f}".format(rank, r["r2"], r["signal_over_control"]))

print("\nPARTIAL signal (target = frac linear-in-features + rest unrelated), rank 64:")
basis = rs.normal(size=(64, D))
W = rs.normal(size=(64, K))
Xa, Za = lowrank(768, 64, basis)
Xb, Zb = lowrank(384, 64, basis)
for frac in (0.25, 0.5, 0.75, 1.0):
    Ya = (frac * (Za @ W) + (1 - frac) * rs.normal(size=(768, K))).astype(np.float32)
    Yb = (frac * (Zb @ W) + (1 - frac) * rs.normal(size=(384, K))).astype(np.float32)
    r = g.cross_episode(Xa, Ya, Xb, Yb, np.random.default_rng(1))
    print("  frac {:.2f}    R2 {:.3f}   signal {:+.3f}".format(frac, r["r2"], r["signal_over_control"]))
