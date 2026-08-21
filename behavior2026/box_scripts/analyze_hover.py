"""Discriminate the run-2 hover: COMMIT failure vs REACH/workspace failure vs TARGETING error.

Same-signature rivals (organizers' ckpt hover was diagnosed REACH, fingertip median 20.5 cm,
never <8 cm — NOT commit). Action-channel signatures (R1Pro 23-dim: base 0:3, torso 3:7,
arms 7:21, grippers 21:23):

  COMMIT failure : grippers never command close during the hover; torso/arm still active
                   (positioning forever); base parked.
  REACH failure  : arm/torso commands saturated (pushing a limit, flat); base parked too far
                   and never repositioning.
  TARGETING error: gripper closes repeatedly at range (committing to a wrong point).

Compares the legal run-2 log against the oracle campaign's SUCCESS run over the same window.
"""

import glob
import json

import numpy as np

BASE, TORSO, ARMS, GRIP = slice(0, 3), slice(3, 7), slice(7, 21), slice(21, 23)


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return np.array([r["full"] for r in rows if "full" in r], np.float64)


def profile(name, A, lo=0.4, hi=1.0):
    """Stats over the [lo, hi] fraction of the episode (the hover/terminal window)."""
    W = A[int(len(A) * lo):int(len(A) * hi)]
    g = W[:, GRIP]
    # gripper command polarity: values near open vs near close (sign convention: read range)
    print(f"\n[{name}] steps={len(A)} window={len(W)}")
    print(f"  gripper L: min {g[:, 0].min():.3f} max {g[:, 0].max():.3f} "
          f"frac<0.5 {np.mean(g[:, 0] < 0.5):.3f}  transitions {np.sum(np.abs(np.diff(g[:, 0] > 0.5)))}")
    print(f"  gripper R: min {g[:, 1].min():.3f} max {g[:, 1].max():.3f} "
          f"frac<0.5 {np.mean(g[:, 1] < 0.5):.3f}  transitions {np.sum(np.abs(np.diff(g[:, 1] > 0.5)))}")
    t = W[:, TORSO]
    print(f"  torso cmd mean {np.round(t.mean(0), 3)}  std {np.round(t.std(0), 4)}")
    b = np.linalg.norm(W[:, BASE], axis=1)
    print(f"  base |cmd| p50 {np.median(b):.4f} p95 {np.percentile(b, 95):.4f} "
          f"frac>0.02 {np.mean(b > 0.02):.3f}")
    a = W[:, ARMS]
    print(f"  arm cmd std (per-dim mean) {a.std(0).mean():.4f}  "
          f"step-to-step |delta| p50 {np.median(np.abs(np.diff(a, axis=0))):.5f}")


import sys

paths = sys.argv[1:] if len(sys.argv) > 1 else [
    "/root/rate_legal/run_2/action_log.jsonl:LEGAL run2 (q=0 hover)",
    "/root/rate_legal/run_3/action_log.jsonl:LEGAL run3 (q=0)",
    "/root/rate_blend/run_5/action_log.jsonl:CONVERTING run (blend q=1.0)",
    "/root/rate/run_1/action_log.jsonl:ORACLE fail run1",
]
for spec in paths:
    path, _, name = spec.partition(":")
    try:
        profile(name or path, load(path))
    except Exception as e:
        print(f"SKIP {path}: {e}")
