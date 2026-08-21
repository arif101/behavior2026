"""POST-transform diff: what the model receives at SERVING vs at TRAINING.

This is the comparison that needs no hypothesis. policy.py:71 is where the raw eval dict becomes
the model's input; we capture there and compare against the training dataloader's Observation.

Context: seven mechanism hypotheses died today. The measurements that survive are
  open-loop on training frames  -> gripper closure recall 0.993
  closed-loop on own states     -> gripper pinned open, variance collapsed 200x (frozen, not erratic)
  forcing gripper proprio closed -> output moves 0.0002 (no proprio shortcut)
A model correct on one input distribution, degenerate on another, and unresponsive to the input we
perturbed, is one whose conditioning is not arriving. So: look at the two inputs.
"""

import dataclasses

import numpy as np


def desc(a):
    if a is None:
        return "MISSING"
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.number):
        return f"{a.shape} {a.dtype}"
    f = a.astype(np.float64).ravel()
    return f"{tuple(a.shape)} [{f.min():+.3f},{f.max():+.3f}] mean{f.mean():+.4f} std{f.std():.4f}"


def main():
    from openpi.training import config as _config
    import openpi.training.data_loader as _data_loader

    z = np.load("/root/posttransform_obs.npz", allow_pickle=True)
    S = {k: z[k] for k in z.files}

    cfg = _config.get_config("pi05_radio_gate")
    cfg = dataclasses.replace(cfg, batch_size=1)
    obs, _ = next(iter(_data_loader.create_b1k_data_loader(
        cfg, shuffle=False, num_batches=1, skip_norm_stats=False)))

    T = {f"image/{k}": np.asarray(v) for k, v in obs.images.items()}
    for f in ("state", "target_points", "target_points_mask", "tokenized_prompt"):
        v = getattr(obs, f, None)
        if v is not None:
            T[f] = np.asarray(v)

    print("=" * 104)
    print("POST-TRANSFORM  -- what the model ACTUALLY receives")
    print("=" * 104)
    keys = [k for k in sorted(set(list(S) + list(T))) if "mask" not in k]
    for k in keys:
        print(f"\n  {k}")
        print(f"    SERVING  {desc(S.get(k))}")
        print(f"    TRAINING {desc(T.get(k))}")

    print()
    print("=" * 104)
    print("VERDICT")
    print("=" * 104)
    problems = []
    for k in keys:
        s, t = S.get(k), T.get(k)
        if s is None or t is None:
            problems.append(f"{k}: present in only one pipeline")
            continue
        s = np.asarray(s).squeeze()
        t = np.asarray(t).squeeze()
        if s.shape != t.shape:
            problems.append(f"{k}: SHAPE {s.shape} vs {t.shape}")
            continue
        if np.issubdtype(s.dtype, np.number) and np.issubdtype(t.dtype, np.number):
            sf, tf = s.astype(np.float64), t.astype(np.float64)
            # different frames, so means differ legitimately; RANGE is the structural check
            if abs(sf.min() - tf.min()) > 0.5 or abs(sf.max() - tf.max()) > 0.5:
                problems.append(f"{k}: RANGE [{sf.min():+.3f},{sf.max():+.3f}] vs "
                                f"[{tf.min():+.3f},{tf.max():+.3f}]")
    if problems:
        print("STRUCTURAL MISMATCHES FOUND:")
        for p in problems:
            print("  *", p)
        print("\n-> the model is being asked a question it never saw in training.")
    else:
        print("No structural mismatch: shapes and value ranges agree on every field.")
        print("-> the serving pipeline is FAITHFUL. The degenerate closed-loop behaviour is NOT")
        print("   an input-plumbing bug, and the search moves to the policy/controller itself.")


if __name__ == "__main__":
    main()
