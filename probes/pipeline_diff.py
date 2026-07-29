"""Diff what the SERVING path hands the policy against what TRAINING hands the model.

No theory required: same semantic fields, two pipelines, compare dtype / shape / range / moments.
Seven mechanism hypotheses died today; this one just looks.

Reads /root/serving_obs.npz (captured live from the eval harness by patch_obs_capture.py) and a
batch from create_b1k_data_loader (the exact loader train_b1k.py uses).

What a mismatch would look like, and what each implies:
  * value range [0,255] vs [-1,1] or [0,1]  -> images arrive unnormalised; the VLM sees noise
  * uint8 vs float32                        -> same
  * different spatial size                  -> resize happens in the wrong place / not at all
  * channel count 4 vs 3                    -> RGBA leaking through (OmniGibson renders RGBA!)
  * per-channel means transposed            -> BGR/RGB swap
  * state dim or scale differing            -> proprio normalisation not applied at serve
Any of these produce a policy that is correct on training observations and degenerate on served
ones -- which is exactly what we measured (0.993 open-loop vs 0.000 closed-loop, with variance
collapsed 200x rather than erratic).
"""

from __future__ import annotations

import argparse

import numpy as np


def describe(name, a):
    a = np.asarray(a)
    s = f"  {name:<46} {str(a.dtype):<9} {str(a.shape):<22}"
    if a.size and np.issubdtype(a.dtype, np.number):
        f = a.astype(np.float64).ravel()
        s += f" min {f.min():>9.4f} max {f.max():>9.4f} mean {f.mean():>9.4f} std {f.std():>8.4f}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serving", default="/root/serving_obs.npz")
    ap.add_argument("--config", default="pi05_radio_gate")
    a = ap.parse_args()

    print("=" * 100)
    print("SERVING PATH (captured live from the eval harness)")
    print("=" * 100)
    z = np.load(a.serving, allow_pickle=True)
    serve = {k: z[k] for k in z.files}
    for k in sorted(serve):
        print(describe(k, serve[k]))

    print()
    print("=" * 100)
    print("TRAINING PATH (create_b1k_data_loader -- what train_b1k.py feeds the model)")
    print("=" * 100)

    import dataclasses

    from openpi.training import config as _config
    import openpi.training.data_loader as _data_loader

    cfg = _config.get_config(a.config)
    cfg = dataclasses.replace(cfg, batch_size=1)
    loader = _data_loader.create_b1k_data_loader(cfg, shuffle=False, num_batches=1, skip_norm_stats=False)
    obs, act = next(iter(loader))

    for k, v in obs.images.items():
        print(describe(f"images/{k}", v))
    for fld in ("state", "target_points", "target_points_mask", "tokenized_prompt"):
        v = getattr(obs, fld, None)
        if v is not None:
            print(describe(fld, v))
    print(describe("actions", act))

    print()
    print("=" * 100)
    print("THE COMPARISON THAT MATTERS")
    print("=" * 100)
    # serving RGB streams vs training images
    srgb = {k: v for k, v in serve.items() if k.endswith("::rgb")}
    timg = dict(obs.images)
    print(f"serving RGB streams : {len(srgb)}   training image streams : {len(timg)}")
    for k, v in sorted(srgb.items()):
        v = np.asarray(v)
        print(f"  SERVE {k.split('::')[-2][:28]:<30} shape {str(v.shape):<20} dtype {v.dtype} "
              f"range [{v.min():.3f}, {v.max():.3f}]")
    for k, v in sorted(timg.items()):
        v = np.asarray(v)
        print(f"  TRAIN {k[:28]:<30} shape {str(v.shape):<20} dtype {v.dtype} "
              f"range [{v.min():.3f}, {v.max():.3f}]")

    sp = np.asarray(serve.get("robot_r1::proprio"))
    ts = np.asarray(obs.state)
    print()
    print(f"  SERVE proprio  shape {sp.shape}  range [{sp.min():.4f}, {sp.max():.4f}]  mean {sp.mean():+.4f}")
    print(f"  TRAIN state    shape {ts.shape}  range [{ts.min():.4f}, {ts.max():.4f}]  mean {ts.mean():+.4f}")

    print()
    print("NOTE: serving values here are PRE-transform (as received by the server); training values")
    print("are POST-transform (as the model sees them). A raw-vs-normalised difference is EXPECTED.")
    print("What is NOT expected: different channel counts, spatial sizes, or a serving stream that")
    print("is missing entirely on the training side (or vice versa).")


if __name__ == "__main__":
    main()
