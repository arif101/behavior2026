"""Add the `pi05_radio_gate` TrainConfig — the single-task gate run.

Applied as a patch script rather than an edit because /root/openpi on a box is NOT a git checkout;
editing in place there loses work when the box dies (it already cost us once). Run this on the box
after staging the fork.

What the gate run changes vs pi05_g3_lang_point, and why each is exactly one variable:

1. extra_delta_transform=False  -> ABSOLUTE joint targets.
   Three independent lines agree:
     * Comet (arXiv 2512.10071) Table 4, measured ON turning_on_radio: Delta Joint 0.00 vs
       Absolute Joint 0.30. The single most direct measurement that exists for our setup.
     * Physical Intelligence's own pi05_libero sets extra_delta_transform=False. Only the older
       pi0 configs use delta; ours inherited a pi0-era default onto a pi0.5 model.
     * Our own traces: base motion achieves only ~65% of what is commanded (42-52% gap), the
       open-loop-chunk failure delta encoding produces — MappedDeltaActions subtracts the state at
       CHUNK START from all 32 actions, so an under-delivered step 1 corrupts steps 2..32.

2. repo_id / dataset_root -> the 200-episode radio set with ~100% point coverage.
   Phase A carried real points on 2.67% of frames; the action-expert probe reads R2 0.369 for
   decoding the point but 0.046 for decoding the future action. Present, never converted to intent.

NOT changed, deliberately — the gate tests one hypothesis, not four:
  * camera resolution stays 240x240. Comet measured head720/wrist480 at 0.60 vs 224 at 0.30, but
    it is a real VRAM/throughput cost and would confound this run.
  * action_horizon stays 32 (already the measured optimum: 8->0.00, 16->0.10, 32->0.30, 50->0.25).
  * control rate is already 30 Hz (eval_utils.py:177 action_frequency: 30) — the good side.
  * no depth in the trunk (Comet: RGB 0.30, RGB+Depth 0.20 — depth HURT).
  * point stays on the ACTION EXPERT via AdaLN, not the prefix. prefix_probe.py will tell us
    whether the KV cache is target-blind; until it does, moving the injection site would add a
    variable to the one experiment we have been trying to get clean.

Sizing: 430,128 frames. At batch 32 x 4 GPUs = 128 effective, one epoch is ~3,360 steps. The
default here is 15 epochs (~50k steps) because 200 demos need far more passes than the ~2 epochs
Larchenko ran over 10,000 demos to get comparable gradient exposure — and at this scale the run is
hours, not days, so depth is cheap.
"""

from __future__ import annotations

import argparse
import ast
import pathlib

BLOCK = '''    TrainConfig(
        # GATE RUN: can this architecture reach nonzero q on ONE task?
        # turning_on_radio, 200 episodes, ~100% point coverage, ABSOLUTE joint actions.
        # Comet measured plain pi0.5 at 0.30-0.60 on this exact task, so nonzero is reachable.
        name="pi05_radio_gate",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/root/b1k_radio",
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},
            ),
            robot_config_name="b1k/R1Pro",
            # ABSOLUTE joint targets. Delta measured 0.00 vs absolute 0.30 on this task, PI's own
            # pi05_libero uses absolute, and our traces show ~65% command delivery under delta.
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*lora.*|.*point_.*",  # fresh zero-init AdaLN point params
        ),
        batch_size=32,
        num_train_steps=50_000,
        save_interval=3_360,   # ~1 epoch at 430k frames / (32 x 4 GPUs)
        log_interval=100,
        num_workers=8,
        exp_name="radio_gate",
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/openpi/src/openpi/training/config.py")
    a = ap.parse_args()

    p = pathlib.Path(a.config)
    src = p.read_text()
    if "pi05_radio_gate" in src:
        print("pi05_radio_gate already present — nothing to do")
        return

    anchor = '        name="pi05_g3_lang_point",'
    if anchor not in src:
        raise SystemExit(f"anchor not found in {p}; config layout changed")
    start = src.rindex("    TrainConfig(", 0, src.index(anchor))
    out = src[:start] + BLOCK + src[start:]

    ast.parse(out)          # never write a config that will not import
    p.write_text(out)
    print(f"inserted pi05_radio_gate into {p}")
    print("  absolute joint actions : extra_delta_transform=False")
    print("  dataset                : /root/b1k_radio")
    print("  save_interval          : 3360 steps (~1 epoch)")


if __name__ == "__main__":
    main()
