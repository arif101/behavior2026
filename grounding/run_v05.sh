#!/bin/bash
# v0.5 multi-task grounding: full run (sequential on the single A40).
set -x
cd /root/grounding
export HF_HUB_DISABLE_XET=1
OUT=/root/grounding_v05
mkdir -p $OUT/grids

# ---- v2 (DINOv2 ViT-B/14) ----
python3.11 mt_train.py --backbone dinov2_vitb14 --steps 30000 --bs 48 \
  --out $OUT/ckpt_v05_dinov2.pt > $OUT/train_v2.log 2>&1
python3.11 mt_eval.py --ckpt $OUT/ckpt_v05_dinov2.pt \
  --out $OUT/results_v05_dinov2.json --grids-dir $OUT/grids \
  --grid-heldout-eps turning_on_radio sorting_vegetables hiding_Easter_eggs \
  --grid-heldout-tasks putting_dishes_away_after_cleaning loading_the_car \
  > $OUT/eval_v2.log 2>&1

# ---- v3 (DINOv3 ViT-B/16) ----
python3.11 mt_train.py --backbone dinov3_vitb16 --steps 30000 --bs 48 \
  --out $OUT/ckpt_v05_dinov3.pt > $OUT/train_v3.log 2>&1
python3.11 mt_eval.py --ckpt $OUT/ckpt_v05_dinov3.pt \
  --out $OUT/results_v05_dinov3.json --grids-dir $OUT/grids \
  --grid-heldout-eps turning_on_radio sorting_vegetables hiding_Easter_eggs \
  --grid-heldout-tasks putting_dishes_away_after_cleaning loading_the_car \
  > $OUT/eval_v3.log 2>&1

# ---- single-task radio control (interference denominator) ----
python3.11 mt_train.py --backbone dinov2_vitb14 --steps 4000 --bs 48 \
  --only-tasks turning_on_radio \
  --out $OUT/ckpt_v05_radio_control.pt > $OUT/train_control.log 2>&1
python3.11 mt_eval.py --ckpt $OUT/ckpt_v05_radio_control.pt \
  --out $OUT/results_v05_radio_control.json \
  > $OUT/eval_control.log 2>&1

echo ALL_DONE > $OUT/RUN_DONE
