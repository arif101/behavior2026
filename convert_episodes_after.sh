#!/bin/bash
# After the segment-clip conversion finishes, convert the 58 relay episodes
# into their own dataset (same converter, lazy frames, memory to itself).
while pgrep -f "convert_clips_t[o]_parquet|convert_after_e[f]" >/dev/null; do sleep 120; done
sleep 30
setsid nohup /root/miniconda3/envs/behavior/bin/python -u /root/convert_clips_to_parquet.py \
  --clips "/root/factory_obs2/rac_*_r*.npz" --out /root/b1k_radio_episodes --overwrite \
  > /root/convert_episodes.log 2>&1 < /dev/null
echo "CONVERT_EPISODES_DONE exit=$?" >> /root/convert_episodes.log
