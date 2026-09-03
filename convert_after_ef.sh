#!/bin/bash
# Wait for the episode factory night run to release the 50GB cgroup, then
# convert the 38 segment clips (lazy-frame converter) with the memory to itself.
while pgrep -f "relay_episode_factor[y]|ef_nigh[t].sh" >/dev/null; do sleep 120; done
sleep 30
setsid nohup /root/miniconda3/envs/behavior/bin/python -u /root/convert_clips_to_parquet.py \
  --clips "/root/factory_obs2/rac_*_0.npz" --out /root/b1k_radio_factory --overwrite \
  > /root/convert_factory.log 2>&1 < /dev/null
echo "CONVERT_SEQ_DONE exit=$?" >> /root/convert_factory.log
