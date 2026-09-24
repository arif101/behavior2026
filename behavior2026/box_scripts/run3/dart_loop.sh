#!/bin/bash
# Run DART rounds back to back (each round: convert+push the pending renders, then 14 new attempts) until no attempt is
# left or 6 rounds have run. Usage: START=3 setsid nohup bash /root/dart_loop.sh > /root/dart_loop.out 2>&1 &
set -u; START=${START:-3}
for r in $(seq $START $((START+5))); do
  echo "[dart_loop $(date -u +%m-%dT%H:%M:%S)] round $r"
  ROUND=$r bash /root/dart_round.sh > /root/dart_round$r.out 2>&1
  tail -1 /root/dart_round$r.out
  n_new=$(grep -cE "^d[0-9]+ [a-z0-9]+ [0-9:]+" /root/dart_round$r.out)
  [ "$n_new" -eq 0 ] && { echo "[dart_loop] no attempts left"; break; }
done
echo "[dart_loop $(date -u +%m-%dT%H:%M:%S)] DART_LOOP_DONE strict=$(grep -l "\"honest_strict\": true" /root/factory_clips_dart/*_meta.json | wc -l)"
