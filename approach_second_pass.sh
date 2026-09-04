#!/bin/bash
# After the first night pass completes, run one more pass: the driver retries every demo
# that has no rac_<D>_200.npz (e.g. d40 rejected under the old honesty metric; transient fails)
# with the corrected metric and the 360-iteration orient budget.
until grep -aq "APPROACH_NIGHT_COMPLETE" /root/approach_runs.log; do sleep 120; done
sleep 30
echo "=== APPROACH SECOND PASS $(date -u +%H:%M:%S) ===" >> /root/approach_runs.log
export APPROACH_PASS=2
exec /root/approach_night.sh
