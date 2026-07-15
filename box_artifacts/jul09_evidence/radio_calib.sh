#!/bin/bash
# CALIBRATION: organizers' own pi05 radio ckpt (claimed ~10% success on Discord)
# across radio public instances 0-9. Anchors our recipe's relative position.
set -x
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH

# --- 1. wait for trash gate to release the GPU ---
while ! grep -q "TRASH_GATE_DONE" /workspace/trash_gate.log 2>/dev/null; do sleep 60; done
echo CALIB_GATE_RELEASED

# --- 2. wait for download, then extract to local disk ---
while pgrep -f "gdow[n]" > /dev/null; do sleep 30; done
python3 -c 'import zipfile; zipfile.ZipFile("/workspace/pi05_radio_baseline.zip")' || { echo CALIB_FAIL_BADZIP; exit 1; }
mkdir -p /workspace/baseline_ckpt
/root/miniconda3/envs/behavior/bin/python -c 'import zipfile; zipfile.ZipFile("/workspace/pi05_radio_baseline.zip").extractall("/workspace/baseline_ckpt")'
rm -f /workspace/pi05_radio_baseline.zip
CKPT_DIR=$(dirname $(find /workspace/baseline_ckpt -maxdepth 5 -type d -name params | head -1))
[ -z "$CKPT_DIR" ] && { echo CALIB_FAIL_NOPARAMS; find /workspace/baseline_ckpt -maxdepth 3 | head -20; exit 1; }
echo CALIB_CKPT_DIR=$CKPT_DIR

# --- 3. serve THEIR ckpt with THEIR verbatim flags (stock descriptive radio prompt) ---
pgrep -f serve_b1k | xargs -r kill; sleep 10
cd /workspace/openpi
nohup uv run scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/turning_on_radio \
  --repo-id turning_on_radio --policy.config pi05_b1k --policy.dir "$CKPT_DIR" \
  --control_mode receding_horizon --action_horizon 16 --port 8000 \
  > /workspace/serve_calib.log 2>&1 &
for i in $(seq 1 60); do grep -q "Uvicorn running\|serving on" /workspace/serve_calib.log && break; sleep 10; done
sleep 20
echo CALIB_SERVER_UP

# --- 4. radio public instances 0-9, trace + battery each ---
source /root/miniconda3/etc/profile.d/conda.sh && conda activate behavior
hash -r
cd /workspace/BEHAVIOR-1K
for IDX in 0 1 2 3 4 5 6 7 8 9; do
  export TRACE_PATH=/workspace/traces_calib_i$IDX.jsonl
  python -m omnigibson.eval.eval --task-name turning_on_radio \
    --robot-config /workspace/robot_openpi.yaml --mode public_test --instance-indices $IDX \
    --host 127.0.0.1 --port 8000 --output-dir ./eval_logs/calib_i$IDX --write-video \
    > /workspace/eval_calib_i$IDX.log 2>&1
  echo CALIB_INSTANCE_${IDX}_DONE
  grep -E "Result:" /workspace/eval_calib_i$IDX.log | tail -1
  /root/miniconda3/envs/behavior/bin/python /workspace/probes/analyze_rollout.py \
    --trace /workspace/traces_calib_i$IDX.jsonl --out /workspace/verdict_calib_i$IDX.json 2>/dev/null | tail -3
done
pgrep -f serve_b1k | xargs -r kill
echo RADIO_CALIB_DONE
