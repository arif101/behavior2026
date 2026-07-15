#!/bin/bash
# v2 chain: trash gate i1-4 (our 30k LoRA) -> radio calibration i0-9 (organizers' ckpt).
# Hardened after MooseFS-quota mass-kill: big files on /root, write-test each phase.
set -x
export PATH=/root/miniconda3/bin:/root/.local/bin:$PATH

wtest() { dd if=/dev/zero of=/workspace/.wt bs=1M count=10 oflag=direct 2>/dev/null && rm -f /workspace/.wt || { echo QUOTA_FAIL_$1; exit 1; }; }

serve_up() {  # $1 config $2 ckpt_dir $3 task $4 log
  pgrep -f serve_b1k | xargs -r kill; sleep 10
  cd /workspace/openpi
  nohup env CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro --task b1k/$3 --repo-id $3 \
    --policy.config $1 --policy.dir "$2" \
    --control_mode receding_horizon --action_horizon 16 --port 8000 > "$4" 2>&1 &
  for i in $(seq 1 60); do ss -tln | grep -q 8000 && break; sleep 10; done
  ss -tln | grep -q 8000 || { echo SERVE_FAILED; exit 1; }
  sleep 20; echo SERVE_UP_$3
}

run_eval() {  # $1 task $2 idx $3 tag
  export TRACE_PATH=/workspace/traces_$3_i$2.jsonl
  python -m omnigibson.eval.eval --task-name $1 \
    --robot-config /workspace/robot_openpi.yaml --mode public_test --instance-indices $2 \
    --host 127.0.0.1 --port 8000 --output-dir ./eval_logs/$3_i$2 --write-video \
    > /workspace/eval_$3_i$2.log 2>&1
  echo ${3^^}_INSTANCE_$2_DONE
  grep -E "Result:" /workspace/eval_$3_i$2.log | tail -1
  /root/miniconda3/envs/behavior/bin/python /workspace/probes/analyze_rollout.py \
    --trace $TRACE_PATH --out /workspace/verdict_$3_i$2.json 2>/dev/null | tail -3
}

source /root/miniconda3/etc/profile.d/conda.sh && conda activate behavior; hash -r

# ---------- PHASE A: trash gate, our 30k LoRA, instances 1-4 ----------
wtest A
serve_up pi05_b1k_trash_lora /root/g2_ckpts/pi05_b1k_trash_lora/trash_lora/29999 picking_up_trash /workspace/serve_trash.log
cd /workspace/BEHAVIOR-1K
for IDX in 1 2 3 4; do wtest A$IDX; run_eval picking_up_trash $IDX trash; done
echo TRASH_GATE_DONE

# ---------- PHASE B: calibration, organizers' ckpt, radio 0-9 ----------
while pgrep -f "gdow[n]" > /dev/null; do sleep 30; done
/root/miniconda3/envs/behavior/bin/python - << 'PY' || { echo CALIB_FAIL_BADZIP; exit 1; }
import zipfile
z = zipfile.ZipFile("/root/pi05_radio_baseline.zip")
names = z.namelist()
keep = [n for n in names if "train_state" not in n]
print("zip entries:", len(names), "keeping:", len(keep))
z.extractall("/root/baseline_ckpt", members=keep)
PY
rm -f /root/pi05_radio_baseline.zip
CKPT_DIR=$(dirname $(find /root/baseline_ckpt -maxdepth 6 -type d -name params | head -1))
[ -z "$CKPT_DIR" ] && { echo CALIB_FAIL_NOPARAMS; find /root/baseline_ckpt -maxdepth 3 | head -20; exit 1; }
echo CALIB_CKPT_DIR=$CKPT_DIR
wtest B
serve_up pi05_b1k "$CKPT_DIR" turning_on_radio /workspace/serve_calib.log
cd /workspace/BEHAVIOR-1K
for IDX in 0 1 2 3 4 5 6 7 8 9; do wtest B$IDX; run_eval turning_on_radio $IDX calib; done
pgrep -f serve_b1k | xargs -r kill
echo RADIO_CALIB_DONE
