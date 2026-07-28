#!/bin/bash
# Predicate-eval replay sweep (v4): one OmniGibson playback per manifest
# episode -> /root/predicates/<task>/ep{fi}.json (real p_sat targets) +
# /root/sweep_labels_v4/<task>/labels_{fi}.jsonl (extractor labels w/ AABB
# extents). One process per episode (Isaac state does not survive scene
# swaps); boot mutex serializes startup; teardown segfaults after PRED_DONE
# are benign (replay_labeled contract).
# Prereqs: bringup_stage_v4.sh done; prep_data.py wrote /root/data/manifest.json;
#          goal_literals.py wrote /root/task_literals.json.
# Run detached:  setsid nohup bash run_pred_sweep.sh > /root/pred_sweep.log 2>&1 < /dev/null &
set -uo pipefail
# Provider-image quirk (found 2026-07-22): Mesa Vulkan ICDs collide with the
# NVIDIA GLX ICD and Vulkan silently falls back to llvmpipe -- Isaac won't
# boot. Mesa ICDs are removed at bringup; this pin is belt-and-braces.
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json
# RunPod seccomp EPERMs FUTEX_LOCK_PI -> glibc fatal ("The futex facility
# returned an unexpected error code") the moment PhysX contends a PI mutex.
# nopi.so (built at bringup) no-ops pthread_mutexattr_setprotocol.
export LD_PRELOAD=/root/nopi.so
export PATH=/root/miniconda3/bin:$PATH
PY=/root/miniconda3/envs/behavior/bin/python
SYSPY=/root/miniconda3/envs/behavior/bin/python   # conda env carries hf_hub etc. (24.04 system py has no pip)
MANIFEST="${MANIFEST:-/root/data/manifest.json}"   # override to sweep a subset
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16

# --- 1. fetch the rawdata HDF5s this manifest needs -------------------------
$SYSPY - <<'EOF'
import json, os, re
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download
for e in json.load(open("/root/data/manifest.json")):
    m = re.search(r"task-(\d{4})/episode_(\d+)\.json", e["annot"])
    tid, did = int(m.group(1)), int(m.group(2))
    rel = f"task-{tid:04d}/episode_{did:08d}.hdf5"
    dst = f"/root/rawdata/{rel}"
    if os.path.exists(dst):
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("FETCH", rel, flush=True)
    p = hf_hub_download("behavior-1k/2026-challenge-rawdata", rel,
                        repo_type="dataset", local_dir="/root/rawdata")
print("RAWDATA_OK", flush=True)
EOF
if [ $? -ne 0 ]; then echo RAWDATA_FETCH_FAILED; exit 1; fi

# --- 2. per-episode replay + predicate eval ---------------------------------
# First episode is the CANARY: eval_predicates.py has never met the real API
# surface -- if the canary fails both attempts, abort the sweep for inspection
# instead of burning 40 x 2 x timeout.
cd /root/BEHAVIOR-1K/OmniGibson
first=1
for key in $($SYSPY -c "
import json
for e in json.load(open('$MANIFEST')):
    print(f\"{e['task']}/{e['file_idx']}\")"); do
  task=${key%/*}; fi=${key#*/}
  done_marker=$(printf '/root/predicates/%s/ep%03d.json.done' "$task" "$fi")
  [ -f "$done_marker" ] && { echo "SKIP $key"; continue; }
  for attempt in 1 2; do
    echo "=== $key attempt $attempt $(date +%H:%M:%S) ==="
    timeout 5400 taskset -c 0-23 $PY /root/stage/eval_predicates.py \
        --manifest "$MANIFEST" --only "$key" \
        --rawdata_root /root/rawdata --task_literals /root/task_literals.json \
        --out_dir /root/predicates --labels_dir /root/sweep_labels_v4 \
        --scratch /root/pred_scratch
    [ -f "$done_marker" ] && break
  done
  if [ ! -f "$done_marker" ]; then
    echo "FAILED $key"
    if [ "$first" = 1 ]; then
      echo "CANARY_FAILED -- aborting sweep, inspect the log above"
      exit 1
    fi
  fi
  first=0
  rm -rf /root/pred_scratch/*
done

n_done=$(find /root/predicates -name '*.done' | wc -l)
n_all=$($SYSPY -c "import json; print(len(json.load(open('$MANIFEST'))))")
echo "PRED_SWEEP_DONE $n_done/$n_all"
