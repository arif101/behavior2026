---
name: setup-sim-box
description: Provision and validate a fresh RT-core GPU box for BEHAVIOR-2026 OmniGibson eval/RL rollouts. Use when the user gives an SSH string for a new sim/rollout box (e.g. "try this box: ssh root@... -p ... -i ..."). Runs a fast acceptance gate FIRST (rejects bad boxes in seconds), then the Vulkan/Isaac fix ladder, bring-up, eval-prep, and smoke tests. NOT for A100/H100 training boxes (those don't run OmniGibson — RT-core only).
---

# Set up a BEHAVIOR-2026 sim/rollout box

Goal: take an SSH string for a fresh GPU box → a validated OmniGibson eval/rollout box, or a
fast REJECT with the reason. OmniGibson needs **RT cores** (4090 / L40S / RTX-PRO / L4), NOT
A100/H100. Everything here is hard-won (see the `isaac_gpu_cloud_playbook` memory). Use
`$SSH` = the given `ssh ... -p PORT -i KEY root@HOST`.

## Step 0 — ACCEPTANCE GATE (run FIRST; reject fast before any setup)

A box passing `nvidia-smi` is NOT enough. Run this and reject on failure:

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "proc-gpus=$(ls /proc/driver/nvidia/gpus 2>/dev/null | wc -l)  dev-nodes=$(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l)"
df -h / | tail -1
which vulkaninfo || (apt-get update -qq && apt-get install -y -qq vulkan-tools >/dev/null 2>&1)
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|deviceType" | head
```

REJECT rules (these cost us 4 boxes this session):
- **`proc-gpus != dev-nodes`** → container slice of a multi-GPU host; Vulkan will refuse even
  though nvidia-smi/CUDA work. REJECT unless you rent the WHOLE host (all GPUs → all nodes).
- GPU is not RT-core class (A100/H100) → REJECT (won't run OmniGibson).
- `< ~200GB` free disk → REJECT (assets + checkpoints need it).

If `vulkaninfo` already shows the real GPU (`deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU`,
the NVIDIA name), skip Step 1. If it shows only `llvmpipe`, go to Step 1.

## Step 1 — Vulkan/Isaac fix ladder (only if vulkaninfo shows llvmpipe)

Apply in order; re-check vulkaninfo after each:
1. **Missing GLVND (most common on this marketplace):** `apt-get install -y libegl1 libgl1
   libglvnd0 libglx0 libopengl0`. NVIDIA's Vulkan ICD dlopens libEGL.so.1 during init; without
   it, dies silently. Confirm cause with `strace -e trace=openat vulkaninfo 2>&1 | grep ENOENT
   | grep .so` (ENOENT on libEGL.so.1). This alone fixes most boxes.
2. If still llvmpipe: check the ICD loads — `python3 -c 'import ctypes;
   ctypes.CDLL("libGLX_nvidia.so.0")'` and diff the container's `libnvidia-*.so.<ver>` against
   the official driver `.run --extract-only` (missing libs like `libnvidia-gpucomp` on a
   read-only FS = REJECT, can't fix).
3. **PI-futex (Isaac boots then aborts at "app ready"):** the runtime denies FUTEX_LOCK_PI
   (probe: `python3 -c 'import ctypes;print(ctypes.CDLL(None).syscall(202, ctypes.create_string_buffer(4), 6, 0, None, None, 0))'`
   → -1 EPERM). Fix = an LD_PRELOAD shim no-opping the PI mutex setters:
   ```
   printf 'int pthread_mutexattr_setprotocol(void*a,int p){return 0;}\nint pthread_mutexattr_setprioceiling(void*a,int c){return 0;}\n' > /root/nopi.c
   apt-get install -y gcc >/dev/null 2>&1; gcc -shared -fPIC -o /root/nopi.so /root/nopi.c
   ```
   Then run ALL Isaac/OmniGibson with `LD_PRELOAD=/root/nopi.so`.

If after this ladder vulkaninfo still shows only llvmpipe → REJECT the box.

## Step 2 — HF token + scripts

- Stage the HF token to `/root/.hf_token` (chmod 600). NEVER inline the token in a shell
  command. Token needs ROTATION (see memory) — use the current one from the user/HF file-based.
- Pull the canonical scripts from HF `arif101/behavior2026-artifacts`:
  `g3_pipeline/bringup.sh` and `g3_pipeline/g3_evalprep.sh` (also in this repo's root +
  behavior2026/). They carry every network/quota fix (IPv4, CloudFront wheels, aria2 assets,
  MooseFS mitigations, appdata symlink).

## Step 3 — Bring-up (BEHAVIOR-1K + OmniGibson)

Run `bringup.sh` in the background (`setsid nohup ... &`), poll for `STAGE_*_OK` markers.
Gotchas learned this session:
- The bring-up's own `setup.sh` downloads assets to `/tmp/.../huggingface/download/*.incomplete`
  then moves them in — a large negative `du` delta is finalize cleanup, not a failure.
- Its stage-7 (G2-era full-repo HF restore) is OBSOLETE and races the eval-prep — kill the
  `bringup.sh` process tree once assets are populated; use the targeted `g3_evalprep.sh` instead.
- First OmniGibson boot compiles shaders (~4.5 min, silent, log frozen at "app ready", CPU
  pinned) — do NOT kill it. Warm boots ~1-2 min.

## Step 4 — Eval-prep (our fork + checkpoints)

Run `g3_evalprep.sh` (replaces bringup stages 6-8): installs OUR openpi fork
(`code/openpi_fork_adaln_src.tar.gz`, has the AdaLN point conditioning), pulls the needed G3/
Phase-A checkpoints + norm stats + grounding + labels, and ends with a serve smoke
(`STAGE_SERVE_SMOKE_OK (32, 23)`).

## Step 5 — Smoke tests (confirm before real rollouts)

1. **OmniGibson-on-GPU smoke:** load a scene + robot, step physics, render one RGB frame,
   `os._exit(0)` (og.shutdown() HANGS in these containers). Env:
   `LD_PRELOAD=/root/nopi.so OMNIGIBSON_HEADLESS=1 CUDA_VISIBLE_DEVICES=<gpu>`.
   Obs nesting: `obs[robot_name][robot_name:eyes:Camera:0][rgb]` (or the eval kit's
   `robot_r1::robot_r1:<cam>:Camera:0::rgb`).
2. **Serve + eval integration gotchas** (cost several attempts this session):
   - Provider ships **nginx on port 8000/8001** → serve on a clean high port (e.g. 8901);
     the eval "health check" can answer from nginx, masking the real server.
   - Robot config `name` must be **`robot_r1`** (eval kit), and camera obs_keys are
     **double-prefixed**: `robot_r1::robot_r1:<link>:Camera:0::rgb`. If unsure, add a one-shot
     obs-key dump in the serve handler and read the actual keys.
   - Eval entry: `python -m omnigibson.eval.eval --task-name <t> --port <p> --mode public_test
     --instance-indices ... ` (mode `train` for training layouts). `--write-video` for footage.

## Notes
- Cost-conscious: spin the box DOWN when idle (rental burn). Everything is on HF; recreation is
  this skill again (~1-2h with warm knowledge).
- If a box fails the gate at Step 0/1, don't thrash — say why and ask for a different box
  (prefer whole-host rentals or AWS g6e/g5, GCP g2/L4, Paperspace = real VMs by construction).
- After success, back up any new box-only artifacts to HF before spin-down.
