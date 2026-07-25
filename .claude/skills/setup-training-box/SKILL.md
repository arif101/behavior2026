---
name: setup-training-box
description: Provision a fresh A100/H100-class box for BEHAVIOR-2026 π0.5 TRAINING (BC fine-tune, distillation, RL trainer). Use when the user gives an SSH string for a training/data box. Covers openpi install, dataset assembly + norm stats, config/weight-loader patches, GPU smoke, and launch with the DISK-SAFE driver. NOT for OmniGibson sim/rollout boxes (those need RT cores + Vulkan — use /setup-sim-box). Training boxes do NOT render, so no Vulkan/Isaac; the killers here are silent cgroup-OOM and disk-full-at-checkpoint.
---

# Set up a BEHAVIOR-2026 training box

Goal: SSH string → a validated box running (or ready to run) a π0.5 fine-tune / distillation /
RL trainer. Training does NOT run OmniGibson, so A100/H100 are fine (no RT cores, no Vulkan).
The two things that WILL silently kill you here are **container cgroup-OOM** and
**disk-full at checkpoint save** — both cost us runs this session. `$SSH` = the given ssh line.

## Step 0 — ACCEPTANCE GATE (run first)

```
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
df -h / | tail -1
free -g | head -2 | tail -1
# THE SILENT KILLER — real memory cap is the cgroup limit, NOT what `free` shows:
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || cat /sys/fs/cgroup/memory.max 2>/dev/null
nproc
```

REJECT / plan rules:
- **Disk: need ~200GB+ free.** A single full π0.5 checkpoint is **~43GB** (12GB params + 31GB
  train_state). Orbax keeps the previous rolling checkpoint WHILE writing the next → transient
  peak ≈ 3× a checkpoint. Budget accordingly or the run dies at the final save (see Step 6).
- **RAM: check the CGROUP cap, not `free`.** Empty training log + no traceback + GPU idle =
  container cgroup-OOM (the host may show 500GB free while the cgroup caps you at ~46GB). If the
  cap is low, the DATA LOADER must two-pass preallocate-and-fill (never list+concat+index — that
  peaks at ~48GB for our data; two-pass ≈ 17GB). See the `container_cgroup_oom` memory.
- GPU: A100/H100/L40S all fine for training. 2 GPUs is the practical minimum for π0.5 at batch
  32–64 (~2s/step).

## Step 1 — openpi install

```
# our fork (AdaLN point conditioning): pull code/openpi_fork_adaln_src.tar.gz from HF, or use
# an existing /root/openpi. Then:
cd /root/openpi && uv sync            # background it; ~10-20 min
apt-get install -y ffmpeg             # REQUIRED — torchcodec/libtorchcodec video decode fails
                                      # without FFmpeg (libavutil.so.* not found)
```
Verify: `.venv/bin/python -c "import openpi; print('ok')"` and that the G3/Phase-A configs
import (`import openpi.training.config as c; c.get_config('<name>')`).

## Step 2 — dataset + norm stats

Two options:
- **Pull a pre-assembled dataset** if one exists on HF, OR
- **Assemble on-box** with the lean assembler (`g3_pipeline/g3_assemble.py` on HF): partial-
  mirror fetch (only the files the mapped episodes live in — do NOT copy the 20k source),
  action-fp label mapping, `add_target_points`, merge. For Phase A: loop the 24-task set
  (`data_assembly/phaseA_taskset.json`) at 200 eps. VALIDATE: action-fp mapping 0-collision,
  target_points coverage, frame counts (see `g3_conversion_bugs` memory — 3 stacked silent-
  corruption bugs; contiguous episode_index, length collisions, multi-demo hdf5).

Norm stats — the full frame set is huge (600k+), so SAMPLE:
```
.venv/bin/python scripts/compute_norm_stats.py --config-name <cfg> --max-frames 30000
# copy the resulting outputs/assets/<cfg>/ to the other configs that share the dataset
```

**`RepositoryNotFoundError` / 401 during a LOCAL dataset load is a LIAR.** LeRobot's
`LeRobotDataset.__init__` calls `self.reader.try_load()`, and on *any* failure silently falls
back to downloading from the Hub — so every local-integrity problem surfaces as an auth error
against a repo that was never meant to exist. Do not chase the token. Check, in order:

1. **Every column written must be registered in `meta/info.json` `features`** — a column present
   in the parquet but absent from `features` (e.g. an added `sample_weight`) throws
   `DatasetGenerationError` first, then the 401.
2. **Every feature declared must actually exist** — the reverse: declared depth-video streams
   with no files on disk fail the same way.
3. **Every episode must have ALL its video files.** This is the one that hides longest. A
   partial mirror fetches *data* parquets and *video* chunks separately, and a parquet chunk
   can carry "bonus" episodes whose video chunks were never fetched. They look like free extra
   data (I had 2041 episodes where 1800 were planned) but `_check_cached_episodes_sufficient`
   rejects the whole load. Diagnose and filter with:
```python
# per episode, confirm a file exists for every video key; keep only complete ones
ok = [e for e in eps if all((root/meta.get_video_file_path(e,k)).exists() for k in meta.video_keys)]
```
   Then pass `ok` as `dataset_kwargs["episodes"]`. Verify the fix by loading and printing
   `num_episodes`/`num_frames` — a successful local load never touches the network.

**Patching the data loader: there are TWO loader paths, and b1k uses the less obvious one.**
`train_b1k.py` calls `create_b1k_data_loader()`, which builds `TorchDataLoader` directly —
NOT `create_torch_data_loader()`. Patching the latter produces no error, no log line, and a
silently unmodified loader. Any sampler/weighting change must go in `create_b1k_data_loader`,
and must **raise** rather than fall back, or an A/B arm can be labelled "weighted" while
actually training uniform. Verify with a log line printed from inside the loader, and treat
its absence as a failed launch.

Also: `torch.utils.data.WeightedRandomSampler` calls `torch.multinomial`, which **caps at
2**24 = 16,777,216 categories**. Any frame-level weighting over a bigger dataset needs a
custom sampler (for two-valued weights, sample a pool by Bernoulli then an index within it —
exact, and no cap). See `data_assembly/_patch_contact_sampler.py`.

## Step 3 — config + weight-loader patches

- Fill the dataset_root placeholder in the config; wire the episodes list (partial mirror →
  `dataset_kwargs={"episodes": [...]}`).
- **Point-conditioning weight-loader fix (critical):** warm-starting from pi05_base, the AdaLN
  point params (`point_mlp_in/out`, `point_null_embed`) don't exist in the base checkpoint and
  will crash the pytree-equality check. Make `CheckpointWeightLoader.missing_regex` configurable
  and set the point arm to `".*lora.*|.*point_.*"` so those fresh zero-init params are kept.
  (Patch script: `g3_pipeline/patch_point_loader.py` on HF.)
- Pull the base weights: `pi05_base` from `gs://openpi-assets/checkpoints/pi05_base/params`.

## Step 4 — GPU smoke (before the real run)

20-step train of the target config, batch small, save at the end — confirms weight-merge → JIT
→ steps → checkpoint write all work:
```
CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
.venv/bin/python scripts/b1k/train_b1k.py <cfg> --exp_name=smoke --overwrite \
  --num_train_steps=20 --save_interval=1000 --no-wandb-enabled
```
NOTE: the repo's `train_b1k.sh` hardcodes the upstream author's venv path — drive
`train_b1k.py` directly. Expect a 12.5GB base-checkpoint restore + JIT before steps start.

## Step 5 — launch with the DISK-SAFE driver

Do NOT use a naive `set -e` sequential driver (it dies at the disk-full checkpoint save and
silently skips remaining arms). Use the disk-safe pattern (`g3_pipeline/g3_train_v2.sh` on HF):
- **Sidecar pruner:** every ~10 min, strip `train_state/` from all but the newest committed
  checkpoint (keeps footprint ≈ 1 full + N params-only, not 3× full).
- **keep_period=10000** (permanents at 10k/20k/30k, params-only after prune).
- **finalize_arm:** on each arm's completion, upload params to HF and slim the local copy
  before the next arm starts (also closes the no-off-box-checkpoints gap).
- Resume support: `--resume` from the last committed step if a run dies.

Monitor for: `elapsed_steps`/`Progress`, and the failure signatures `Traceback|
RESOURCE_EXHAUSTED|No space|Killed|OOM|CUDA_ERROR`. RESOURCE_EXHAUSTED at a `*.orbax-checkpoint-
tmp-*` path = disk-full at save → the sidecar/keep_period wasn't aggressive enough.

## Notes
- HF token → `/root/.hf_token` (chmod 600); NEVER inline in shell; token needs ROTATION.
- Back up checkpoints to HF as each arm finishes (finalize_arm does this) — never leave a
  trained arm only on-box.
- Cost-conscious: spin DOWN when the run + backup complete; keep up only as retrain insurance
  during an active eval cycle.
- Silent train death diagnosis order: empty log + GPU idle → cgroup-OOM (Step 0 cap); log stops
  at a checkpoint path → disk-full (Step 5); JIT hang → normal first-compile, wait.
