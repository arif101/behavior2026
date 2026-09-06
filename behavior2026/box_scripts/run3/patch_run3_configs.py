"""Patch the openpi fork with the RUN-3 pre-registered DATA-ARM TrainConfigs
(pi05_radio_run3_a0 .. a5). Anchored, assertion-guarded (the patch_*.py convention).
Apply AFTER the five Run-2 patches (patch_stage_head / map_adaln / depth_aux /
modality_dropout / run2_config) on a tree that already carries the run-1 snapshot +
the Run-3 data_loader.py (sample_weight column).

Single-variable discipline: every arm shares model flags, optimizer, LR schedule, seed,
batch, steps, warm start (Run-2 final params) and norm stats (Run-2's). Arms differ
ONLY in (a) dataset_root — which manufactured sources are merged in — and (b) the launch
env B1K_SAMPLE_WEIGHT_COL (unset for a0; =sample_weight for a1..a5).

Usage:  FORK_ROOT=/path/to/src/openpi RUN3_STEPS=15000 python3 patch_run3_configs.py
"""
import os
import py_compile

FORK = os.environ.get("FORK_ROOT", "/root/openpi_fork/src/openpi")
STEPS = int(os.environ.get("RUN3_STEPS", "15000"))
WARMUP = int(os.environ.get("RUN3_WARMUP", "500"))
SAVE = int(os.environ.get("RUN3_SAVE", "2500"))
MARKER = "patch_run3_configs"


def patch(path, subs):
    s = open(path).read()
    if MARKER in s:
        print(f"SKIP {path}: already patched")
        return
    for pat, rep in subs:
        assert pat in s, f"anchor not found in {path}: {pat[:70]!r}"
        assert s.count(pat) == 1, f"anchor not unique in {path}: {pat[:70]!r}"
        s = s.replace(pat, rep, 1)
    compile(s, path, "exec")
    open(path, "w").write(s)
    py_compile.compile(path, doraise=True)
    print(f"patched {path}")


HELPER_ANCHOR = "_CONFIGS = [\n"
HELPER = f'''
# ---- RUN 3 ({MARKER}.py): pre-registered DATA arms -------------------------------------
# Identical model / optimizer / schedule / seed / batch / steps; all warm-started from the
# Run-2 final params (arif101/b26-run2-params @49999, every module present => missing_regex
# admits nothing but lora). Norm stats = Run-2's, copied per arm (NOT recomputed per mix, so
# input normalisation is identical across arms and matches the init checkpoint).
# Launch env: B1K_STAGE_OVERSAMPLE=8 for every arm (Run-2 setting);
#             B1K_SAMPLE_WEIGHT_COL=sample_weight for a1..a5, UNSET for a0.
_RUN3_STEPS = {STEPS}
_RUN3_WARMUP = {WARMUP}
_RUN3_SAVE = {SAVE}
_RUN3_WARMSTART = "/root/warmstart_run3/params"
_RUN3_ARMS = {{
    "a0": "/root/b1k_radio_mix_a1",  # map only, NO poison down-weight (control / floor)
    "a1": "/root/b1k_radio_mix_a1",  # map only + poison down-weight (sample_weight col)
    "a2": "/root/b1k_radio_mix_a2",  # a1 + b1k_radio_factory   (honest grasp+transport)
    "a3": "/root/b1k_radio_mix_a3",  # a1 + b1k_radio_episodes  (honest grasp + learned press)
    "a4": "/root/b1k_radio_mix_a4",  # a1 + factory + episodes  (full mix)
    "a5": "/root/b1k_radio_mix_a5",  # a4 + b1k_radio_approach  (data lands later)
}}


def _run3_cfg(arm: str, dataset_root: str) -> "TrainConfig":
    return TrainConfig(
        name=f"pi05_radio_run3_{{arm}}",
        model=pi0_config.Pi0Config(
            action_horizon=32,
            pi05=True,
            point_conditioning=True,
            point_noise_std=0.02,
            map_tokens_k=8,
            anti_shortcut=False,
            modality_dropout_p=0.2,
            map_geo_conditioning=True,
            depth_aux=True,
            stage_head=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_radio",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root=dataset_root,
                prompt_from_task=True,
                dataset_kwargs={{"tolerance_s": 5e-4}},
            ),
            robot_config_name="b1k/R1Pro",
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(_RUN3_WARMSTART, missing_regex=".*lora.*"),
        freeze_filter=nnx_utils.PathRegex(".*map_(proj|registers|alpha|recon).*"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=_RUN3_WARMUP, peak_lr=2.5e-5, decay_steps=_RUN3_STEPS, decay_lr=2.5e-6),
        batch_size=32,
        num_train_steps=_RUN3_STEPS,
        save_interval=_RUN3_SAVE,
        log_interval=100,
        num_workers=8,
        exp_name=f"radio_run3_{{arm}}",
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    )


_CONFIGS = [
'''

RUN2_TAIL = (
    '        exp_name="radio_run2",\n'
    '        assets_base_dir="./outputs/assets",\n'
    '        checkpoint_base_dir="./outputs/checkpoints",\n'
    "    ),\n"
)
RUN3_ENTRIES = RUN2_TAIL + f"    *[_run3_cfg(_a, _r) for _a, _r in _RUN3_ARMS.items()],  # {MARKER}\n"

patch(f"{FORK}/training/config.py", [
    (HELPER_ANCHOR, HELPER),
    (RUN2_TAIL, RUN3_ENTRIES),
])
print(f"RUN-3 CONFIG PATCH APPLIED: steps={STEPS} warmup={WARMUP} save_interval={SAVE}")
