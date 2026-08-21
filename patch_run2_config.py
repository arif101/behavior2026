"""Patch the openpi fork with the RUN-2 TrainConfig (pi05_radio_run2).

Anchored, assertion-guarded edits (the patch_*.py convention). compile() before write,
py_compile after. Apply AFTER patch_stage_head / patch_map_adaln / patch_depth_aux /
patch_modality_dropout (the model flags referenced here exist only post-patch; this
script only guarantees syntax, the flag check happens at config construction).

THE RUN-2 RECIPE (all measurement-determined; RUN2_PATCHES_BUILD.md + RUN2_CONFIG.md):
  data     = /root/b1k_radio_run2mix (assemble_run2_mix.py: b1k_radio_map + corrective
             + rac, schema-exact, all carrying gt_depth_ds)
  model    = run1b arch + stage_head + map_geo AdaLN + depth aux + modality dropout 0.2
             (anti_shortcut OFF — superseded, enable exactly one)
  warmstart= run1b params; missing_regex covers ONLY the new modules (map/aux params
             exist in run1b and restore)
  freeze   = the PREFIX map-token route params (proj/registers/alpha/recon) — G1 verdict:
             consumed-but-unpaid at 90% visibility; its venue is multi-task. Aux heads
             stay trainable EXACTLY as run1b (single-variable-change discipline; the
             alternative — also neutralizing e_map/e_recon — is noted in RUN2_CONFIG.md).
  sampling = B1K_STAGE_OVERSAMPLE=8 env (patch_stage_oversample.py) + source weights
             baked at assembly time.
  PRE-REGISTER the eval protocol BEFORE looking at results (G1 early-separation lesson).

Usage:  FORK_ROOT=/path/to/src/openpi python3 patch_run2_config.py
        (default FORK_ROOT = /root/openpi_fork/src/openpi)
"""

import os
import py_compile

FORK = os.environ.get("FORK_ROOT", "/root/openpi_fork/src/openpi")

MARKER = "patch_run2_config"


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


IMP_ANCHOR = "import openpi.shared.normalize as _normalize\n"
IMP_BLOCK = "import openpi.shared.nnx_utils as nnx_utils  # patch_run2_config: freeze_filter\n"

CFG_ANCHOR = (
    '        exp_name="radio_map",\n'
    '        assets_base_dir="./outputs/assets",\n'
    '        checkpoint_base_dir="./outputs/checkpoints",\n'
    "    ),\n"
)
CFG_BLOCK = (
    "    TrainConfig(\n"
    "        # RUN 2 (patch_run2_config.py): the measured-recipe run. Mix = b1k_radio_map\n"
    "        # + splice-corrective (initiation neighborhood, 0.03 m) + RaC-recovery bank\n"
    "        # (approach-stall recovery on policy-visited states, all far-field >0.35 m —\n"
    "        # measured 2026-08-07), merged schema-exact by assemble_run2_mix.py, all three\n"
    "        # carrying gt_depth_ds. New heads/routes per RUN2_PATCHES_BUILD.md. Prefix\n"
    "        # map-token route FROZEN (G1: consumed-but-unpaid at 90% visibility); the\n"
    "        # map GEOMETRY->AdaLN route is the Run-2 map venue. Oversampling: set\n"
    "        # B1K_STAGE_OVERSAMPLE=8 in the launch env (patch_stage_oversample.py) —\n"
    "        # source-level corrective weighting is baked at assembly. PRE-REGISTER the\n"
    "        # eval protocol BEFORE looking at results (G1 early-separation lesson).\n"
    '        name="pi05_radio_run2",\n'
    "        model=pi0_config.Pi0Config(\n"
    "            action_horizon=32,\n"
    "            pi05=True,\n"
    "            point_conditioning=True,\n"
    "            point_noise_std=0.02,\n"
    "            map_tokens_k=8,\n"
    "            anti_shortcut=False,        # superseded by modality_dropout_p; enable ONE\n"
    "            modality_dropout_p=0.2,\n"
    "            map_geo_conditioning=True,\n"
    "            depth_aux=True,\n"
    "            stage_head=True,\n"
    "        ),\n"
    "        data=LeRobotB1KDataConfig(\n"
    '            repo_id="b1k_radio",\n'
    "            base_config=DataConfig(\n"
    "                data_cls=_lerobot_compat.LeRobotDataset,\n"
    '                dataset_root="/root/b1k_radio_run2mix",\n'
    "                prompt_from_task=True,\n"
    '                dataset_kwargs={"tolerance_s": 5e-4},\n'
    "            ),\n"
    '            robot_config_name="b1k/R1Pro",\n'
    "            # ABSOLUTE joint targets (run-1 measurement: delta 0.00 vs absolute 0.30).\n"
    "            extra_delta_transform=False,\n"
    "        ),\n"
    "        weight_loader=weight_loaders.CheckpointWeightLoader(\n"
    '            "/root/warmstart_run2/params",   # = run1b final params, restored on the trainer\n'
    '            missing_regex=".*lora.*|.*stage_head.*|.*map_geo.*|.*depth_aux.*",\n'
    "        ),\n"
    "        # Freeze ONLY the prefix map-token route (params restore from run1b, then\n"
    "        # never update): input/output projections, registers, ReZero alpha, recon\n"
    "        # anti-sink. Aux decode heads stay trainable exactly as run1b.\n"
    "        freeze_filter=nnx_utils.PathRegex(\".*map_(proj|registers|alpha|recon).*\"),\n"
    "        batch_size=32,\n"
    "        num_train_steps=50_000,\n"
    "        save_interval=3_650,   # ~1 epoch at ~467k mix frames / (32 x 4 GPUs)\n"
    "        log_interval=100,\n"
    "        num_workers=8,\n"
    '        exp_name="radio_run2",\n'
    '        assets_base_dir="./outputs/assets",\n'
    '        checkpoint_base_dir="./outputs/checkpoints",\n'
    "    ),\n"
)

patch(f"{FORK}/training/config.py", [
    (IMP_ANCHOR, IMP_ANCHOR + IMP_BLOCK),
    (CFG_ANCHOR, CFG_ANCHOR + CFG_BLOCK),
])

print("REMINDER: launch env must set B1K_STAGE_OVERSAMPLE=8 (after patch_stage_oversample.py)")
print("REMINDER: norm stats must be recomputed over the MERGED mix before training")
print("RUN-2 CONFIG PATCH APPLIED")
