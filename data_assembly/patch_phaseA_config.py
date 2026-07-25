"""Add Phase-A training config: 1800-episode dataset, no episode filter (G3's config pins
G3_EPISODES=120 which would silently shrink the run), point conditioning on.
The contact-weight A/B is env-gated (B1K_CONTACT_WEIGHTS), so ONE config serves both arms."""
P = "/root/openpi/src/openpi/training/config.py"
src = open(P).read()
if "pi05_phaseA_point" in src:
    print("already patched"); raise SystemExit

anchor = '        name="pi05_g3_smoke",'
i = src.find(anchor)
assert i > 0
# find the start of that TrainConfig( block
j = src.rfind("    TrainConfig(", 0, i)
new_cfg = '''    TrainConfig(
        # PHASE A: 1800 episodes / 24 tasks (4 gate x 200 + 20 diversity x 50), mixed
        # supervision (real target_points where labelled, mask=False elsewhere).
        # Contact oversampling is env-gated (B1K_CONTACT_WEIGHTS=1) so the A/B is one variable.
        name="pi05_phaseA_point",
        model=pi0_config.Pi0Config(
            action_horizon=32, pi05=True, point_conditioning=True, point_noise_std=0.02,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="b1k_phaseA",
            base_config=DataConfig(
                data_cls=_lerobot_compat.LeRobotDataset,
                dataset_root="/root/phaseA/b1k_phaseA",
                prompt_from_task=True,
                dataset_kwargs={"tolerance_s": 5e-4},   # NOTE: no episodes filter (use all)
            ),
            robot_config_name="b1k/R1Pro",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*lora.*|.*point_.*",
        ),
        num_train_steps=60_000,
        save_interval=10_000,
        keep_period=20_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
    ),
'''
src = src[:j] + new_cfg + src[j:]
open(P, "w").write(src)
import ast; ast.parse(src)
print("added pi05_phaseA_point; syntax OK")
