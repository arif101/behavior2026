# Run-3 packing recipe (trainer box)

Sources (all LeRobot-v3, schema-identical, each carrying `gt_depth_ds` + `sample_weight`):
1. `b1k_radio_map` — 200 human demos (organizers' preprocessed; restore from `arif101/b26-foveated-backup-20260730`).
2. `b1k_radio_factory` — 38 honest grasp+transport segment clips (HF `arif101/b26-radio-manufactured/b1k_radio_factory`).
3. `b1k_radio_episodes` — 58 complete episodes: honest grasp + policy's learned in-hand press (HF `.../b1k_radio_episodes`).

Steps (on the trainer, after the three roots exist):
```
python add_sample_weights.py --root /root/b1k_radio_map --poison poison_windows.json   # 10,000 rows (2.33%) -> 0.1
python add_sample_weights.py --root /root/b1k_radio_factory
python add_sample_weights.py --root /root/b1k_radio_episodes
python assemble_run3_mix.py --dry-run          # validated 2026-09-03 on the A5000: 296 eps / 464,242 frames
python assemble_run3_mix.py --out /root/b1k_radio_run3mix
```
Trainer env: `B1K_SAMPLE_WEIGHT_COL=sample_weight` (fork_snapshot/data_loader.py multiplies the column into the
frame sampler; combine with `B1K_STAGE_OVERSAMPLE` as in Run-2). Advisory source weights from the dry run:
map 1.0 / factory 4.78 / episodes 2.0 (factory weight matches the map's stage-1->2 transition-neighborhood mass).

Arms (single-variable, pre-registered): A0 demos-only (map, no down-weighting) · A1 map+down-weighting ·
A2 A1+factory · A3 A1+episodes · A4 A1+factory+episodes. Bars: held-out instance grasp completion (fingertip
closes the last 10cm and the weld condition triggers), copycat guard, liveness probes; 2025 winner = 26%.

## Source 4 (added 2026-09-04): `b1k_radio_approach` — honest pre-contact approach episodes
Manufactured by `factory_approach_cap.py` (restore pre-pull, 11-DOF servo across the rig's gap, finger-aligned closure,
verified weld, carry-in, transport). Convert with `--clips "/root/factory_obs2/rac_*_200.npz" --out /root/b1k_radio_approach`.
Validated on d20: converts cleanly; right-EE-to-button 0.427 m at frame 0 -> 0.136 m (the approach is present); stage 43% acquire.
Meta carries the causal-honesty certificate (first_contact, radio_disp_approach) + verified weld + bit-exact re-render.
Covers the reachable subset (arm+trunk, pull <= ~0.55 m); demos with larger rig pulls (d10: 1.04 m) need the base-drive variant.
Arm A5 = A1 + factory + episodes + approach. Known residual: in-hand attitude ~13 deg off the demo grip (v5 recipe; v7 fixes).

**A5 data is READY (2026-09-06):** `b1k_radio_approach` = 12 honest pre-contact approach episodes / 6,942 frames, on HF arif101/b26-radio-manufactured:b1k_radio_approach. Validated: gt_depth_ds + sample_weight present, right-EE-to-button starts 0.36m (approach present), stage 54/46. Covers the mainstream-upright demo subset (high-yaw instances did not generalize). Pull it, add_sample_weights.py --root /root/b1k_radio_approach, include as A5 = A4 + approach.
