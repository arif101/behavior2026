# 4D-attendable perception for the action expert (spec v1, 2026-09-22)

Why (measured, RUN3_CONFIG.md): at the pre-grasp the policy has no image-driven corrective response (radio moved 3 cm,
proprio fixed: +0.004 m push-back = noise), corrects hand displacements only through proprio (joint-trajectory return),
does not hold the grasp attitude (wrist to a policy-preferred pose 20-80 deg off within one chunk), and closes on the
demo timeline. Our spatial/temporal conditioning enters through adaRMS (pooled, global scale/shift): nothing the expert
can attend to by location. The 2025 winner's base is the same pi0.5; its gains were attention-side + a gripper rule.

## A. Tokens the expert can attend to, positioned in one 4D frame
A1 Ego3D on current patches (SpatialVLA-style, but exact): per SigLIP patch, the 3D point from TRUE depth (train:
   gt_depth_ds, 16x16 per camera, already in the mix; eval: wrapper depth) + intrinsics, then camera->base via the
   robot's own kinematics (wrapper `rel(sensor)`; train: one FK precompute from proprio). All three cameras land in the
   BASE frame (SpatialVLA: monocular-estimated depth, single camera, camera frame). Encoding: sinusoidal on (x,y,z) ->
   MLP -> added to the patch token in embed_prefix right after PaliGemma.img; output layer zero-init.
A2 History as 4D tokens (replaces the pooled gist -> gate): per past head frame (K=8, stride 32) the frozen tower's
   16x16 patch tokens average-pooled to a 4x4 grid = 16 tokens, each with the 3D point of its cell (pooled depth) and
   its time offset; positions re-expressed in the CURRENT base frame through the base odometry between capture and now
   (proprio twist integration, drift 0.14 m/episode, already in the wrapper) -> 4D PE (x,y,z,t). Appended after the map
   tokens (RoPE-safe tail), ar_mask False. Precompute: one token set per FRAME (16 x 1152 bf16 = 37 KB; 473k frames =
   17 GB parquet), the loader gathers K frames by delta_timestamps as it does for gists; serve: the ring buffer stores
   token sets + base poses. Aux: predict the rail's CURRENT-frame 3D position from history tokens only ("temporal
   grounding"; label = target_points_v2 of the current frame) so the 4D PE cannot be ignored.
A3 Geometric attention bias (query-side geometry; SpatialVLA has none): in gemma.py where logits are computed, add
   gamma[layer, head] * exp(-|p_s - a_t|^2 / sigma^2) for suffix queries t (anchors = right and left fingertip
   positions from proprio, base frame; two kernels) against 3D-positioned keys s (current patches, history tokens, map
   tokens); zero for text. gamma init 0 -> bit-identical at step 0; gamma trajectory during training and an eval-time
   "3D off" ablation are the liveness readouts (the 4 dead injected channels are the precedent).
A4 Mixed-layer attention (winner): each expert layer reads a learned blend of all VLM layers' K/V, identity init.

## B. Training signal that forces A to be used
B1 Corrective corpus (DART/DAgger offline, servo supervisor): from harvested near-grasp states on training layouts,
   servo to P0 + delta (delta over +-8 cm position, +-25 deg wrist, incl. the policy's own stall poses), then the v13d
   servo to a verified weld + the human transport/press tail; record obs+actions as clips (factory pipeline). ~300 clips.
B2 Proprio noise/dropout at training (copycat remedy): Gaussian noise on EE/joint proprio (~2-3 cm equivalent) and
   p=0.3 proprio token masking, so the fine approach cannot be read off the joints.
B3 Serving: stage-conditioned gripper REOPEN rule (winner's 2.2x): gripper closed + tracker says not lifted for N steps
   -> open -> retry. Correlated-noise inpainting at replans later.

## C. Order and cost
1 (sim box, 1 day) B3 + full-ckpt n=25 with the rule.  2 (trainer, 3-4 days) A1-A4 + B2 in the fork, parity smokes.
3 (sim box, parallel, 3 days) B1 corpus + A2 token precompute (tower pass; reuse decode pipeline).
4 (trainer, 2 days) fine-tune from A4, 15k steps; gamma/ablation liveness; n=25 eval with B3.
5 4-task multi-task with A in place from the start.
