# Closure research pass (2026-09-21) — why the policy does not servo the last 15 cm, and what the field says

Context: probe round closed (RUN3_CONFIG.md): pointer on/off/oracle, map tokens, history-off, pre-oriented wrist, close
gate all ~0/10 grasps on instance 301; the policy reaches the human closing distance band but not the rail pose and
closes on the demo timeline. Corrective-field measurement (jacobian_probe.sh) running on the sim box.

## 1. The 2025 winner (arXiv 2512.06951, pi0.5, 26 % q over 50 tasks) — same base model, same cameras, same 224 px
- Data: 10,000 demos / 50 tasks (200 per task, as ours), NO corrective or self-collected data, ~2 epochs, 15 days on
  8xH200 multi-task then ~1 week per task group. Frozen SigLIP-So400m/14, 3 cams at 224x224; 720 px "did not lead to
  meaningful changes". Chunk 30, execute 26 (spline-compressed), 4-step overlap inpainted at replans.
- Attention change: LEARNABLE MIXED-LAYER ATTENTION — each action-expert layer j attends to a learned linear blend of
  ALL VLM layers' K/V (init = identity to layer j; ~18 scalars per layer). Not ablated.
- System-2 stage tracking: linear classifier on VLM features (99 % train acc), 5-15 stages per task, 2-of-3 sliding
  window to advance, unanimous to roll back. Addresses non-Markovian aliasing (start/end look alike).
- Correlated noise for flow matching (N(0, 0.5 Sigma_actions + 0.5 I)) + correlation-aware inpainting at replans.
- THE RULE THAT MATTERED: "if the gripper is closed and it was never closed in training data for this task at the same
  stage, we treat it as a failed grasp and completely open the gripper" -> "approximately doubled our success rate in
  selected tasks where grabbing objects was a common failure mode" (2.2x q on a 13-task subset). This is OUR failure
  mode, seen at scale: premature/failed closes followed by post-grasp behaviour. Their fix converts it into RETRIES.
- Radio-specific rule: roll back 2 stages if the final stage is reached without success (the press/toggle problem).
- Failure analysis: dexterity (cannot reliably pick up or release) ~1/3 of failures; 14/50 tasks at 0; binary success
  11-12 %. Radio: partial credit, few full successes.
- Implication: with the same model, cameras and 200 demos/task, they got a nonzero radio number WITHOUT corrective data.
  Two things they have that we do not: (a) multi-task pretraining on 50 tasks (general grasp/release dexterity), (b) the
  stage-conditioned gripper reopen rule (retries). Everything else we already match or exceed in machinery.

## 2. Copycat / causal confusion (why a low-loss BC policy replays instead of servoing)
- Causal Confusion in Imitation Learning (de Haan 2019); Fighting Copycat Agents (Wen 2020): with observation history
  the imitator predicts the previous action instead of reading the state; residual-action prediction and history
  masking are the standard remedies. 2025: Action Chunking and Exploratory Data Collection (Simchowitz 2025) — chunking
  plus exploratory/noisy data collection gives exponential improvements in continuous control; policies with vision +
  proprio over-rely on proprio and fail to recover from OOD states. Directly the mechanism our measurement tests.

## 3. Corrective coverage (the in-policy fix)
- DART (Laskey 2017): inject noise during demonstration so the supervisor demonstrates recovery; parity with DAgger at
  3x lower cost. We can run DART and DAgger offline in sim with the factory servo as the supervisor (restore + perturb +
  servo-to-grasp), which no other team can do with human teleop. pi-labs RLT (2026): a special RL token bootstraps a
  lightweight online RL policy from the VLA for the precision phases only (screw insertion 20 % -> 65 %, up to 3x faster
  in the precise stages); SFT alone gets no signal from failures. Our reverse-curriculum + weld reward is the sim analogue.

## 4. Spatial reasoning in the attention (the user's question)
- SpatialVLA (RSS 2025): Ego3D position encoding — depth-derived 3D coordinates added to SigLIP patch tokens so the
  policy attends by geometry; adaptive action grids. We have depth at eval; a 3D PE on wrist/head patch tokens is a
  contained change on top of the frozen tower.
- Embodied CoT (Zawalski 2024; Revisiting ECoT 2026; CoT-VLA CVPR 2025): the policy emits grounded intermediate tokens
  (gripper pixel position, object bounding boxes, movement primitive) BEFORE the action; gains on precision; reasoning
  dropout keeps the grounding fields from destabilising. For us: an explicit gripper-frame rail-offset token stream with
  exact sim labels, consumed by the action attention, is the principled "reason then act" for the alignment.
- pi0 mechanics: the action expert attends to the VLM prefix K/V through a blockwise mask; adaRMS conditioning is a
  POOLED, global scale/shift. Our temporal gists and target points enter through adaRMS, i.e. not spatially. The winner's
  mixed-layer attention widens WHICH VLM layers the expert reads; SpatialVLA/ECoT change WHAT is in the tokens.
- Caveat from our own data: the oracle-pointer probe fed the exact target displacement (global vector) on every step
  and the policy still did not align. Representational changes alone are unlikely to create a servo behaviour that the
  training signal never demanded; they need to be paired with corrective data (Sec. 3) so the attention has something
  to learn from at the misaligned states.

## 5. Ranked plan (pending the corrective-field measurement)
1. Serving: stage-conditioned gripper REOPEN rule (legal: gripper qpos from proprio + StageV2 tracker "not lifted") ->
   retries instead of holding a closed gripper for 2,500 steps. The winner's single largest measured gain. One day.
2. Data: DART/DAgger offline with the servo supervisor at the policy's own near-grasp states; train from A4.
3. Architecture (with 2, not instead of it): mixed-layer attention (cheap), Ego3D PE on patch tokens, explicit
   rail-offset reasoning tokens with sim labels. Correlated-noise inpainting at replans for the chunk-boundary closes.
4. Longer: multi-task pretraining across the challenge tasks (the winner's biggest structural advantage).
