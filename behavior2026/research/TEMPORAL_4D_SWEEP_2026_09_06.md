# Temporal / 4D / history-conditioned VLA sweep — 2026-09-06 (skeptic-verified)

Scope: what changed since the 08-20 Temporal Forcing read, restricted to evidence fetched today from primary
sources (arXiv HTML, GitHub API, Semantic Scholar). Every number below was read from the source; nothing is from
memory. Our pipeline: pi0.5 fork (PaliGemma/SigLIP + flow-matching expert), single frame, 3 RGB + proprio, chunk 32.
Prior context: `box_artifacts/temporal_forcing_dossier_2026_08_20.md`, `DYNAMIC_HASHMAP_RESEARCH_2026_08_20.md`.

## 0. Headline in five lines

1. Temporal Forcing (2608.30643) is still v1, code unreleased, 0 citations, no reproduction, no flow-matching result.
2. Its thesis now has THREE independent pi0.5/pi0 flow-matching analogs: TemporalFlow-VLA (2608.26821, pi0.5, 3 seeds,
   history-shuffle diagnostic), MotionVLA (2606.08288, pi0), ForeTime-VLA (2608.20735, pi0.5, mask-off controls).
3. Convergent, measured on pi0.5 by three unrelated groups: route history to the ACTION EXPERT, not the VLM prefix
   (2608.03052: +10.8 vs -0.6; RoboMME 2603.04639: modulator 44.5 vs context 30.7; TemporalFlow: queries visible to
   action tokens only). Raw multi-frame prefix is repeatedly neutral-to-harmful on flow VLAs.
4. Temporal context helps multi-stage/aliased tasks and is neutral-to-negative on single-stage contact skills in
   every paper that splits the two (TemporalFlow H=1 vs H=3; 2608.03052 family C; IntentVLA spoon; HAMLET mug).
5. Only StreamPI (2608.26067) ships runnable pi0.5/openpi code today; it is raw-frame, no liveness control, +1.4 LIBERO.

## 1. Temporal Forcing status (2608.30643, Ding et al., v1 2026-08-31)

- arXiv: single version (v1, 31 Aug 2026). Abstract: "Code will be publicly available." No URL. GitHub repo search
  "Temporal Forcing" VLA: 0 results (fetched 09-06). Semantic Scholar: citationCount 0, no citing papers. alphaXiv: 69
  views, no discussion. HF papers page: 404. No reproduction, follow-up, or rebuttal found anywhere.
- Base models in every table: StarVLA-OFT (Qwen3-VL-4B, L1 head) only. pi0 appears as a comparison row, never as a
  host. No flow-matching or diffusion head is trained with the method. No limitations section; no transfer discussion.
- Re-verified numbers: gate 0.6e-3 (history-only row 3) -> 1.1e-3 after transient open/collapse (row 4, w/o L_temp) ->
  13.4e-3 (row 8, full, "about 22 times"). Real robot: isolated 78/90 vs 75/90; sequential full-task 13/30 vs 6/30
  (43.3 vs 20.0). Latency 45 -> 58 ms/chunk; 16 history tokens; "backbone sequence length is unchanged".
  L_temp = 1/6 (L_state + 4 L_change + L_read). Trials: LIBERO 500/suite, RoboTwin 100/task, real 30. NO seed count.
- Verdict unchanged: best ablation discipline in the family, but as of today it is an unreplicated single-seed,
  single-backbone, L1-head result. Its transferable content is the *protocol* (zero-init gate + telemetry +
  supervise-the-pathway-itself + temporally-consistent targets), not the numbers.

## 2. What is new since 08-20 (fetched; numbers quoted with baselines)

### 2a. TemporalFlow-VLA — 2608.26821, v1 2026-08-27, HKUST-GZ + AgiBot (Yang, ..., Liang, Li)  **closest analog on pi0.5**
- Mechanism: two learnable temporal query tokens q8, q15 read frames t-8/t-15 (+ current) through a hierarchical
  mask; training-only decoders map each query (FiLM on patch states + 2-layer MLP) to a 16x16x2 "robot-surface
  temporal flow" built from FK + robot mesh + calibrated cameras (Huber loss, lambda_temp = 1.0). "Action tokens
  obtain historical information only through these two queries and cannot directly access historical image patches."
  "At deployment, the geometric label pipeline is absent and the auxiliary flow decoders are not evaluated."
- Base: pi0.5 (openpi), flow-matching expert. Numbers, RoboTwin 2.0 12 tasks, 100 rollouts/task (Table III, avg):
  Clean: pi0.5 79.0 | multi-frame raw history 79.8 | 2 queries w/o flow 83.2 | full 85.5.
  Randomized: 72.3 | 81.5 | 83.6 | 84.2. By horizon (clean): H=1 81.3->84.0 (+2.7); H=2 76.8->83.5; H=3 79.0->89.0.
  NOTE: raw multi-frame at H=1 clean DROPS 81.3->75.5; flow supervision at H=1 randomized is -1.2 vs 2Q w/o flow.
  LIBERO 3 seeds (0,2,5), 500 rollouts/suite/seed: 97.63+/-0.26 avg, Long 96.60+/-0.87 (cited pi0.5: 96.85 / 92.4).
  Real (AgiBot A3, 45 trials): three-cup stacking 57.8->77.8; two-bottle packing 86.7->97.8.
- Liveness evidence: offline diagnostic on a chronologically-ordered multi-frame pi0.5 baseline: "shuffling the three
  historical observations while fixing the current frame leaves the offline action flow-matching loss nearly
  unchanged, whereas removing history increases it by about 4.6%" -> raw history is consumed as a bag, not a sequence.
  With the full model, shuffling t-15/t-8 raises loss on 5 of 6 tested tasks. No gate (queries, not gated x-attn).
- Cost: "+approximately 20% wall-clock" training; 62.8-68.1 ms/replan on RTX 4090 (LIBERO Long). 8xH100 used.
- Skeptic: RoboTwin/real are single-seed; LIBERO pi0.5 baseline is cited, not reproduced; NO code or project page
  anywhere in the paper; H=1 randomized shows the supervision term slightly hurting. Grade B (3-seed LIBERO, shuffle
  diagnostic, clean H-split) — the strongest pi0.5-native evidence in this sweep.

### 2b. StreamPI — 2608.26067, v1 2026-08-26, HKU (Liu, ..., Zhao). Code: github.com/hku-sail/StreamPI (real openpi fork)
- Mechanism: no new params. Prefix = [img_t, text] x T with bidirectional attention inside each pair and causal across
  pairs; the flow expert's action block attends to ALL visible pairs (read from `src/openpi/models/pi0_rtc.py`
  embed_prefix/embed_suffix). Training masks a random number of LEADING history frames (`mask_num = randint(0, T)`),
  i.e. built-in history dropout; loss only on the last frame's chunk (RTC-style prefix inpainting also present).
  T=3/5 train, T=5 infer; random-interval sampling of history (`hist_interval`, `TemporalJitter`).
- Numbers (LIBERO, Table 1): pi0.5 avg 96.9 -> T=3 97.5 -> T=5 98.3; Long 92.4 -> 95.0. CALVIN avg len 4.313 -> 4.547.
  Real (ALOHA-style, 15-30 trials/task): shell game +33.3, rolling bottle +36.6, pen insertion +26.7, cup insertion +32.0.
  Table 5 (their own reproduced pi0.5 = 96.5): T=5-trained model at T=1 infer 97.1, T=3 97.4, T=5 98.3.
- Liveness: the Table-5 sweep IS a mask-off-at-eval delta: +0.6 from training alone, +1.2 more from history at
  inference. No shuffle / repeat-current control. Latency (4090): 94.4 -> 103.6 ms at T=5 (+9.2 ms).
- Skeptic: Table 1 pi0.5 row prints Goal 96.8 with avg 96.9, but (98.8+98.2+96.8+92.4)/4 = 96.55; openpi README says
  Goal 98.0 / avg 96.85. Either the Goal cell or the average is wrong; do not quote the Goal gain. No seeds. Weights
  promised for 08-30, not present in the repo as of 09-06 (last push 08-27; 215 stars). Raw-frame design is the one
  TemporalFlow and MotionVLA show hurting at H=1 — the random history-mask may be what saves it. Grade C+ (code).

### 2c. ForeTime-VLA — 2608.20735, v2 2026-08-24. pi0.5 + flow matching, conveyor-belt only
- Mechanism: 8-frame causal history encoder -> 64-D future-state code, distilled from a frozen video world model
  (Wan2.2-VAE-based "Fast-WAM"); slow path = 4 future tokens + 1 phase token appended to VLM prefix; fast path = 1024-D
  residual added to action-expert tokens. 1.251M added params; +2.5-2.9% latency (66.97 -> 68.62 ms).
- Numbers (90 trials/condition): stationary grasp 65.56 -> 81.11; slow-moving grasp 36.67 -> 58.89; fast 2/30 -> 11/30.
  Inference-time component masking (Table III, 96 windows): full MAE 0.1184; w/o future condition 0.1288; slow path
  only 0.1360; fast path only 0.1271. Paired bootstrap 95% CI on MAE gain [+0.82%, +4.48%].
- Skeptic: one embodiment, one domain (moving objects), no code URL, no seeds, world model in the loop for
  distillation. Useful only as a second pi0.5 demonstration that a distilled temporal code is read at inference
  (mask-off deltas) when the label depends on it. Grade C.

### 2d. How should VLAs use proprioceptive state? — 2608.03052, v1 2026-08-04. pi0.5, full fine-tune, RoboCasa365
- Not image history: 16-D kinematic state (EEF pose, base pose, gripper), K frames. Four routes: VLM prefix (vp),
  action prefix (ap: state tokens in the causal action suffix ahead of noisy actions), state expert (se), feature
  modulation (fm: per-layer x-attn -> gamma/beta).
- Composite (20 multi-stage tasks, 25 rollouts each), K=1 -> K=8: ap 28.2 -> 39.0 (+10.8); vp 34.4 -> 33.8 (-0.6);
  fm 27.8 -> 32.2; se 25.8 -> 28.0. Atomic (45 tasks, 50 rollouts): ap 55.7 -> 59.6; others ~flat.
  "Short histories improve ... whereas deeper uncompressed histories provide no additional benefit and eventually
  degrade control" (sweep to K=96). Family C knob/switch (no-state 39.5) degrades under long histories, worst via vp.
- Liveness (the cleanest control in the sweep): slot-matched REPEAT-CURRENT — "replacing the ordered eight-frame
  history by eight copies of the current state": ordered K=8 39.0 vs repeated 30.8 vs K=1 30.8; "the paired
  task-bootstrap confidence interval excludes zero." Also a flow-matching correction-alignment readout
  cos(c_t, r_t): ap1 0.079 -> ap8 0.270 (vp1 0.245).
- Skeptic: single seed per config; no code; state-only. Grade B for the routing and control methodology.

### 2e. IntentVLA — 2605.14712, v3 2026-08-30. Qwen3-VL-4B + GR00T-style DiT flow head; code: task defs only
- Mechanism: frozen VGGT-1B over K=16 frames -> 1 camera + 4 register tokens/frame -> gated x-attn
  F' = F + sigmoid(alpha) MHA(...) + one pooled "history evidence" token. No alignment loss. Gate init/value NOT reported.
- AliasBench (RoboTwin2, 12 tasks built so o_t^(1) ~ o_t^(2) but a^(1) != a^(2)): frame-only 9.0 | best raw history
  (4 of last 16 frames) 28.1 | MemoryVLA 14.9 | IntentVLA 45.8; back-and-forth phase repetition 6.0 -> 49.3.
  LIBERO-Long 97.4 (pi0.5 92.4). SimplerEnv: Put Spoon on Towel DROPS 83.0 -> 70.8 (single-stage contact task).
  K sweep: 8: 39.9, 16: 45.8, 24: 43.7. Encoder swap: V-JEPA2 38.6 vs VGGT 45.8.
- Liveness proxy only: inter-chunk L2 disagreement 0.219 -> 0.181. No shuffle/mask-off. 16xH100, 30k steps.
  Latency 11.77 -> 7.53 Hz, +4.8 GiB. Repo (13 stars, pushed 08-27) has envs/task_config only; model "coming soon".
- Skeptic: different backbone, gate unmeasured, code absent. AliasBench itself is the useful artifact. Grade C+.

### 2f. MotionVLA — 2606.08288, v1 2026-06-06. pi0 (PaliGemma-3B + flow expert)
- Past-only window (5 frames, stride 2) -> frozen TraceAnything trajectory-field tokens -> query-conditioned x-attn
  retrieval + MLP recoupling; stage-I alignment + auxiliary trajectory loss.
- RoboTwin2 avg 41 -> 53; LIBERO 94.2 -> 95.4 (Long 85.2 -> 91.2); real 28.7 -> 41.9. Ablation (2-task avg): full 53 |
  history-off 37 | RAW HISTORY FRAMES ALONE 13 | unaligned 4D tokens 16 | w/o stage-I alignment 40 | w/o aux loss 50.
- Skeptic: no seeds, no trial counts for sim, no code, 7.1 FPS. Grade C. Value: third replication that raw frames
  hurt a flow VLA and that alignment on the injected pathway is what makes it usable.

### 2g. RoboMME — 2603.04639, v1 2026-03-04 (Dai, ..., Finn, Fazeli, Chai). pi0.5 memory-route study, 16 tasks
- pi0.5 17.93 | + past actions 19.73 | FrameSamp+Context 30.68 | FrameSamp+Modulator 44.51 | FrameSamp+Expert 36.25 |
  TokenDrop+Modul 38.04 | TTT variants 21.96-22.28 | RMT variants 19.46-20.17 | human oracle 90.5.
  Modulator = action features x-attend memory tokens -> AdaLN scale/shift in the action expert (512-token budget).
  "Fine-tuning pi0.5 with shallow recurrent layers leads to unstable training." Benchmark + code public.
- Skeptic: no shuffle control; no per-variant compute. Grade B- for the routing result (many variants, one codebase).

### 2h. Others fetched (short)
- RoboTTT 2607.15275 (NVIDIA GEAR): GR00T N1.7 DiT + TTT-MLP fast weights, 8K-step context; 71.5% at 8K vs 43.9% at 1K;
  Gear Bot 2/10 vs 0/10. 16x GB200 pretrain + 8 per task. No code. Out of a 1-GPU-day budget; RoboMME's TTT-on-pi0.5
  ~22% says the small-scale port fails. Discard for us.
- muVLA 2606.12497: OpenVLA-OFT + 64 recurrent memory tokens; MIKASA-Robo 0.42 -> 0.84; LIBERO 97.1 -> 96.2; noise
  intervention 0.94 -> 0.09 (memory is read). Needs receding-horizon re-query; open-loop chunking "Long-10: 95.8 -> 5.4".
  Incompatible with chunk-32 execution as designed.
- MemoryVLA++ 2606.09827 / LaMem-VLA 2607.07608: CogACT-based (7B Prismatic + DiT). Big gains vs CogACT (SimplerEnv
  57.3 -> 73.9 both) but no shuffle/corruption controls, no code (LaMem repo link only). Grade C, wrong backbone.
- FM-VLA 2607.18231: pi0.5/openpi; 8 force-VAE tokens + 1 state-history token in the ACTION SUFFIX. Press-button-N-times
  11.1 -> 72.2 (visual memory pi-MEM 33.3); wipe-N-times 0.0 -> 77.8; +3.3 ms. 18 trials/task. Relevant to our
  ToggledOn press bug: a count/contact signal, not visual history, fixed it. No force sensing in our obs (proprio only).
- Explicit Language Memory 2608.04765: pi0.5 + high-level PaliGemma writing text memory. BEHAVIOR-1K "turn on radio":
  PressOn 0 -> 10%, avg 30 -> 40, trial count NOT stated, no ablation, no code. Same premature-stage failure we see.
- World Tokens 2608.09730: Qwen3-VL-2B + DiT-B flow; 256 world tokens exclusively routed to the expert; training-only
  Cosmos-Predict2.5 denoiser. R1 Pro single-arm fruit placing 59.4 -> 76.0 (96 trials); LIBERO Long w/o wm 97.0 -> 95.0.
  Not history-conditioned; Cosmos-2B in the training loop is far outside 1 GPU-day. Watch.
- VLA-JEPA 2602.10098 (ECCV26, code, 562 stars) and JEPA-VLA 2602.11832: pretraining-stage recipes (50k steps, 8xA100).
  Not a fine-tune add-on. Watch.
- HAMLET 2510.00695 v3: GR00T N1.5 LIBERO 95.6 -> 97.6; naive moment concat 62.7 vs 62.6 baseline on RoboCasa; Coffee
  Setup Mug 29 -> 28. Already parked; Temporal Forcing beat it 98.8 vs 97.6 in-paper.
- Spatial Forcing 2510.12276 v2: pi0 on RoboTwin, OpenVLA-OFT on LIBERO (97.1 -> 98.5); layer-24 cosine alignment;
  3.8x faster convergence. No pi0.5. Predecessor protocol only.
- TTF-VLA 2508.19257 v3 (training-free token reuse, OpenVLA, code): LIBERO-Spatial -1.5 in one variant. Not for us.
- ST-pi 2604.17880 (code repo is README+assets only), VLA-4D 2511.17199, ConsisVLA-4D 2605.05126, StreamingVLA
  2603.28565 (latency, not history), TBD-VLA 2606.07895: no liveness controls or not history-conditioned. Discard.

## 3. Long-horizon vs single-stage: where the split is measured

Helps full-task, isolated stage unchanged/small:
- Temporal Forcing real: isolated 75 -> 78 /90; sequential 6 -> 13 /30. (L1 head, single seed.)
- TemporalFlow-VLA RoboTwin clean: H=1 +2.7 | H=2 +6.7 | H=3 +10.0; "explicit recent execution history is most
  useful when success depends on maintaining progress across multiple sequential stages." (pi0.5.)
- 2608.03052 pi0.5: composite +10.8 (ap, K=8) vs atomic +3.9; repeat-current control rules out capacity.
- IntentVLA AliasBench back-and-forth phase repetition 6.0 -> 49.3 (the "repeat a done step" failure, by construction).
- RoboMME counting/permanence suites and RoboMemArena (pi0.5 TSR 21.5 vs PrediMem 38.5) are memory-by-construction.
- 2608.04765 radio: pi0.5 "predicts place the radio on the table, advancing before the required button press".

Does NOT help / hurts contact-rich single-stage:
- TemporalFlow raw multi-frame H=1 clean 81.3 -> 75.5; full model H=1 randomized 85.5 (2Q) -> 84.3 (2Q+flow).
- IntentVLA Put Spoon on Towel 83.0 -> 70.8. HAMLET Coffee Setup Mug 29 -> 28. muVLA LIBERO 97.1 -> 96.2.
- 2608.03052 family C knob/switch: long histories "substantially degrade", worst through the VLM prefix.
- MotionVLA raw frames 13 vs history-off 37. Precision/contact stages want the current frame, cleanly.
Reading for BEHAVIOR: temporal context is a full-task (progress/aliasing) instrument. It will not move grasp precision
or the commit-mode absence in RUN2_EVAL_REPORT; the RL skill remains the terminal-value manufacturer.

## 4. Recommendation for THIS pipeline (one channel, pi0.5 flow expert, 1 GPU-day)

Design (each choice cites the evidence that picked it):
1. Content: K=8 past observations at chunk-boundary stride (covering ~2-4 chunks of 32), NOT consecutive frames
   (2608.03052 K=8 sweet spot; TemporalFlow t-8/t-15; StreamPI `hist_interval`). Encode with the frozen SigLIP we
   already run (pool to per-frame gist) — do not add DINOv2/VGGT (latency, IntentVLA 11.8 -> 7.5 Hz).
2. Compression: a 2-4 learnable temporal query tokens reading the K gists through a small causal transformer
   (TemporalFlow's q8/q15; Temporal Forcing's 16 tokens). No raw patch tokens reach the expert.
3. Route: into the ACTION EXPERT ONLY — either as extra suffix tokens ahead of the noisy actions (2608.03052 "ap";
   FM-VLA; TemporalFlow mask) or via x-attn -> AdaRMS modulation (RoboMME modulator, our existing AdaLN plumbing).
   Never the VLM prefix (2608.03052 -0.6; RoboMME context 30.7 vs 44.5). Zero-init the gate/residual (Temporal Forcing).
4. Supervising loss ON the pathway (pre-gate), training-only heads deleted at inference, weight ~1.0 (TemporalFlow
   lambda_temp = 1.0): predict per-frame change targets, not per-frame state (Temporal Forcing: framewise-3D hurt;
   change term weighted 4x). Legal targets we have in sim and nobody else does:
   (a) robot-surface / end-effector flow between t-k and t from GT joint state + FK + camera intrinsics (TemporalFlow
       recipe, exact in sim); (b) GT object-pose deltas for task-relevant objects (scene-flow of the BDDL objects);
   (c) GT-depth-derived 3D point displacement of wrist-cam patches; (d) a stage-progress/"done-set" readout target from
   the BDDL predicate trace (our stage head already exists — this is the aliasing-breaking target the others lack).
   Optional: StreamVGGT features run offline on RGB (Temporal Forcing) — legal, but (a)-(d) are strictly better targets.
   Compose with counterfactual pairs (hashmap fix #1): pairs with identical current obs and different correct chunks
   are exactly IntentVLA's aliasing definition; the flow-loss floor ||a-a'||^2/4 is the demand-side readout.
5. Liveness instrumentation (all cheap, all from fetched papers): gate magnitude curve every eval (TF signatures:
   flat = inert, transient-collapse = abandoned, open-hold = adopted); repeat-current control (2608.03052) and
   history-shuffle control (TemporalFlow) as two extra eval arms of the same checkpoint; mask-off-at-eval delta on
   full-task success (StreamPI T=5 -> T=1; TF gate-off 5.4pp); flow-correction alignment cos(c_t, r_t) (2608.03052).
6. Budget: TemporalFlow reports ~+20% wall-clock; StreamPI +9 ms; ForeTime 1.25M params. A 4-token pathway with two
   MLP flow heads fits a 15k-step arm at batch 32 with margin.

Null result and how to read it:
- Inert channel: gate stays <1e-3, mask-off delta ~0, repeat-current == ordered, cos alignment ~0.08 unchanged.
  Cause is supervision (Temporal Forcing row 3). Fix: raise lambda_temp, check the target is not window-constant.
- Live-but-unneeded: gate opens and holds, aux loss falls, but mask-off delta ~0 AND full-task success unchanged.
  Then aliasing is not the binding failure. Confirm by measuring the aliasing rate in our own data first: fraction of
  training rows whose nearest-neighbour by current-obs embedding has a different correct chunk (AliasBench criterion)
  and the offline "remove history" vs "shuffle history" loss deltas (TemporalFlow: 4.6% vs ~0 on a raw baseline).
  If remove-delta ~ shuffle-delta ~ 0 on our data, the channel cannot be demanded by BC and the arm should not run.
- Abandoned: gate opens then collapses (TF row 4 pattern) — supervision is landing off-pathway; move it.

## 5. Ranked table

| # | Paper (arXiv, date) | Mechanism | Grade | Fit to pi0.5 flow expert | Cost | Verdict |
|---|---|---|---|---|---|---|
| 1 | TemporalFlow-VLA 2608.26821 (08-27) | 2 temporal queries, FK-flow training-only targets, expert-only interface | B (3 seeds LIBERO, shuffle diag, H-split; RoboTwin single seed, no code) | Native pi0.5/openpi | +20% train, ~0 infer | adopt-candidate: the design above is this recipe with sim-GT targets |
| 2 | Proprio-history study 2608.03052 (08-04) | K=8 state history into action prefix; repeat-current control | B (many routes, one seed, CI on control) | Native pi0.5 | ~0 | adopt: routing rule + two probes, zero-cost arm |
| 3 | Temporal Forcing 2608.30643 (08-31) | zero-init gated x-attn + StreamVGGT alignment | B- (best ablation; 1 seed, L1 head, no code, 0 cites) | Untested on flow | +13 ms | watch; adopt protocol only |
| 4 | RoboMME 2603.04639 (03-04) | pi0.5 memory-route bake-off | B- (one codebase, no controls) | Native pi0.5 | n/a | adopt finding: modulator > context; TTT/RMT fail |
| 5 | StreamPI 2608.26067 (08-26) | raw multi-frame prefix, causal pairs, history dropout | C+ (no seeds, table arithmetic off, weights pending) | Native openpi code | +9 ms, 0 params | watch; use its data pipeline + T=1 mask-off probe, not its design |
| 6 | IntentVLA 2605.14712 (v3 08-30) | VGGT 16-frame gist + gated x-attn, AliasBench | C+ (gate unmeasured, model code absent) | GR00T-style flow head, other VLM | -36% Hz | watch; AliasBench criterion for our aliasing audit |
| 7 | MotionVLA 2606.08288 (06-06) | trajectory-field tokens + alignment on pi0 | C (no seeds, no code) | pi0 flow | 7 FPS | watch; raw-frames-hurt replication |
| 8 | ForeTime-VLA 2608.20735 (08-24) | 8-frame code distilled from video WM, dual path on pi0.5 | C (one domain) | Native pi0.5 | 1.25M params | watch; mask-off protocol |
| 9 | FM-VLA 2607.18231 (07-20) | force/state tokens in action suffix | C (18 trials) | Native pi0.5 | +3 ms | watch; needs force we lack; motivates count/progress target (4d) |
| 10 | Explicit Lang. Memory 2608.04765 (08-05) | HL text memory over pi0.5, BEHAVIOR radio | C- (no trial count, no ablation) | pi0.5 low-level | +1 VLM call | discard as design; keep as failure-mode witness |
| 11 | RoboTTT 2607.15275 (07-15) | TTT fast weights in DiT, 8K context | C (20 trials, GB200 scale) | wrong scale | 16x GB200 | discard |
| 12 | muVLA 2606.12497 (06-10) | recurrent memory tokens, receding horizon | C+ (interventions) | needs per-step re-query | high | discard (chunk-32 incompatible) |
| 13 | MemoryVLA++ / LaMem-VLA (06/07) | memory banks + imagination on CogACT | C | wrong backbone | +0.05 s, +6 GB | discard |
| 14 | World Tokens 2608.09730 (08-10) | training-only Cosmos denoiser, exclusive routing | C+ (R1 Pro 96 trials) | flow DiT, other VLM | Cosmos-2B in loop | watch |
| 15 | VLA-JEPA 2602.10098 (ECCV26, code) | JEPA pretrain then fine-tune | B- | other VLM, pretrain stage | 50k steps 8xA100 | watch |
| 16 | HAMLET 2510.00695 | moment tokens + memory module | C+ | GR00T flow | 1.02x | superseded |
| 17 | TTF-VLA, ST-pi, VLA-4D, ConsisVLA-4D, StreamingVLA, TBD-VLA | misc | C/– | – | – | discard |

## 6. Caveats the reader must carry

- Every "adopt" above rests on <=3 seeds; only TemporalFlow's LIBERO row has error bars. No paper measures our
  regime (3.2% base success, 100 tasks, chunk 32, mobile bimanual). Chunk-32 open-loop execution means a history
  channel can only act at chunk boundaries; nobody in this set evaluated a 32-step chunk with history.
- The two strongest pi0.5 results (TemporalFlow, 2608.03052) have no code; StreamPI has code but a weaker design.
- Temporal Forcing's gate telemetry is the only *direct* liveness measurement in the family; everything else infers
  liveness from removal/shuffle deltas. Ship the gate, and ship the repeat-current arm.
