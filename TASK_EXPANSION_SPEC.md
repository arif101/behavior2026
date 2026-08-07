# TASK EXPANSION SPEC — 4 new tasks (signed off 2026-08-07)

picking_up_trash (task-0001) · picking_up_toys (task-0007) · bringing_in_wood (task-0015) ·
moving_boxes_to_storage (task-0016). Evidence: three scoping investigations 2026-08-07
(per-task ×2 + assembly-pipeline recon), all paths verified on box 69.30.85.197. Rule:
architecture stays task-agnostic — task-specificity lives ONLY in data/labels.

## 0. Why these tasks pay
3 of 4 are multi-room with long target-out-of-view stretches (trash: living→kitchen 16 m;
wood: garden→corridor ×3; boxes: living→garage through closed doors, first leg starts a
full house away) — the memory-demanding venue the frozen prefix-map bet was reserved for,
and the venue where displacement-target conditioning acts as a compass when vision can't.
All are multi-object sequential tasks → the stage/current-target machinery stops being
degenerate (radio had one cycle; these have 2-6). q scores per goal literal (1/3, 1/6,
1/3, 1/2 granularity) — partial credit exists on every one.

## 1. Per-task profiles (measured)

| | trash (0001) | toys (0007) | wood (0015) | boxes (0016) |
|---|---|---|---|---|
| scene | house_double_floor_lower (= radio) | house_single_floor (NEW) | =radio | =radio |
| rooms | kitchen+living | childs_room | garden+corridor | living→garage |
| demo frames (mean) | 5.5k | 19.5k (9× radio) | 13.4k | 14.6k |
| raw GB | 3.4 | 13 | 16 | 8.8 |
| goal | 3 cans inside trash_can (forall) | 6 toys inside toy_box (3 forall literals) | 3 plywood ontop corridor floor | 2 boxes stacked in garage (either order) |
| q granularity | 1/3 | 1/6 | 1/3 | 1/2 |
| new skills | floor grasp (z≈0.07, deep crouch) | near-flat puzzle pinch (1 cm), box packing | thin-plank floor grasp (2 cm) | precision stack (0.40 m footprint), doors |
| targets | can bbox-center → trash_can rim (23×23×28 cm, open top) | toy bbox-center → toy_box top (0.32×0.46×0.14 — shallow!) | plank base-link → hindsight floor placement | box base → hindsight placement → live top-face of placed box |
| notable | 2 can models; cans on FLOOR | rollbacks up to 41/demo; −1/6 regressions IN successful demos; box overflow risk | plywood is a 0.47 m ~0.5 kg PLANK (not a sheet — bimanual NOT required); ep 00151170 junk (1 frame), 00152720 suspicious | garage door closed; stack topple before episode end zeroes the literal |

Shared (all verified = radio conventions): 20 public_test instances each (ids 301-320,
`--instance-indices 0..19`), 300 train instances (pose-only variation, overlaid on the
0_0_template scene); 23-d action; 61-d proprio robot-level; raw hdf5 = state-only (obs
must come from the organizers' rendered videos or replay capture); doors start CLOSED in
wood+boxes; eval timeout 1.5× mean demo (toys ≈ 28k steps ≈ 16 sim-min/rollout).

## 2. The assembly recipe (radio path, generalized per task)

Training OBS come from the organizers' consolidated LeRobot repo (2026-challenge-demos —
955 data shards + 17k videos, filtered per task via meta/episodes `tasks`/`task_index`;
per-key video chunk/file columns, NOT data indices — the Phase-A boundary-shard trap ate
204 episodes silently). POSE LABELS come from sim replay of the raw hdf5s (poses_x JSONs).
Ordered steps per task (reusable → new):

1. ✅ raw demos fetched (42 GB, all 4 tasks, 0 failures).
2. Mirror extraction — generalize `data_assembly/build_radio_dataset.py` (TASK_INDEX,
   names) + `fix_meta.py`. **Videos are the disk problem: ~700-800 GB total for 4 tasks;
   box has ~108 GB free** → per-task assemble→label→upload→delete, or rent storage.
   Also fetch `annotations/task-XXXX/` (skill segments with `manipulating_object_id` +
   frame ranges = ground-truth current-target stream).
3. Pose replay (`pose_fleet.py`/`replay_poses_batch.py`, reusable; needs 3 new
   `task_objects.json` entries — trash's exists). **THE LONG POLE: ~0.12 s/step →
   t1≈51 h, t7≈130 h, t15≈76 h, t16≈100 h single-proc; ~2-4 days wall at 4-8 workers.**
   Requires an RT-core sim box; conflicts with eval campaigns + ResFiT on the A5000.
4. Object-center target labels — `poses_to_labels.py` → **`add_target_points.py` (already
   multi-object!** nearest not-yet-retired instance per arm, displacement+stationary+
   released retirement logic, per-frame `chosen` names) → per-task dataset.
5. NEW current-target resolver: reconcile add_target_points `chosen` ∥ reward-delta
   segmentation (per-literal ±1/N spikes in raw hdf5, frame-aligned to parquet) ∥
   annotation `manipulating_object_id`. Emits per-frame target name + world pos —
   feeds map write_target, stages, aux_pixels.
6. `build_episode_map.py` (raw_episode_id column exists; keep the EE-proximity physical
   verification — naive id arithmetic was wrong for 192/200 radio episodes).
7. Map tokens — `offline_driver.py` generalized (per-frame current-target). 21 ms/frame,
   OMP_NUM_THREADS=1 mandatory; t7 needs a NEW house_single_floor map.
8. Join — `add_map_labels.py` + `add_stage_pixel_labels.py` generalized; NEW per-task
   stage rule: per-cycle {approach, acquire, transport, deposit} keyed on the
   current-target stream + grasp state (radio's static-button geometry doesn't transfer);
   progress scalar = cumsum(reward)=q(t) — mechanically available, better than the
   stage-midpoint fallback radio used.
9. Norm stats + config (episodes filter, train/val, prompt = bare task identifier via
   fix_prompt pattern).

Container-opening refinement (trash rim / toy_box interior / stack top-face): one-time
AABB-top offsets per container model (offset_extract pattern) — design choice, not
blocker; v1 = bbox top-center.

Eval-side (separate from assembly): affordance head retrain per task (machinery reusable,
labels = projected object-center, ~2 h/task GPU); TASK_REGISTRY prompt patches;
target-object selection at serve = the affordance head's job on the current-target class.

## 3. Corrective data (the thesis carries over)
- Rollback branches in raw demos (t15: 176/200, t16: 191/200, t7 median 15/demo) =
  teleop error+recovery segments — FOUND corrective data, same species as RaC clips.
  Candidate: splice-convert rollback recoveries once the base pipeline lands.
- The RaC scaffold + splice generator + stage machinery are all task-agnostic by
  construction — they port with the data pipeline, per-task thresholds only.

## 4. Recommended execution order
1. **trash (0001) first**: cheapest (5.5k frames), same scene as radio (existing map
   reusable), task_objects entry exists, floor-grasp is the only new skill. Proves the
   generalized pipeline end-to-end.
2. **boxes (0016) + wood (0015)**: same scene again, door+carry skills, hindsight-target
   labels get exercised.
3. **toys (0007) last**: 9× frames, new scene (new map), hardest grasps, biggest disk.

## 5. Open decisions (user)
- **Compute**: pose replay needs ~2-4 days of an RT-core box that isn't doing eval/RL.
  Rent a second sim box for the replay fleet (isaac playbook applies), or serialize on
  the A5000 after Run-2 data-collection quiets?
- **Disk**: per-task assemble→upload→delete on the box (slow, careful) vs a ≥1 TB volume.
- **Scope of Run-3**: single-task pipelines first (4 independent finetunes to validate
  data), or go straight to the multi-task mix (the map-memory thesis venue)?

## 6. Known traps (inherited, do not re-learn)
Boundary-shard bonus episodes (drop explicitly) · positional pose-JSON↔episode pairing
(demo ids sparse; verify EE-proximity) · OMP threads kill the mapper (93.8→13 ms) ·
playback needs OG_PLAYBACK_REAL_FREQS=1 for any policy-control-after-restore ·
hdf5 `state` raw vector is NOT dump_state (623≠667 etc.) · per-episode init state lives
in state[0] not the scene json · eval room-loading may exceed demo room-loading (verify
before re-render) · `inside` raycast semantics on the shallow toy_box rim unverified.
