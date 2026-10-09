# v2 labels (2026-09-16): crisp stage, progress, stage-indexed target, toggled — for the full-stack retrain

Script: `box_scripts/run3/relabel_v2.py` (validation: `validate_relabel.py`; inputs: `dl_relabel_inputs.py`). Adds FOUR new
columns to every source and leaves `stage` / `target_points` untouched, so A0–A4 and S1 stay byte-identical and the press
config selects the v2 columns explicitly.

| column | type | definition |
|---|---|---|
| `stage_v2` | int32 | 0 APPROACH, 1 GRASP (holding hand within 12 cm of the rail point, radio not lifted), 2 TRANSPORT (lifted), 3 PRESS (the free hand's final reach to the button and everything after). Monotone within an episode (cumulative max). |
| `progress` | f32 | (stage_v2 + within-stage fraction) / 4 — a piecewise-linear task clock that needs no demo timeline (works for mid-task clips). |
| `target_points_v2` | f32[6] | stage-indexed target for [L, R] EE displacements: before lift = RAIL grasp point = button + (0, 0, +0.141) in the base frame; after lift = the toggle button (= `target_points`). |
| `toggled` | int32 | 1 from the toggle frame on (human demos: the recorded reward spikes to 1 there; matches the snapshot bank's press frames within 5 frames). 0 elsewhere. |

Per-frame signals used (present in every source): `target_points` = [button − EE_L, button − EE_R] (base frame),
`stage` (converter pre/post-lift for manufactured clips), `metalink_labels/ep*.npz` button world z for the human demos
(lift = z rises > 3 cm over frame 0), the raw hdf5 `reward` for the toggle frame. Holding hand = the hand nearest the
button at the first lifted frame (map 191 R / 9 L; all manufactured clips R).

**Why the press is anchored on the toggle, not a radius.** Humans hover the free hand near the carried radio early in the
transport (median 22 cm, a quarter of frames under 11 cm): a plain radius test put press ~160 frames early and left
transport at 2 s. The final reach is the last frame before the toggle where the free hand was still > 25 cm away.
Factory and approach clips end 60 frames before the human's press → no press frames; policy-press episodes anchor on the
closest approach within the last 40%.

**Why the rail target is yaw-free.** Human grasps (19 strict clips) sit at (−0.013, y, 0.129) in the radio frame with y
free along the rail (std 0.065 m); the button metalink is at (0.045, 0.042, −0.012). The rail-minus-button vector
(−0.058, −0.042, +0.141) has a frame-independent vertical part and a 7 cm horizontal part that needs the radio yaw, which
the eval wrapper (affordance-predicted button point) cannot know. Training and eval use the SAME yaw-free target.
Measured against the true rail line with the approach clips' exact poses (4,295 pre-lift frames): 6.5 cm off at the median
(5.8–7.1), vertical error 3 mm, unbiased across yaws.

**Measured on the human demos (200 eps / 429,928 frames):** approach 49.5%, grasp 8.8%, transport 7.2%, press 34.5% of
which 28.0% (120,220 frames) is POST-TOGGLE — the toggle happens at ~70% of every demo and the human keeps recording
500–900 frames after success (carry/place). Lift − bank closure = 29 ± 2 frames on all 38 bank demos; toggle − lift median
254 frames (8.5 s); final reach median 176 frames before the toggle. Packing note: consider down-weighting `toggled==1`
frames — the policy is scored at the episode end and post-success wandering is a risk, not a skill.

**Eval-wrapper mirror (TODO, sim box):** AffordanceMapFullRes predicts the button point per frame → the wrapper can run
the same rule: lifted = predicted button z rises > 3 cm over its initial value; hand distances from proprio; stage_v2 and
target_points_v2 exactly as above (rail = predicted button + 0.141 up before lift). The stage token at serve therefore
comes from this geometric mirror, not from the stage head (whose logits `sample_actions` does not return).

Sources relabeled on the S1 trainer box (2026-09-16 00:10 UTC): b1k_radio_map, b1k_radio_factory, b1k_radio_episodes,
b1k_radio_approach_v2 (pre-v2 meta backed up as meta.pre_v2). The S1 mix is untouched (own files).

## Config switch + eval mirror (2026-09-16 01:xx UTC, commits ce1f71d, e75f0fb)
- `LeRobotB1KDataConfig.stage_key / progress_key / stage_tokens_key=None`; `pi05_radio_press` consumes stage_v2 /
  progress / target_points_v2 (tokens broadcast from stage_v2; the old stage_tokens repack KeyError'd on every dataset).
  `preflight_press_v2.py` PASS on `/root/b1k_radio_mix_a5_v2` (315 eps / 473,161 frames, real batches, CPU).
- Serve side: `eval/stage_v2_rule.StageV2Tracker` (causal) + `eval/stage_v2_wrapper.StageV2AffordanceWrapper`
  (AffordanceMapFullRes + tracker, exposes `_last_v2`) + `box_scripts/patch_v2_passthrough.py` (applied on the RTX box).
  Eval driver: `--env-wrapper behavior2026_eval.stage_v2_wrapper.StageV2AffordanceWrapper` with `--policy.config pi05_radio_press`.
- Reach radius 0.20 m (joint sweep over {0.25, 0.20, 0.15, 0.10}): mirror-vs-label agreement map 99.6% (transitions
  within 4 frames; press token leads the label by ≤ 70 frames in the worst 5%), episodes 98.8%, approach clips 76%
  (their transport tails keep the free hand near the radio; the mirror fires press there — benign direction).
- Sources re-labeled at reach 0.20 (map press 33.7%, transport 8.0%); v2 mix rebuilt.
