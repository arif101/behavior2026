# BEHAVIOR-1K Task Head (Orchestrator)

VLM-based planner for a hierarchical control stack:

```
task description + RGB
        │
        ▼
┌───────────────────┐   Subtask (JSON)   ┌────────────────┐  3D point  ┌────────┐
│  Task Head (this) │ ─────────────────▶ │ Grounding Head │ ─────────▶ │ Pi 0.5 │
│  plan / verify /  │                    │  (open-vocab   │            │ policy │
│  retry / replan   │ ◀───────────────── │   detection)   │            └────────┘
└───────────────────┘  post-exec image
```

The Task Head decomposes a long-horizon BEHAVIOR-1K task into an ordered list
of atomic subtasks, hands them out **one at a time**, and after each execution
looks at a fresh camera image to decide `continue` / `retry` / `replan`
(closed-loop).

## Install & run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # or `ant auth login`
python example_run.py scene.png "put the apple in the fridge"
```

## Interface contract for the grounding head

Each call to `orchestrator.current_subtask()` returns a `Subtask`
(`task_head/schemas.py`). Serialized it looks like:

```json
{
  "index": 2,
  "action": "pick",
  "instruction": "pick up the red apple on the counter",
  "grounding_query": "red apple on the kitchen counter",
  "reference_query": null,
  "success_criteria": "the apple is held in the gripper, no longer on the counter"
}
```

The grounding head's job:

- Locate **`grounding_query`** in the current RGB(-D) frame and return one 3D
  point (robot/world frame — pick a convention with the Pi 0.5 wrapper).
- The query always names **exactly one object**, described distinctively
  (color/material/location), so a single detection is expected.
- For relational actions (`place_on`, `place_inside`, `pour_into`) the query
  is the **destination** (surface/container region); the held object is only
  mentioned in `instruction`. `reference_query`, when non-null, is an extra
  landmark that can be grounded to disambiguate.
- `action` is one of the closed vocabulary in `ActionType`
  (`navigate_to, pick, place_on, place_inside, open, close, toggle_on,
  toggle_off, pour_into, wipe, push, pull, release`).

After Pi 0.5 runs, the driver calls
`orchestrator.report_outcome(new_rgb, execution_report)` — the verifier VLM
checks `success_criteria` against the new image and advances/retries/replans.

## Integration points (in `run_episode`, `task_head/orchestrator.py`)

| Callable | Signature | Replace with |
|---|---|---|
| `get_observation` | `() -> rgb` | OmniGibson camera read (numpy `HxWx3 uint8`, path, bytes, or PIL all accepted) |
| `grounding_head` | `(Subtask, rgb) -> point_3d` | your friend's module |
| `execute_pi05` | `(Subtask, point_3d) -> str report` | Pi 0.5 rollout wrapper; return a short status string ("grasp closed", "IK failed", "timeout" — the verifier reads it) |

## Swapping the VLM backend

The orchestrator only uses `VLMClient.generate_structured(system, text,
images, schema)`. `ClaudeVLMClient` (Claude Opus 4.8, structured outputs) is
implemented; `QwenVLClient` is a stub for a fully-local deployment — serve
Qwen2.5-VL behind vLLM with guided JSON decoding and validate into the same
Pydantic schemas.

## Tuning knobs

- `max_retries_per_subtask` (default 2): retries before escalating to replan.
- `max_replans` (default 3): full replans before declaring the episode failed.
- Planner/verifier behavior lives in `task_head/prompts.py` — ordering rules,
  grounding-query style, and verification strictness are all prompt-level.
