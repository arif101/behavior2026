"""Smoke test for the Task Head with stubbed grounding + Pi 0.5.

Usage:
    export ANTHROPIC_API_KEY=...   # or `ant auth login`
    python example_run.py path/to/scene.png "put the apple in the fridge"

The grounding head and Pi 0.5 are stubs here — swap them for the real
modules; their call signatures are what the real ones should match.
"""

import logging
import sys

from task_head import ClaudeVLMClient, Subtask, TaskOrchestrator, run_episode

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    image_path = sys.argv[1]
    task = sys.argv[2] if len(sys.argv) > 2 else "put the apple in the fridge"

    def get_observation():
        # Real version: return env's current RGB frame (numpy HxWx3 uint8).
        # Stub reuses the same image, so expect the verifier to say "retry"
        # or "replan" — that exercises the closed loop.
        return image_path

    def grounding_head(subtask: Subtask, rgb):
        print(f"[grounding stub] query={subtask.grounding_query!r}")
        return (0.5, 0.1, 0.8)  # fake 3D point in robot frame

    def execute_pi05(subtask: Subtask, point_3d):
        print(f"[pi05 stub] {subtask.action.value}: {subtask.instruction!r} @ {point_3d}")
        return "policy finished the motion"

    orch = TaskOrchestrator(ClaudeVLMClient(), max_retries_per_subtask=1, max_replans=1)
    status = run_episode(orch, task, get_observation, grounding_head, execute_pi05, max_steps=6)
    print(f"\nEpisode status: {status}")
    print(f"Completed subtasks: {[s.instruction for s in orch.completed]}")


if __name__ == "__main__":
    main()
