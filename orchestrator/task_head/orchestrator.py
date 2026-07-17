"""Closed-loop Task Head orchestrator for BEHAVIOR-1K.

Lifecycle:
    orch = TaskOrchestrator(ClaudeVLMClient())
    plan = orch.start(task_description, initial_rgb)
    while (sub := orch.current_subtask()) is not None:
        point_3d = grounding_head(sub, rgb, depth)      # friend's module
        report   = pi05_execute(sub, point_3d)          # low-level policy
        rgb      = get_camera_image()
        orch.report_outcome(rgb, report)                # verify + advance/retry/replan
    print(orch.status)  # "success" | "failed"
"""

import logging
from typing import Any, List, Optional

from . import prompts
from .schemas import Plan, Subtask, VerificationResult
from .vlm_client import VLMClient

logger = logging.getLogger("task_head")


class TaskOrchestrator:
    def __init__(
        self,
        vlm: VLMClient,
        max_retries_per_subtask: int = 2,
        max_replans: int = 3,
    ):
        self.vlm = vlm
        self.max_retries_per_subtask = max_retries_per_subtask
        self.max_replans = max_replans

        self.task_description: str = ""
        self.plan: Optional[Plan] = None
        self._cursor: int = 0
        self._retries: int = 0
        self._replans: int = 0
        self.completed: List[Subtask] = []
        self.status: str = "idle"  # idle | running | success | failed

    # ------------------------------------------------------------------ plan
    def start(self, task_description: str, rgb_image: Any) -> Plan:
        """Decompose the long-horizon task into ordered subtasks."""
        self.task_description = task_description
        self.plan = self.vlm.generate_structured(
            system=prompts.PLANNER_SYSTEM,
            text=prompts.build_planning_prompt(task_description),
            images=[rgb_image],
            schema=Plan,
        )
        self._cursor = 0
        self._retries = 0
        self.completed = []
        self.status = "running" if self.plan.subtasks else "success"
        logger.info(
            "Planned %d subtasks: %s",
            len(self.plan.subtasks),
            [s.instruction for s in self.plan.subtasks],
        )
        return self.plan

    # ------------------------------------------------------------ sequencing
    def current_subtask(self) -> Optional[Subtask]:
        """The subtask to execute next, or None when the episode is over."""
        if self.status != "running" or self.plan is None:
            return None
        return self.plan.subtasks[self._cursor]

    # ----------------------------------------------------------- closed loop
    def report_outcome(
        self, rgb_image: Any, execution_report: str = "policy finished the motion"
    ) -> VerificationResult:
        """Feed the post-execution image back; verify and advance the plan.

        `execution_report` is any status string from the Pi 0.5 wrapper
        (e.g. "grasp closed on object", "IK failed", "timeout").
        """
        sub = self.current_subtask()
        if sub is None:
            raise RuntimeError("report_outcome called but no subtask is active")

        result = self.vlm.generate_structured(
            system=prompts.VERIFIER_SYSTEM,
            text=prompts.build_verification_prompt(
                self.task_description,
                sub.model_dump_json(indent=2),
                execution_report,
            ),
            images=[rgb_image],
            schema=VerificationResult,
        )
        logger.info(
            "Subtask %d (%s): succeeded=%s decision=%s — %s",
            sub.index,
            sub.action.value,
            result.subtask_succeeded,
            result.decision,
            result.observation,
        )

        if result.decision == "continue":
            self.completed.append(sub)
            self._cursor += 1
            self._retries = 0
            if self._cursor >= len(self.plan.subtasks):
                self.status = "success"
        elif result.decision == "retry":
            self._retries += 1
            if self._retries > self.max_retries_per_subtask:
                logger.warning("Retry budget exhausted; escalating to replan")
                self._replan(rgb_image, result.observation)
        else:  # replan
            self._replan(rgb_image, result.replan_hint or result.observation)

        return result

    # --------------------------------------------------------------- replan
    def _replan(self, rgb_image: Any, hint: str) -> None:
        self._replans += 1
        if self._replans > self.max_replans:
            self.status = "failed"
            logger.error("Replan budget exhausted; episode failed")
            return

        failed = self.current_subtask()
        completed_str = "\n".join(
            f"{s.index}. [{s.action.value}] {s.instruction}" for s in self.completed
        )
        self.plan = self.vlm.generate_structured(
            system=prompts.PLANNER_SYSTEM,
            text=prompts.build_replan_prompt(
                self.task_description,
                completed_str,
                failed.model_dump_json(indent=2) if failed else "(none)",
                hint,
            ),
            images=[rgb_image],
            schema=Plan,
        )
        self._cursor = 0
        self._retries = 0
        self.status = "running" if self.plan.subtasks else "success"
        logger.info(
            "Replanned (%d/%d): %d subtasks remain",
            self._replans,
            self.max_replans,
            len(self.plan.subtasks),
        )


def run_episode(
    orchestrator: TaskOrchestrator,
    task_description: str,
    get_observation,  # () -> rgb image (any format encode_image accepts)
    grounding_head,   # (subtask: Subtask, rgb) -> point_3d
    execute_pi05,     # (subtask: Subtask, point_3d) -> str execution report
    max_steps: int = 50,
) -> str:
    """Reference driver wiring the three heads together. Returns final status."""
    rgb = get_observation()
    orchestrator.start(task_description, rgb)

    for _ in range(max_steps):
        sub = orchestrator.current_subtask()
        if sub is None:
            break
        rgb = get_observation()
        point_3d = grounding_head(sub, rgb)
        report = execute_pi05(sub, point_3d)
        rgb = get_observation()
        orchestrator.report_outcome(rgb, report)

    if orchestrator.status == "running":
        orchestrator.status = "failed"  # step budget exhausted
    return orchestrator.status
