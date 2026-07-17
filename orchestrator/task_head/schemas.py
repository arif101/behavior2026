"""Data contracts for the BEHAVIOR-1K hierarchical stack.

The `Subtask` model is the interface between the Task Head (this package),
the grounding head (locates `grounding_query` in the image -> 3D point),
and Pi 0.5 (executes `instruction` conditioned on that 3D point).
"""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Primitive skills Pi 0.5 can execute. Keep this vocabulary closed so the
    low-level policy always receives an action it was trained/prompted on."""

    NAVIGATE_TO = "navigate_to"
    PICK = "pick"
    PLACE_ON = "place_on"
    PLACE_INSIDE = "place_inside"
    OPEN = "open"
    CLOSE = "close"
    TOGGLE_ON = "toggle_on"
    TOGGLE_OFF = "toggle_off"
    POUR_INTO = "pour_into"
    WIPE = "wipe"
    PUSH = "push"
    PULL = "pull"
    RELEASE = "release"


class Subtask(BaseModel):
    """One atomic step. This is the JSON object handed to the grounding head."""

    index: int = Field(description="0-based position in the plan")
    action: ActionType = Field(description="Primitive skill for Pi 0.5")
    instruction: str = Field(
        description=(
            "Short natural-language command for Pi 0.5, e.g. "
            "'pick up the red mug on the counter'"
        )
    )
    grounding_query: str = Field(
        description=(
            "Open-vocabulary phrase for the grounding head to locate in the "
            "image — the single object this action operates on, described "
            "distinctively, e.g. 'red ceramic mug on the kitchen counter'"
        )
    )
    reference_query: Optional[str] = Field(
        default=None,
        description=(
            "Second object for relational actions (place_on/place_inside/"
            "pour_into): the destination to ground, e.g. 'top shelf of the "
            "open cabinet'. None for single-object actions."
        ),
    )
    success_criteria: str = Field(
        description=(
            "A visually checkable condition that holds after this subtask "
            "succeeds, e.g. 'the mug is inside the cabinet and visible on "
            "the shelf'"
        )
    )


class Plan(BaseModel):
    """Full decomposition returned by the Task Head planner."""

    task_summary: str = Field(description="One-sentence restatement of the goal")
    scene_notes: str = Field(
        description="Relevant objects/state observed in the initial image"
    )
    subtasks: List[Subtask]


class VerificationResult(BaseModel):
    """Closed-loop verdict after a subtask was executed."""

    subtask_succeeded: bool
    observation: str = Field(
        description="What the new image shows relative to the success criteria"
    )
    decision: Literal["continue", "retry", "replan"] = Field(
        description=(
            "continue: advance to the next subtask. "
            "retry: re-issue the same subtask (transient failure). "
            "replan: the scene diverged from the plan; regenerate remaining steps."
        )
    )
    replan_hint: Optional[str] = Field(
        default=None,
        description="If decision=replan, what changed and how the plan should adapt",
    )
