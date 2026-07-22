"""Request/response schemas for the REST API.

The worker's scheduling tools speak this shape (see agent/agent.py), and the card renders the
``TaskOut`` payload the tools return. ``ScheduleSpec`` carries a discriminating ``type``;
``ExecutionSpec`` holds a list of deterministic ``steps`` and/or a natural-language
``instruction``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ScheduleSpec(BaseModel):
    type: Literal["once", "recurring"]
    # "once": an ISO-8601 datetime. The worker attaches the home timezone before sending, so
    # this is normally offset-aware; a naive value is interpreted in `timezone`.
    run_at: Optional[str] = None
    # "recurring": a standard 5-field cron expression (min hour dom month dow).
    cron: Optional[str] = None
    timezone: str = "UTC"

    @model_validator(mode="after")
    def _check(self) -> "ScheduleSpec":
        if self.type == "once" and not self.run_at:
            raise ValueError("run_at is required when schedule.type is 'once'")
        if self.type == "recurring" and not self.cron:
            raise ValueError("cron is required when schedule.type is 'recurring'")
        return self


class ToolCall(BaseModel):
    """One deterministic tool call (agent or MCP tool + args) replayed at fire time."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ExecutionSpec(BaseModel):
    """What a task does, in one shape:

    - ``steps``: an ordered list of concrete tool calls, replayed deterministically. They run
      in order and stop at the first failure.
    - ``instruction``: a natural-language instruction the LLM runs (via ``session.run``) after
      the steps — seeing their results — to summarize and/or chain further tool calls.

    At least one of the two must be present.
    """

    steps: list[ToolCall] = Field(default_factory=list)
    instruction: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "ExecutionSpec":
        if not self.steps and not (self.instruction and self.instruction.strip()):
            raise ValueError("execution needs at least one step or an instruction")
        return self


class TaskCreate(BaseModel):
    description: str
    schedule: ScheduleSpec
    execution: ExecutionSpec


class TaskUpdate(BaseModel):
    description: Optional[str] = None
    schedule: Optional[ScheduleSpec] = None
    execution: Optional[ExecutionSpec] = None
    enabled: Optional[bool] = None


class RunOut(BaseModel):
    id: str
    fired_at: str
    status: str
    result: Optional[str] = None


class TaskOut(BaseModel):
    id: str
    description: str
    schedule_type: str
    run_at: Optional[str] = None
    cron: Optional[str] = None
    timezone: str
    execution: dict
    status: str
    enabled: bool
    created_at: str
    # The next fire instant (ISO). For recurring tasks this is APScheduler's computed next run.
    next_run_at: Optional[str] = None
    runs: list[RunOut] = Field(default_factory=list)


class RunReport(BaseModel):
    """Posted by the worker after it executes a scheduled task."""

    status: Literal["success", "error"]
    result: Optional[str] = None
