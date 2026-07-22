"""Request/response schemas for the REST API.

The worker's scheduling tools speak this shape (see agent/agent.py), and the card renders the
``TaskOut`` payload the tools return. ``ScheduleSpec`` and ``ExecutionSpec`` each carry a
discriminating ``type`` with the fields required for that variant validated up front.
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


class ExecutionSpec(BaseModel):
    type: Literal["command", "function_call"]
    # "command": natural-language instruction re-interpreted by the LLM at fire time.
    text: Optional[str] = None
    # "function_call": a Home Assistant MCP tool name + args, replayed deterministically.
    tool: Optional[str] = None
    args: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "ExecutionSpec":
        if self.type == "command" and not self.text:
            raise ValueError("text is required when execution.type is 'command'")
        if self.type == "function_call" and not self.tool:
            raise ValueError("tool is required when execution.type is 'function_call'")
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
