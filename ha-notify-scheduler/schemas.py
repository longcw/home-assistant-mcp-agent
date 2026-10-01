"""Request/response schemas for the REST API.

The worker's scheduling tools speak this shape (see agent/agent.py), and the card renders the
``TaskOut`` payload the tools return. ``ScheduleSpec`` carries a discriminating ``type``;
``ExecutionSpec`` holds a reminder's ``notification`` or a natural-language ``instruction``.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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


class Notification(BaseModel):
    """A reminder, sent as is; the title defaults to the task's description."""

    message: str
    title: Optional[str] = None


class ExecutionSpec(BaseModel):
    """What a task does at fire time, exactly one of:

    - ``notification``: sent to the owner's Home Assistant channels.
    - ``instruction``: sent as a message in the owner's text conversation, where the agent
      carries it out.
    """

    notification: Optional[Notification] = None
    instruction: Optional[str] = None

    @model_validator(mode="after")
    def _check(self) -> "ExecutionSpec":
        has_instruction = bool(self.instruction and self.instruction.strip())
        if (self.notification is not None) == has_instruction:
            raise ValueError("execution needs exactly one of notification or instruction")
        return self


class TaskCreate(BaseModel):
    description: str
    schedule: ScheduleSpec
    execution: ExecutionSpec
    user: Optional[str] = None


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
    user: Optional[str] = None
    # The next fire instant (ISO). For recurring tasks this is APScheduler's computed next run.
    next_run_at: Optional[str] = None
    runs: list[RunOut] = Field(default_factory=list)


class UserSettings(BaseModel):
    """One person: their id, which every service and the agent know them by, a display
    name, an optional HA login, their devices, and what they may use."""

    # fixed once set; defaults to the casefolded name, which is what older records used
    id: str = ""
    name: str
    ha_user_id: Optional[str] = None
    notify_targets: list[str] = Field(default_factory=list)
    # ids of the restricted MCP servers (mcp.yaml) the agent may use for this person
    servers: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("user name is empty")
        return v.strip()

    @model_validator(mode="after")
    def _default_id(self) -> "UserSettings":
        self.id = (self.id or self.name).strip().casefold()
        return self


class SettingsOut(BaseModel):
    # notify.* services notifications are also pushed to (besides the always-on
    # persistent_notification), e.g. ["mobile_app_iphone"], for anyone not in users.
    notify_targets: list[str] = Field(default_factory=list)
    users: list[UserSettings] = Field(default_factory=list)


class SettingsUpdate(BaseModel):
    notify_targets: Optional[list[str]] = None
    users: Optional[list[UserSettings]] = None

    @field_validator("users")
    @classmethod
    def _unique(cls, v: Optional[list[UserSettings]]) -> Optional[list[UserSettings]]:
        ids = [u.id for u in v or []]
        if len(ids) != len(set(ids)):
            raise ValueError("user ids must be unique")
        return v
