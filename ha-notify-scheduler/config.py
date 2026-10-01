"""Runtime configuration for the scheduler service, read from the environment.

The service shares the repo-root ``.env`` with the agent worker (see docker-compose.yml),
so the Home Assistant credentials and ``TEXT_API_TOKEN`` are the values the worker uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Home Assistant base URL and long-lived token, for a reminder's notification.
    ha_url: str
    ha_token: str
    # The agent's text chat endpoint and its bearer token, which run an instruction.
    chat_url: str
    chat_token: str
    # Seconds a fired instruction may take, waiting out the person's running turn included.
    run_timeout: float
    # SQLite file; the source of truth for tasks + run history. Mounted on a volume so it
    # survives container restarts (see docker-compose.yml).
    db_path: str
    # Timezone used for the APScheduler default and as the fallback when a task omits one.
    default_tz: str
    # A one-shot task whose fire time was missed while the service was down still runs once
    # if the outage was within this window; older misses are marked "missed" instead.
    misfire_grace_seconds: int
    port: int
    # Shared secret required on every request (Authorization: Bearer <token>) once the
    # service is published beyond the compose network. Empty disables the check.
    auth_token: str


def load_config() -> Config:
    return Config(
        ha_url=os.environ.get("HA_URL", ""),
        ha_token=os.environ.get("HA_TOKEN", ""),
        chat_url=os.environ.get("AGENT_CHAT_URL", "http://agent:8081/chat"),
        chat_token=os.environ.get("TEXT_API_TOKEN", ""),
        run_timeout=float(os.environ.get("SCHEDULED_RUN_TIMEOUT", "200")),
        db_path=os.environ.get("SCHEDULER_DB", "/data/scheduler.db"),
        default_tz=os.environ.get("AGENT_TZ") or os.environ.get("TZ") or "UTC",
        misfire_grace_seconds=int(os.environ.get("MISFIRE_GRACE_SECONDS", "3600")),
        port=int(os.environ.get("PORT", "8080")),
        auth_token=os.environ.get("SCHEDULER_TOKEN", ""),
    )
