"""Runtime configuration for the scheduler service, read from the environment.

The service shares the repo-root ``.env`` with the agent worker (see docker-compose.yml),
so LiveKit credentials and ``AGENT_NAME`` are the same values the worker uses. It needs the
LiveKit *server* credentials (not the inference gateway) because it dispatches the worker
into rooms via the AgentDispatchService API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    # Dispatch name of the worker to fire scheduled tasks against; must match the worker's
    # AGENT_NAME (agent/agent.py) and the integration's configured agent name.
    agent_name: str
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
        livekit_url=os.environ.get("LIVEKIT_URL", ""),
        livekit_api_key=os.environ.get("LIVEKIT_API_KEY", ""),
        livekit_api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
        agent_name=os.environ.get("AGENT_NAME", "ha-agent"),
        db_path=os.environ.get("SCHEDULER_DB", "/data/scheduler.db"),
        default_tz=os.environ.get("AGENT_TZ") or os.environ.get("TZ") or "UTC",
        misfire_grace_seconds=int(os.environ.get("MISFIRE_GRACE_SECONDS", "3600")),
        port=int(os.environ.get("PORT", "8080")),
        auth_token=os.environ.get("SCHEDULER_TOKEN", ""),
    )
