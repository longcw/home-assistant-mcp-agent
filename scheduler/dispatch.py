"""Fire a scheduled task by dispatching the worker into a fresh room.

At fire time the scheduler doesn't touch Home Assistant itself; it hands the task to the same
worker that serves interactive sessions, via the LiveKit AgentDispatchService. The worker's
entrypoint branches on the ``kind: "scheduled"`` metadata and runs the task headlessly (see
agent/agent.py). Keeping all Home Assistant access in the worker means the scheduler needs
only LiveKit credentials.
"""

from __future__ import annotations

import json
import logging

from livekit import api

from config import Config

logger = logging.getLogger("scheduler.dispatch")


def _http_url(url: str) -> str:
    """LiveKitAPI wants an http(s) URL; the shared LIVEKIT_URL is usually a ws(s) URL."""
    if url.startswith("wss://"):
        return "https://" + url[len("wss://") :]
    if url.startswith("ws://"):
        return "http://" + url[len("ws://") :]
    return url


async def dispatch_scheduled(
    cfg: Config,
    *,
    task_id: str,
    description: str,
    execution: dict,
    run_id: str,
    room: str,
) -> None:
    metadata = json.dumps(
        {
            "kind": "scheduled",
            "task_id": task_id,
            "run_id": run_id,
            "description": description,
            "execution": execution,
        },
        ensure_ascii=False,
    )
    async with api.LiveKitAPI(
        url=_http_url(cfg.livekit_url),
        api_key=cfg.livekit_api_key,
        api_secret=cfg.livekit_api_secret,
    ) as lkapi:
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=cfg.agent_name, room=room, metadata=metadata
            )
        )
    logger.info("dispatched task %s (run %s) to room %s", task_id, run_id, room)
