"""Async REST client for the scheduler service (the docker-compose 'scheduler').

The scheduling function tools call the task CRUD here, and people.py reads the Settings
tab through `request`. Calls raise RuntimeError with the server detail.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlencode

import httpx

from config import settings

logger = logging.getLogger("ha-mcp-agent.scheduler")


async def request(method: str, path: str, payload: dict | None = None) -> Any:
    url = f"{settings.scheduler_url.rstrip('/')}{path}"
    headers = (
        {"Authorization": f"Bearer {settings.scheduler_token}"}
        if settings.scheduler_token
        else None
    )
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.request(method, url, json=payload, headers=headers)
    if resp.status_code >= 400:
        detail = resp.text
        try:
            detail = resp.json().get("detail", detail)
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"scheduler {method} {path} -> {resp.status_code}: {detail}")
    return resp.json() if resp.content else None


def _owner(user: str | None) -> str:
    """The query naming whose tasks these are; other people's are not found."""
    return f"?{urlencode({'user': user})}" if user else ""


async def create_task(payload: dict, user: str | None) -> Any:
    return await request("POST", f"/tasks{_owner(user)}", payload)


async def list_tasks(user: str | None, active_only: bool = True) -> Any:
    flag = "true" if active_only else "false"
    sep = "&" if user else "?"
    return await request("GET", f"/tasks{_owner(user)}{sep}active_only={flag}")


async def update_task(task_id: str, payload: dict, user: str | None) -> Any:
    return await request("PATCH", f"/tasks/{task_id}{_owner(user)}", payload)


async def delete_task(task_id: str, user: str | None) -> Any:
    return await request("DELETE", f"/tasks/{task_id}{_owner(user)}")
