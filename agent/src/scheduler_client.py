"""Async REST client for ha-notify-scheduler, which knows each person behind a user id.

The agent knows a person only by that id: the scheduling tools call the task CRUD with
it, and notifications, a text turn's progress and a person's allowed MCP servers go
through the service by it. Task calls raise RuntimeError with the server detail.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote, urlencode

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


async def notify(user_id: str | None, message: str, title: str | None = None) -> bool:
    """Send a notification to a person's channels; False when it was not sent."""
    try:
        data = await request(
            "POST", "/notify", {"user": user_id, "message": message, "title": title}
        )
    except Exception:
        logger.exception("failed to send a notification")
        return False
    return bool(data and data.get("sent"))


async def progress(event: dict[str, Any]) -> None:
    """Show one phase of a person's text turn on their phone (best-effort)."""
    try:
        await request("POST", "/progress", event)
    except Exception:
        logger.exception("failed to post turn progress (%s)", event.get("phase"))


async def user(user_id: str) -> dict[str, Any] | None:
    """A person's settings, such as their allowed MCP servers; None when unlisted."""
    try:
        return await request("GET", f"/users/{quote(user_id, safe='')}")
    except Exception:
        logger.warning("could not read user %s", user_id, exc_info=True)
        return None
