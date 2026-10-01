"""Async REST client for ha-notify-scheduler, which knows each person behind a user id.

The agent knows a person only by that id: the scheduling tools call the task CRUD with
it, and notifications, a text turn's progress and a person's allowed MCP servers go
through the service by it.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from config import settings

logger = logging.getLogger("ha-mcp-agent.scheduler")


class SchedulerClient:
    """Task calls raise RuntimeError with the server detail; the rest never raise."""

    def __init__(self, base_url: str, token: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else None

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
        params: dict[str, str] | None = None,
    ) -> Any:
        # naming the owner scopes a task call; other people's tasks are not found
        query = {**(params or {}), **({"user": user_id} if user_id else {})}
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.request(
                method,
                f"{self._base_url}{path}",
                json=payload,
                params=query or None,
                headers=self._headers,
            )
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"scheduler {method} {path} -> {resp.status_code}: {detail}"
            )
        return resp.json() if resp.content else None

    async def create_task(self, payload: dict, user_id: str | None) -> Any:
        return await self._request("POST", "/tasks", payload, user_id=user_id)

    async def list_tasks(self, user_id: str | None, active_only: bool = True) -> Any:
        params = {"active_only": "true" if active_only else "false"}
        return await self._request("GET", "/tasks", user_id=user_id, params=params)

    async def update_task(
        self, task_id: str, payload: dict, user_id: str | None
    ) -> Any:
        return await self._request(
            "PATCH", f"/tasks/{task_id}", payload, user_id=user_id
        )

    async def delete_task(self, task_id: str, user_id: str | None) -> Any:
        return await self._request("DELETE", f"/tasks/{task_id}", user_id=user_id)

    async def notify(
        self, user_id: str | None, message: str, title: str | None = None
    ) -> bool:
        """Send a notification to a person's channels; False when it was not sent."""
        body = {"user": user_id, "message": message, "title": title}
        try:
            data = await self._request("POST", "/notify", body)
        except Exception:
            logger.exception("failed to send a notification")
            return False
        return bool(data and data.get("sent"))

    async def progress(self, event: dict[str, Any]) -> None:
        """Show one phase of a person's text turn on their phone."""
        try:
            await self._request("POST", "/progress", event)
        except Exception:
            logger.exception("failed to post turn progress (%s)", event.get("phase"))

    async def user(self, user_id: str) -> dict[str, Any] | None:
        """A person's settings, such as their allowed MCP servers; None if unlisted."""
        try:
            return await self._request("GET", f"/users/{quote(user_id, safe='')}")
        except Exception:
            logger.warning("could not read user %s", user_id, exc_info=True)
            return None


scheduler = SchedulerClient(settings.scheduler_url, settings.scheduler_token)
