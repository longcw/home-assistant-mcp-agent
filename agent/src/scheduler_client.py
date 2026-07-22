"""Async REST client for the scheduler service (the docker-compose 'scheduler').

The scheduling function tools call the CRUD helpers here; the headless runner reports
each run's outcome via `report_run`. Calls raise RuntimeError with the server detail.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger("ha-mcp-agent.scheduler")


async def _request(method: str, path: str, payload: dict | None = None) -> Any:
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


async def create_task(payload: dict) -> Any:
    return await _request("POST", "/tasks", payload)


async def list_tasks(active_only: bool = True) -> Any:
    flag = "true" if active_only else "false"
    return await _request("GET", f"/tasks?active_only={flag}")


async def update_task(task_id: str, payload: dict) -> Any:
    return await _request("PATCH", f"/tasks/{task_id}", payload)


async def delete_task(task_id: str) -> Any:
    return await _request("DELETE", f"/tasks/{task_id}")


async def report_run(run_id: str, status: str, result: str) -> None:
    """Report a scheduled run's outcome back to the scheduler (best-effort)."""
    if not run_id:
        return
    try:
        await _request(
            "POST", f"/internal/runs/{run_id}", {"status": status, "result": result}
        )
    except Exception:
        logger.exception("failed to report run %s outcome", run_id)
