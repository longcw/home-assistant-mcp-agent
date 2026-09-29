"""Async REST client for the scheduler service (the docker-compose 'scheduler').

The scheduling function tools call the CRUD helpers here; the headless runner reports
each run's outcome via `report_run`. Calls raise RuntimeError with the server detail.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlencode

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


def _owner(user: str | None) -> str:
    """The query naming whose tasks these are; other people's are not found."""
    return f"?{urlencode({'user': user})}" if user else ""


async def create_task(payload: dict, user: str | None) -> Any:
    return await _request("POST", f"/tasks{_owner(user)}", payload)


async def list_tasks(user: str | None, active_only: bool = True) -> Any:
    flag = "true" if active_only else "false"
    sep = "&" if user else "?"
    return await _request("GET", f"/tasks{_owner(user)}{sep}active_only={flag}")


async def update_task(task_id: str, payload: dict, user: str | None) -> Any:
    return await _request("PATCH", f"/tasks/{task_id}{_owner(user)}", payload)


async def delete_task(task_id: str, user: str | None) -> Any:
    return await _request("DELETE", f"/tasks/{task_id}{_owner(user)}")


async def _settings() -> dict | None:
    try:
        return await _request("GET", "/settings") or {}
    except Exception:
        logger.exception("failed to fetch settings")
        return None


def _person(data: dict, user: str | None) -> dict | None:
    if user is None:
        return None
    return next(
        (p for p in data.get("users") or [] if p["name"].casefold() == user), None
    )


async def resolve_user(
    name: str | None = None, ha_user_id: str | None = None
) -> str | None:
    """The person a request speaks for, as a casefolded name; None for no one.

    A typed name is taken as given, even one the Settings tab does not list; an HA login
    counts only when a person in the Settings tab is linked to it.
    """
    if name and name.strip():
        return name.strip().casefold()
    if not ha_user_id:
        return None
    data = await _settings() or {}
    linked = (p for p in data.get("users") or [] if p.get("ha_user_id") == ha_user_id)
    person = next(linked, None)
    return person["name"].casefold() if person else None


async def notify_targets(user: str | None = None) -> list[str] | None:
    """Best-effort: the person's notification channels; HA's own for anyone else.

    None (not []) on failure so the caller falls back to a default channel rather than
    sending nothing; an explicit [] means the person disabled every channel.
    """
    data = await _settings()
    if data is None:
        return None
    person = _person(data, user)
    return list(person.get("notify_targets") or []) if person else None


async def phone(user: str | None) -> str:
    """The notify service showing a person's text-turn progress; "" for none.

    No one in particular gets TEXT_LIVE_ACTIVITY; a person gets the first phone in their
    Settings-tab devices, so one person's chat never shows on another's phone.
    """
    if user is None:
        return settings.text_live_activity
    person = _person(await _settings() or {}, user)
    targets = person.get("notify_targets") or [] if person else []
    return next((t for t in targets if t.startswith("mobile_app_")), "")


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
