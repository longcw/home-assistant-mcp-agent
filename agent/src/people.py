"""The people in the card's Settings tab, which the scheduler service stores.

Who a request speaks for, and that person's notification devices, progress phone and
restricted MCP servers.
"""

from __future__ import annotations

import logging

from config import settings
from scheduler_client import request

logger = logging.getLogger("ha-mcp-agent.people")


async def _settings() -> dict | None:
    try:
        return await request("GET", "/settings") or {}
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


async def allows_server(user: str | None, server_id: str) -> bool:
    """Whether the Settings tab lists a restricted MCP server for this person."""
    person = _person(await _settings() or {}, user)
    return bool(person and server_id in (person.get("servers") or []))


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
