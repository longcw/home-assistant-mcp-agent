"""Home Assistant API surface: the MCP endpoint, result resolver, and notifications."""

from __future__ import annotations

import json
import logging

import httpx
from livekit.agents import mcp
from mcp.types import TextContent

from config import MCP_PATH, settings

logger = logging.getLogger("ha-mcp-agent.ha")


def mcp_url() -> str:
    """Build the HA MCP endpoint from the configured base URL."""
    return f"{settings.ha_url.rstrip('/')}{MCP_PATH}"


def text_result_resolver(ctx: mcp.MCPToolResultContext) -> str:
    """Return MCP results as plain text (HA sends a single text block).

    Keeps results readable for the LLM and lets our function tools parse the payload
    directly instead of unwrapping the default JSON envelope.
    """
    parts = [c.text for c in ctx.result.content if isinstance(c, TextContent)]
    if parts:
        return "\n".join(parts)
    return json.dumps([item.model_dump() for item in ctx.result.content])


async def _post_service(path: str, payload: dict[str, str]) -> bool:
    """POST to an HA service (``/api/services/<path>``). Returns True on success."""
    try:
        base = settings.ha_url.rstrip("/")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{base}/api/services/{path}",
                headers={"Authorization": f"Bearer {settings.ha_token}"},
                json=payload,
            )
            resp.raise_for_status()
        return True
    except Exception:
        logger.exception("failed HA service call %s", path)
        return False


# Fallback channel when the configured list can't be read (e.g. scheduler unreachable),
# so a notification is never silently dropped.
DEFAULT_CHANNELS = ["persistent_notification"]


async def notify(
    message: str, title: str | None = None, targets: list[str] | None = None
) -> bool:
    """Send to each configured channel (best-effort). True if any send succeeded.

    ``targets`` are the enabled channels: ``"persistent_notification"`` raises an in-HA
    notification; any other value is a ``notify.<service>`` (e.g. a phone via the HA
    Companion app). ``None`` means "unknown" → fall back to a persistent notification;
    an empty list means every channel is disabled → nothing is sent. ``title`` optional.
    """
    payload: dict[str, str] = {"message": message}
    if title:
        payload["title"] = title

    channels = DEFAULT_CHANNELS if targets is None else targets
    ok = False
    for target in channels:
        if target == "persistent_notification":
            sent = await _post_service("persistent_notification/create", payload)
        else:
            # accept "notify.mobile_app_x" or the bare service name "mobile_app_x"
            service = target.removeprefix("notify.")
            sent = await _post_service(f"notify/{service}", payload)
        ok = ok or sent
    return ok
