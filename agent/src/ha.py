"""Home Assistant API surface: the MCP endpoint, result resolver, and notifications."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import httpx
from livekit.agents import mcp
from livekit.agents.llm import ToolError
from mcp.types import TextContent

from config import MCP_PATH, settings

logger = logging.getLogger("ha-mcp-agent.ha")


def mcp_url() -> str:
    """Build the HA MCP endpoint from the configured base URL."""
    return f"{settings.ha_url.rstrip('/')}{MCP_PATH}"


def text_result_resolver(ctx: mcp.MCPToolResultContext) -> str:
    """Return MCP results as plain text (HA sends a single text block).

    Keeps results readable for the LLM and lets our function tools parse the payload
    directly instead of unwrapping the default JSON envelope. HA reports some failures
    as a normal result, `{"success": false, "error": ...}`; those raise a ToolError.
    """
    parts = [c.text for c in ctx.result.content if isinstance(c, TextContent)]
    if not parts:
        return json.dumps([item.model_dump() for item in ctx.result.content])
    text = "\n".join(parts)
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if isinstance(data, dict) and data.get("success") is False:
        raise ToolError(str(data.get("error") or text))
    return text


async def _post_service(path: str, payload: dict[str, Any]) -> bool:
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


async def push(
    service: str, message: str, data: dict[str, Any], title: str = ""
) -> bool:
    """Send to one ``notify.<service>`` with a companion-app ``data`` payload."""
    payload: dict[str, Any] = {"message": message, "data": data}
    if title:
        payload["title"] = title
    return await _post_service(f"notify/{service.removeprefix('notify.')}", payload)


async def subscribe(event_type: str) -> AsyncIterator[dict[str, Any]]:
    """Yield the data of each HA event of ``event_type``, reconnecting when dropped."""
    url = f"{settings.ha_url.rstrip('/').replace('http', 'ws', 1)}/api/websocket"
    while True:
        try:
            async with (
                aiohttp.ClientSession() as http,
                http.ws_connect(url, heartbeat=30) as ws,
            ):
                await ws.receive_json()  # auth_required
                await ws.send_json({"type": "auth", "access_token": settings.ha_token})
                if (await ws.receive_json()).get("type") != "auth_ok":
                    raise RuntimeError("Home Assistant refused the token")
                await ws.send_json(
                    {"id": 1, "type": "subscribe_events", "event_type": event_type}
                )
                async for msg in ws:
                    if msg.type is aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if data.get("type") == "event":
                            yield data["event"]["data"]
        except Exception:
            logger.warning(
                "lost the %s subscription; retrying", event_type, exc_info=True
            )
        await asyncio.sleep(10)


async def progress(body: dict[str, Any]) -> bool:
    """Post one turn phase to the livekit_voice integration's progress API."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{settings.ha_url.rstrip('/')}/api/livekit_voice/progress",
                headers={"Authorization": f"Bearer {settings.ha_token}"},
                json=body,
            )
            resp.raise_for_status()
        return True
    except Exception:
        logger.exception("failed to post turn progress (%s)", body.get("phase"))
        return False
