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


async def notify(title: str, message: str) -> None:
    """Raise an HA persistent notification (best-effort)."""
    try:
        base = settings.ha_url.rstrip("/")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{base}/api/services/persistent_notification/create",
                headers={"Authorization": f"Bearer {settings.ha_token}"},
                json={"title": title, "message": message},
            )
            resp.raise_for_status()
    except Exception:
        logger.exception("failed to post HA notification")
