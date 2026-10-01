"""Home Assistant's MCP server: its endpoint, and how its tool results read."""

from __future__ import annotations

import json

from livekit.agents import mcp
from livekit.agents.llm import ToolError
from mcp.types import TextContent

from config import MCP_PATH, settings


def mcp_url() -> str:
    """Build the HA MCP endpoint from the configured base URL."""
    return f"{settings.ha_url.rstrip('/')}{MCP_PATH}"


def text_result_resolver(ctx: mcp.MCPToolResultContext) -> str:
    """Return MCP results as plain text (HA sends a single text block).

    Keeps results readable for the LLM and lets our function tools parse the payload
    directly instead of unwrapping the default JSON envelope. HA reports some failures
    as a normal result, `{"success": false, "error": ...}`, and a script can only fail
    by returning `{"error": ...}` as its response; both raise a ToolError.
    """
    parts = [c.text for c in ctx.result.content if isinstance(c, TextContent)]
    if not parts:
        return json.dumps([item.model_dump() for item in ctx.result.content])
    text = "\n".join(parts)
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if not isinstance(data, dict):
        return text
    if data.get("success") is False:
        raise ToolError(str(data.get("error") or text))
    if isinstance(result := data.get("result"), dict) and result.get("error"):
        raise ToolError(str(result["error"]))
    return text
