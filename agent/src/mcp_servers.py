"""The MCP servers beyond Home Assistant's, as the mcp.yaml config file lists them.

Values take ``${VAR}`` or ``${VAR:-default}`` from the environment, so the file carries
no secrets. An entry whose ``requires`` variables are unset is skipped, and a header
whose variable is unset is left out.
"""

from __future__ import annotations

import copy
import inspect
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml
from livekit.agents import RunContext, mcp
from livekit.agents.llm import (
    RawFunctionTool,
    ToolError,
    ToolFlag,
    Toolset,
    function_tool,
)

import ha
import scheduler_client as scheduler
from config import settings

logger = logging.getLogger("ha-mcp-agent.mcp")

_VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def _expand(value: str) -> str | None:
    """``value`` with its variables filled in; None if one is unset, with no default."""
    missing = False

    def fill(m: re.Match[str]) -> str:
        nonlocal missing
        found = os.getenv(m[1]) or m[2]
        missing = missing or found is None
        return found or ""

    expanded = _VAR.sub(fill, value)
    return None if missing else expanded


# Mem0 scope arguments; a person's memory toolset sets them, so the LLM never does
_MEMORY_SCOPE = ("user_id", "agent_id", "app_id", "run_id")


class PersonalMemory(mcp.MCPToolset):
    """Mem0's tools pinned to one person, so no one reaches another's memories."""

    def __init__(self, *, user_id: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._user_id = user_id
        self._pinned: Any = None

    async def setup(self, *, reload: bool = False) -> PersonalMemory:
        await super().setup(reload=reload)
        # setup() is a no-op once connected; re-pin only a freshly fetched tool list
        if self._tools is not self._pinned:
            self._tools = self._pinned = [self._pin(t) for t in self._tools]
        return self

    def _pin(self, tool: Any) -> Any:
        if not isinstance(tool, RawFunctionTool):
            return tool
        schema = copy.deepcopy(tool.info.raw_schema)
        params = schema["parameters"]
        props = params.get("properties", {})
        for key in _MEMORY_SCOPE:
            props.pop(key, None)
        if "required" in params:
            params["required"] = [r for r in params["required"] if r in props]
        name, user_id = tool.info.name, self._user_id

        async def call(raw_arguments: dict[str, Any]) -> Any:
            args = {k: v for k, v in raw_arguments.items() if k not in _MEMORY_SCOPE}
            if name == "add_memory":
                args["user_id"] = user_id
            elif "filters" in props:
                given = args.get("filters")
                args["filters"] = {
                    "AND": [{"user_id": user_id}, *([given] if given else [])]
                }
            return await tool(args)

        return function_tool(call, raw_schema=schema, flags=tool.info.flags)


class Guarded(mcp.MCPToolset):
    """A server whose tools a person may not be offered, or may have to confirm.

    With ``user_id``, it connects only for a person whose Settings-tab entry lists it. A
    tool in ``confirm`` runs only when called again after the person has spoken since
    its first call, which fails with a request to ask them.
    """

    def __init__(
        self, *, user_id: str | None, confirm: list[str], **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._user_id = user_id
        self._allowed: bool | None = None if user_id else True
        self._confirm = set(confirm)
        self._guarded: Any = None
        # tool name -> the person's latest message when it was first called
        self._asked: dict[str, str] = {}

    async def setup(self, *, reload: bool = False) -> Guarded:
        if self._allowed is None:
            person = await scheduler.user(self._user_id)
            self._allowed = bool(person and self.id in (person.get("servers") or []))
        # left unconnected, the toolset has no tools, so the LLM never sees them
        if not self._allowed:
            return self
        await super().setup(reload=reload)
        # setup() is a no-op once connected; wrap only a freshly fetched tool list
        if self._confirm and self._tools is not self._guarded:
            self._tools = self._guarded = [self._guard(t) for t in self._tools]
        return self

    def _guard(self, tool: Any) -> Any:
        if not isinstance(tool, RawFunctionTool) or tool.info.name not in self._confirm:
            return tool
        name = tool.info.name
        # a tool that reports progress takes the run context before its arguments
        takes_ctx = len(inspect.signature(tool).parameters) > 1

        async def call(ctx: RunContext, raw_arguments: dict[str, Any]) -> Any:
            said = next(
                (
                    i.id
                    for i in reversed(ctx.session.history.items)
                    if i.type == "message" and i.role == "user"
                ),
                "",
            )
            if self._asked.get(name, said) != said:
                self._asked.pop(name)
                return await (
                    tool(ctx, raw_arguments) if takes_ctx else tool(raw_arguments)
                )
            self._asked[name] = said
            # an error, so the call reads as not done rather than as a success
            raise ToolError(
                f"Not done yet: {name} needs the person's yes first. Say what it will "
                "do and ask them. If they agree in their next message, call it again."
            )

        return function_tool(
            call, raw_schema=tool.info.raw_schema, flags=tool.info.flags
        )


# per-person adapters an entry's ``per_person`` names
_PER_PERSON = {"mem0": PersonalMemory}


def _toolset(entry: dict[str, Any], user_id: str | None) -> Toolset | None:
    server_id = entry["id"]
    if any(not os.getenv(var) for var in entry.get("requires") or []):
        return None
    url = _expand(entry["url"])
    if not url:
        return None
    restricted = bool(entry.get("restricted"))
    if restricted and user_id is None:
        return None
    headers = {}
    for name, value in (entry.get("headers") or {}).items():
        if (expanded := _expand(str(value))) is not None:
            headers[name] = expanded
    base = settings.callback_base_url.rstrip("/")
    if entry.get("callback") and user_id and base and settings.text_api_token:
        headers["X-Callback-Url"] = f"{base}/chat/events?user={quote(user_id)}"
        headers["X-Callback-Token"] = settings.text_api_token
    kwargs: dict[str, Any] = {
        "id": server_id,
        "mcp_server": mcp.MCPServerHTTP(
            url=url,
            headers=headers or None,
            client_session_timeout_seconds=float(entry.get("timeout", 5)),
            allowed_tools=entry.get("allowed_tools"),
            tool_result_resolver=ha.text_result_resolver,
        ),
        # a tool that reports progress runs on in the background, where a force stop
        # is the only thing that ends it
        "tool_options": {
            name: mcp.MCPToolOptions(report_progress=True, flags=ToolFlag.CANCELLABLE)
            for name in entry.get("report_progress") or []
        },
    }
    adapter = entry.get("per_person")
    if adapter and adapter not in _PER_PERSON:
        raise ValueError(f"mcp server {server_id!r}: unknown per_person {adapter!r}")
    confirm = list(entry.get("confirm") or [])
    if adapter and (restricted or confirm):
        raise ValueError(
            f"mcp server {server_id!r}: per_person cannot be restricted or confirmed"
        )
    # no one in particular gets the plain server, e.g. Mem0's default user scope
    if adapter and user_id:
        return _PER_PERSON[adapter](user_id=user_id, **kwargs)
    if restricted or confirm:
        return Guarded(
            user_id=user_id if restricted else None, confirm=confirm, **kwargs
        )
    return mcp.MCPToolset(**kwargs)


def toolsets(user_id: str | None) -> list[Toolset]:
    """The configured servers this session gets; none without a config file."""
    path = Path(settings.mcp_config)
    if not path.exists():
        return []
    entries = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("servers")
    found = [_toolset(entry, user_id) for entry in entries or []]
    return [t for t in found if t is not None]
