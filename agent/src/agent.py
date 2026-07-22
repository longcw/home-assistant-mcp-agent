"""The HomeAssistantAgent: HA MCP tools, live-state helpers, and scheduling tools."""

import json
import logging
import time
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
import yaml
from livekit.agents import Agent, ModelSettings, mcp
from livekit.agents.llm import ChatContext, ChatMessage, ToolContext, function_tool

import ha
import scheduler_client as scheduler
from config import LIVE_CONTEXT_TOOL, settings
from utils import current_time_text, to_aware_iso

logger = logging.getLogger("ha-mcp-agent")


def load_instructions() -> str:
    """Read the agent's system prompt from the prompt file (YAML `instructions` key).

    Read fresh per session so editing the mounted prompt file applies to the next
    conversation without rebuilding the image.
    """
    with open(settings.prompt_file, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    instructions = data.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise RuntimeError(
            f"no non-empty 'instructions' in prompt file {settings.prompt_file!r}"
        )
    return instructions.strip()


class HomeAssistantAgent(Agent):
    """A LiveKit agent that controls Home Assistant via its native MCP server.

    The HA MCP tools (HassTurnOn, GetLiveContext, ...) are exposed to the LLM through an
    MCPToolset. On top of that, a few function tools pre-process the live state into
    compact, area/domain-filtered views to keep the LLM context small, and scheduling
    tools defer actions to the scheduler service.
    """

    def __init__(self) -> None:
        toolset = mcp.MCPToolset(
            id="home-assistant",
            mcp_server=mcp.MCPServerHTTP(
                url=ha.mcp_url(),
                headers={"Authorization": f"Bearer {settings.ha_token}"},
                tool_result_resolver=ha.text_result_resolver,
            ),
        )
        super().__init__(instructions=load_instructions(), tools=[toolset])

        self._mcp_toolset = toolset
        self._devices: pd.DataFrame | None = None
        self._devices_updated_at: float = 0
        self._devices_timeout_interval = 30
        # Set by the interactive entrypoint to publish quick replies to the card; None
        # (e.g. headless runs) makes suggest_replies a no-op.
        self._suggest_replies_cb: Callable[[list[str]], None] | None = None

    def llm_node(self, chat_ctx: ChatContext, tools, model_settings: ModelSettings):
        # Inject the current time fresh each turn, before the last user message.
        # Keeps relative-time resolution live without a stale time in the cached
        # system prompt. chat_ctx is a throwaway copy, so this edit isn't persisted.
        items = chat_ctx.items
        idx = next(
            (
                i
                for i in reversed(range(len(items)))
                if getattr(items[i], "role", None) == "user"
            ),
            len(items),
        )
        text = current_time_text(settings.agent_tz)
        items.insert(idx, ChatMessage(role="system", content=[text]))
        return Agent.default.llm_node(self, chat_ctx, tools, model_settings)

    async def tool_context(self) -> ToolContext:
        """All callable tools (agent function tools + HA MCP tools) in one context."""
        await self._mcp_toolset.setup()  # no-op if already connected
        return ToolContext(self.tools)

    @function_tool
    async def get_areas(self) -> list[str]:
        """Get all areas in the home"""
        logger.info("get_areas")
        devices = await self.get_home_state()
        return self._unique(devices, "areas")

    @function_tool
    async def get_device_domains(self) -> list[str]:
        """Get all device domains in the home"""
        logger.info("get_device_domains")
        devices = await self.get_home_state()
        return self._unique(devices, "domain")

    @function_tool
    async def get_devices(self, area: str | list[str]) -> str:
        """Get devices status in the area or areas

        Args:
            area: The area or list of areas to get devices from.
        """
        logger.info(f"get_devices: {area}")

        devices = await self.get_home_state(force_update=True)
        if isinstance(area, str):
            area = [area]
        area = [a.strip() for a in area]
        if "areas" in devices:
            df = devices[devices["areas"].isin(area)]
        else:
            df = devices.iloc[0:0]

        logger.info(f"found {len(df)} devices in {area}")
        if len(df) == 0:
            areas = self._unique(devices, "areas")
            return (
                f"No devices found in {area}, available areas: {areas}, "
                "try to use the current area name"
            )
        return self._df_to_yaml(df)

    @function_tool
    async def get_environment_info(self) -> str:
        """Get the current environment information like temperature, humidity, etc."""
        logger.info("get_environment_info")
        devices = await self.get_home_state(force_update=True)
        return self._df_to_yaml(devices[devices["domain"] == "sensor"])

    @function_tool
    async def suggest_replies(self, replies: list[str]) -> None:
        """Offer up to ~3 one-tap quick replies for your last question, e.g. Yes / No.

        Call this when you ask a yes/no or short-choice question — especially when
        confirming a schedule. Tapping a chip sends that text as the user's reply, so
        phrase each option in the user's language as a natural reply.
        """
        logger.info("suggest_replies: %s", replies)
        if self._suggest_replies_cb:
            self._suggest_replies_cb(replies)
        return None

    # --- Scheduling tools ---
    # These call the scheduler service. Their JSON results flow to the card over the
    # existing tool-execution feed (ha.tool_call), so the UI renders the task list.

    @function_tool
    async def schedule_task(
        self,
        description: str,
        schedule_type: Literal["once", "recurring"],
        execution_type: Literal["instruction", "function_call"],
        run_at: str | None = None,
        cron: str | None = None,
        instruction: str | None = None,
        tool_name: str | None = None,
        tool_args_json: str | None = None,
    ) -> str:
        """Schedule a task to run later, once or on a recurring schedule.

        ALWAYS confirm the resolved time and action with the user before calling this.

        Args:
            description: Short summary, e.g. "Turn off the master bedroom AC".
            schedule_type: "once" or "recurring".
            execution_type: "instruction" (you re-interpret it at run time) or
                "function_call" (a specific tool is replayed exactly). Prefer
                "function_call" for a concrete device action, "instruction" when
                the action needs judgement at run time.
            run_at: For "once": absolute local time in ISO 8601, e.g.
                "2026-07-22T17:30". Resolve relative times yourself.
            cron: For "recurring": a 5-field cron expression, e.g. "0 8 * * 1-5".
            instruction: For "instruction": the natural-language instruction.
            tool_name: For "function_call": one of your available tools.
            tool_args_json: For "function_call": that tool's args as JSON.
        """
        logger.info("schedule_task: %s [%s]", description, schedule_type)
        try:
            if schedule_type == "once":
                if not run_at:
                    return "Error: run_at is required for a one-time task."
                schedule = {
                    "type": "once",
                    "run_at": to_aware_iso(run_at, settings.agent_tz),
                    "timezone": settings.agent_tz,
                }
            elif schedule_type == "recurring":
                if not cron:
                    return "Error: cron is required for a recurring task."
                schedule = {
                    "type": "recurring",
                    "cron": cron,
                    "timezone": settings.agent_tz,
                }
            else:
                return f"Error: unknown schedule_type {schedule_type!r}."

            if execution_type == "instruction":
                if not instruction:
                    return "Error: instruction is required for an instruction task."
                execution = {"type": "instruction", "text": instruction}
            elif execution_type == "function_call":
                if not tool_name:
                    return "Error: tool_name is required for a function_call task."
                ctx = await self.tool_context()
                if tool_name not in ctx.function_tools:
                    valid = ", ".join(sorted(ctx.function_tools))
                    return f"Error: unknown tool {tool_name!r}. Available: {valid}"
                args = json.loads(tool_args_json) if tool_args_json else {}
                execution = {"type": "function_call", "tool": tool_name, "args": args}
            else:
                return f"Error: unknown execution_type {execution_type!r}."

            task = await scheduler.create_task(
                {
                    "description": description,
                    "schedule": schedule,
                    "execution": execution,
                }
            )
            logger.info("scheduled task %s", task.get("id"))
            return json.dumps(task, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - surface a message for the LLM to relay
            logger.exception("schedule_task failed")
            return f"Error scheduling task: {exc}"

    @function_tool
    async def list_scheduled_tasks(self) -> str:
        """List the currently scheduled (active) tasks, soonest first."""
        logger.info("list_scheduled_tasks")
        try:
            tasks = await scheduler.list_tasks(active_only=True)
            return json.dumps(tasks, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_scheduled_tasks failed")
            return f"Error listing scheduled tasks: {exc}"

    @function_tool
    async def cancel_scheduled_task(self, task_id: str) -> str:
        """Cancel (remove) a scheduled task by its id (get the id from
        list_scheduled_tasks)."""
        logger.info("cancel_scheduled_task: %s", task_id)
        try:
            task = await scheduler.delete_task(task_id)
            return json.dumps(task, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("cancel_scheduled_task failed")
            return f"Error cancelling task: {exc}"

    @function_tool
    async def update_scheduled_task(
        self,
        task_id: str,
        description: str | None = None,
        run_at: str | None = None,
        cron: str | None = None,
        enabled: bool | None = None,
    ) -> str:
        """Modify a scheduled task: change its time (run_at or cron), description, or
        whether it is enabled. Confirm changes with the user first."""
        logger.info("update_scheduled_task: %s", task_id)
        try:
            payload: dict[str, Any] = {}
            if description is not None:
                payload["description"] = description
            if enabled is not None:
                payload["enabled"] = enabled
            if run_at is not None:
                payload["schedule"] = {
                    "type": "once",
                    "run_at": to_aware_iso(run_at, settings.agent_tz),
                    "timezone": settings.agent_tz,
                }
            elif cron is not None:
                payload["schedule"] = {
                    "type": "recurring",
                    "cron": cron,
                    "timezone": settings.agent_tz,
                }
            if not payload:
                return "Error: nothing to update."
            task = await scheduler.update_task(task_id, payload)
            return json.dumps(task, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("update_scheduled_task failed")
            return f"Error updating task: {exc}"

    async def get_home_state(self, force_update: bool = False) -> pd.DataFrame:
        if (
            not force_update
            and self._devices is not None
            and time.time() - self._devices_updated_at < self._devices_timeout_interval
        ):
            return self._devices

        raw = await self._call_mcp_tool(LIVE_CONTEXT_TOOL)
        # GetLiveContext returns {"success": true, "result": "<prose>\n- names: ..."}
        data = json.loads(raw)
        result_text = data.get("result", "")
        # drop the human-readable prose prefix line before the YAML device list
        _, _, list_text = result_text.partition("\n")
        devices = yaml.safe_load(list_text) or []

        self._devices = pd.DataFrame(devices)
        self._devices_updated_at = time.time()
        return self._devices

    async def _call_mcp_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        """Invoke a single MCP tool by name (used for GetLiveContext)."""
        ctx = await self.tool_context()
        if fnc_tool := ctx.get_function_tool(name):
            return await fnc_tool(arguments or {})
        raise RuntimeError(f"MCP tool {name!r} is not available on the server")

    @staticmethod
    def _unique(df: pd.DataFrame, column: str) -> list[str]:
        if column not in df:
            return []
        return df[column].dropna().unique().tolist()

    @staticmethod
    def _df_to_yaml(df: pd.DataFrame) -> str:
        """Serialize device rows as YAML.

        Compact and readable for the LLM, and the frontend parses this same output to
        render the status cards, so it flows over the one tool-execution event.
        """
        return yaml.dump(list(df.to_dict(orient="index").values()), allow_unicode=True)
