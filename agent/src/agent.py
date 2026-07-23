"""The HomeAssistantAgent: HA MCP tools, live-state helpers, and scheduling tools."""

import json
import logging
import time
from collections.abc import Callable
from typing import Any, Literal

import pandas as pd
import yaml
from livekit.agents import Agent, ModelSettings, mcp
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
    ToolContext,
    ToolError,
    function_tool,
)
from pydantic import BaseModel, Field

import ha
import scheduler_client as scheduler
from config import LIVE_CONTEXT_TOOL, settings
from utils import current_time_text, to_aware_iso

logger = logging.getLogger("ha-mcp-agent")


class ToolCall(BaseModel):
    """One deterministic tool call in a scheduled task: a tool name + its arguments."""

    tool: str = Field(
        description="Tool to call — one of your available tools, e.g. 'HassTurnOn'."
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for the tool, e.g. {'name': '主卧 空调'}.",
    )


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
        This MUST be called only after the question is asked, not before.
        """
        logger.info("suggest_replies: %s", replies)
        if self._suggest_replies_cb:
            self._suggest_replies_cb(replies)
        return None

    @function_tool
    async def send_notification(self, message: str, title: str | None = None) -> str:
        """Send the user a notification (Home Assistant + their chosen devices).

        Use to reach the user proactively, e.g. from a scheduled task. Write in the
        user's language.

        Args:
            message: The notification body.
            title: Optional short title.
        """
        logger.info("send_notification: %s", message)
        targets = await scheduler.notify_targets()
        ok = await ha.notify(message, title=title, targets=targets)
        return "Notification sent." if ok else "Failed to send the notification."

    # --- Scheduling tools ---
    # These call the scheduler service. Their JSON results flow to the card over the
    # existing tool-execution feed (ha.tool_call), so the UI renders the task list.

    @function_tool
    async def schedule_task(
        self,
        description: str,
        schedule_type: Literal["once", "recurring"],
        run_at: str | None = None,
        cron: str | None = None,
        steps: list[ToolCall] | None = None,
        instruction: str | None = None,
    ) -> str:
        """Schedule a task to run later, once or on a recurring schedule.

        ALWAYS confirm the resolved time and action with the user before calling this.

        A task carries `steps` (concrete tool calls replayed exactly, in order) and/or
        an `instruction` (natural language run at fire time). Provide at least one:

        - Use `steps` for concrete, deterministic device actions you can pin down now.
          Pass MULTIPLE steps when the request needs several actions — e.g. "turn on
          the fan and set it to 50%" is two steps. Steps run in order and stop at the
          first failure.
        - Use `instruction` when the task needs judgement at run time or a natural
          language answer (e.g. "tell me tomorrow's weather"). It runs after any steps,
          sees their results, and may call more tools.
        - Combine both to guarantee an action AND report on it.

        Args:
            description: Short summary, e.g. "Turn off the master bedroom AC".
            schedule_type: "once" or "recurring".
            run_at: For "once": absolute local time in ISO 8601, e.g.
                "2026-07-22T17:30". Resolve relative times yourself.
            cron: For "recurring": a 5-field cron expression, e.g. "0 8 * * 1-5".
            steps: Ordered tool calls to replay at run time; each is a tool name + its
                args, exactly as you would call it for an immediate action.
            instruction: A natural-language instruction to run at fire time.
        """
        logger.info("schedule_task: %s [%s]", description, schedule_type)
        if schedule_type == "once":
            if not run_at:
                raise ToolError("run_at is required for a one-time task.")
            schedule = {
                "type": "once",
                "run_at": to_aware_iso(run_at, settings.agent_tz),
                "timezone": settings.agent_tz,
            }
        elif schedule_type == "recurring":
            if not cron:
                raise ToolError("cron is required for a recurring task.")
            schedule = {
                "type": "recurring",
                "cron": cron,
                "timezone": settings.agent_tz,
            }
        else:
            raise ToolError(f"unknown schedule_type {schedule_type!r}.")

        steps = steps or []
        ctx = await self.tool_context()  # validate each step's tool exists
        for i, step in enumerate(steps, start=1):
            if step.tool not in ctx.function_tools:
                valid = ", ".join(sorted(ctx.function_tools))
                raise ToolError(
                    f"unknown tool {step.tool!r} in step {i}. Available: {valid}"
                )
        instruction_text = instruction.strip() if instruction else None
        if not steps and not instruction_text:
            raise ToolError("provide steps (tool calls) and/or an instruction.")
        execution = {
            "steps": [s.model_dump() for s in steps],
            "instruction": instruction_text,
        }

        try:
            task = await scheduler.create_task(
                {
                    "description": description,
                    "schedule": schedule,
                    "execution": execution,
                }
            )
        except Exception as exc:  # noqa: BLE001 - surface a message for the LLM to relay
            logger.exception("schedule_task failed")
            raise ToolError(f"could not schedule task: {exc}") from exc
        logger.info("scheduled task %s", task.get("id"))
        return json.dumps(task, ensure_ascii=False)

    @function_tool
    async def list_scheduled_tasks(self) -> str:
        """List the currently scheduled (active) tasks, soonest first."""
        logger.info("list_scheduled_tasks")
        try:
            tasks = await scheduler.list_tasks(active_only=True)
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_scheduled_tasks failed")
            raise ToolError(f"could not list scheduled tasks: {exc}") from exc
        return json.dumps(tasks, ensure_ascii=False)

    @function_tool
    async def cancel_scheduled_task(self, task_id: str) -> str:
        """Cancel (remove) a scheduled task by its id (get the id from
        list_scheduled_tasks)."""
        logger.info("cancel_scheduled_task: %s", task_id)
        try:
            task = await scheduler.delete_task(task_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("cancel_scheduled_task failed")
            raise ToolError(f"could not cancel task: {exc}") from exc
        return json.dumps(task, ensure_ascii=False)

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
                raise ToolError("nothing to update.")
            task = await scheduler.update_task(task_id, payload)
            return json.dumps(task, ensure_ascii=False)
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("update_scheduled_task failed")
            raise ToolError(f"could not update task: {exc}") from exc

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
