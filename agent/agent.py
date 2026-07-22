import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
import yaml
from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    ToolExecutionUpdatedEvent,
    TurnHandlingOptions,
    cli,
    inference,
    mcp,
)
from livekit.agents.llm import ToolContext, function_tool
from mcp.types import TextContent

logger = logging.getLogger("ha-mcp-agent")

load_dotenv()


# Models are served by LiveKit Inference (authenticated via LIVEKIT_INFERENCE_* env
# vars, falling back to LIVEKIT_*). Override any of these from the environment.
# For a native Chinese TTS voice, browse https://play.cartesia.ai/voices and set
# TTS_VOICE.
STT_MODEL = os.getenv("STT_MODEL", "assemblyai/universal-3-5-pro")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "multi")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-31b-it")
TTS_MODEL = os.getenv("TTS_MODEL", "fishaudio/s2.1-pro")
TTS_VOICE = os.getenv("TTS_VOICE", "5c353fdb312f4888836a9a5680099ef0")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "")

# Explicit-dispatch name; the frontend dispatches this worker by name.
AGENT_NAME = os.getenv("AGENT_NAME", "ha-agent")

# The system prompt lives in a separate file (YAML: `instructions: |`) so it can be
# tweaked without rebuilding the image: it is bind-mounted into the container (see
# docker-compose.yml) and re-read on each session. Override the path with PROMPT_FILE.
PROMPT_FILE = os.getenv(
    "PROMPT_FILE", os.path.join(os.path.dirname(__file__), "prompt.yaml")
)

# STT is billed continuously, so it follows the mic: enabled whenever audio input is
# live, and torn down this many seconds after the mic is gated. The grace period avoids
# re-initialising STT on quick successive turns. Applies uniformly to manual and auto.
STT_IDLE_TIMEOUT = float(os.getenv("STT_IDLE_TIMEOUT", "120"))

# The Home Assistant MCP Server integration exposes a Streamable HTTP endpoint at
# /api/mcp — see https://www.home-assistant.io/integrations/mcp_server/
HA_MCP_PATH = "/api/mcp"

# Tool exposed by Home Assistant that returns the live state of all exposed entities.
LIVE_CONTEXT_TOOL = "GetLiveContext"

# Data-channel topic the frontend listens on (the HA integration card,
# card/src/lib/tool-feed.ts). It carries the tool-execution lifecycle so the UI can
# render tool cards; the state tools return YAML so the same stream also powers the
# device/sensor status cards.
TOOL_CALL_TOPIC = "ha.tool_call"

# Data-channel topic carrying live session state so the frontend can mirror it: whether
# STT is enabled (listening vs sleeping) and whether the agent speaks (TTS).
# Payload: {"stt_enabled": bool, "audio_output": bool}.
SESSION_STATE_TOPIC = "ha.speech_state"

# Keep forwarded tool outputs under LiveKit's data-packet size budget. Large enough that
# a normal home-state YAML payload stays intact (and stays parseable) for the UI cards.
MAX_TOOL_OUTPUT_CHARS = 12000

# --- Scheduling ---
# The scheduler service (docker-compose) persists tasks and, at fire time, dispatches
# this worker back into a headless room. The scheduling function tools call its REST
# API, and the headless branch (run_scheduled_task) reports each run's outcome back.
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "http://scheduler:8080")

# Home timezone: tells the LLM the current local time (so it can resolve "in 1 hour")
# and stamps scheduled times with an unambiguous offset. Falls back to TZ, then UTC.
AGENT_TZ = os.getenv("AGENT_TZ") or os.getenv("TZ") or "UTC"

# Upper bound on a single headless scheduled execution before it's abandoned as failed.
SCHEDULED_RUN_TIMEOUT = float(os.getenv("SCHEDULED_RUN_TIMEOUT", "120"))

# Shared secret for the scheduler's HTTP API (see docker-compose / .env). Sent as a
# bearer token; empty means the scheduler runs without auth (purely internal).
SCHEDULER_TOKEN = os.getenv("SCHEDULER_TOKEN", "")


def home_assistant_mcp_url() -> str:
    """Build the Home Assistant MCP endpoint from HA_URL.

    e.g. https://ha.example.com -> https://ha.example.com/api/mcp
    """
    base = os.environ["HA_URL"].rstrip("/")
    return f"{base}{HA_MCP_PATH}"


def _text_result_resolver(ctx: mcp.MCPToolResultContext) -> str:
    """Return MCP tool results as plain text (HA returns a single text block).

    This keeps results readable for the LLM and lets our function tools parse the
    payload directly instead of unwrapping the default JSON envelope.
    """
    parts = [c.text for c in ctx.result.content if isinstance(c, TextContent)]
    if parts:
        return "\n".join(parts)
    return json.dumps([item.model_dump() for item in ctx.result.content])


def load_instructions() -> str:
    """Read the agent's system prompt from PROMPT_FILE (a YAML `instructions` key).

    Read fresh per session so editing the mounted prompt file applies to the next
    conversation without rebuilding the image.
    """
    with open(PROMPT_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    instructions = data.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise RuntimeError(
            f"no non-empty 'instructions' in prompt file {PROMPT_FILE!r}"
        )
    return instructions.strip()


async def _scheduler_request(
    method: str, path: str, payload: dict | None = None
) -> Any:
    """Call the scheduler service; raise RuntimeError with the detail on a 4xx/5xx."""
    url = f"{SCHEDULER_URL.rstrip('/')}{path}"
    headers = (
        {"Authorization": f"Bearer {SCHEDULER_TOKEN}"} if SCHEDULER_TOKEN else None
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


def _to_aware_iso(run_at: str) -> str:
    """Normalise a local datetime to an offset-aware ISO string in AGENT_TZ."""
    dt = datetime.fromisoformat(run_at.strip().replace(" ", "T"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(AGENT_TZ))
    return dt.isoformat()


class HomeAssistantAgent(Agent):
    """A LiveKit agent that controls Home Assistant via its native MCP server.

    The Home Assistant MCP tools (HassTurnOn, GetLiveContext, ...) are exposed to the
    LLM through an ``MCPToolset``. On top of that, a few function tools pre-process the
    live state into compact, area/domain-filtered views to keep the LLM context small.
    """

    def __init__(self) -> None:
        toolset = mcp.MCPToolset(
            id="home-assistant",
            mcp_server=mcp.MCPServerHTTP(
                url=home_assistant_mcp_url(),
                headers={"Authorization": f"Bearer {os.environ['HA_TOKEN']}"},
                tool_result_resolver=_text_result_resolver,
            ),
        )
        super().__init__(instructions=self._build_instructions(), tools=[toolset])

        self._mcp_toolset = toolset
        self._devices: pd.DataFrame | None = None
        self._devices_updated_at: float = 0
        self._devices_timeout_interval = 30

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

    # --- Scheduling tools ---
    # These call the scheduler service. Their return values (JSON) flow to the card over
    # the existing tool-execution feed (ha.tool_call), so the UI renders the scheduled
    # task card and the task list from the tool output. No new data channel.

    @staticmethod
    def _build_instructions() -> str:
        """System prompt + the current local time (so relative schedules resolve)."""
        base = load_instructions()
        now = datetime.now(ZoneInfo(AGENT_TZ))
        return (
            f"{base}\n\n"
            "# Current time\n"
            f"The current local time is {now.isoformat()} ({AGENT_TZ}). Use it to "
            'resolve relative times such as "in 1 hour" or "tonight" into an absolute '
            "time when scheduling tasks."
        )

    @function_tool
    async def schedule_task(
        self,
        description: str,
        schedule_type: str,
        execution_type: str,
        run_at: str | None = None,
        cron: str | None = None,
        command_text: str | None = None,
        tool_name: str | None = None,
        tool_args_json: str | None = None,
    ) -> str:
        """Schedule a task to run later, once or on a recurring schedule.

        ALWAYS confirm the resolved time and action with the user before calling this.

        Args:
            description: Short summary, e.g. "Turn off the master bedroom AC".
            schedule_type: "once" or "recurring".
            execution_type: "command" (you re-interpret the instruction at run time) or
                "function_call" (a specific Home Assistant tool is replayed exactly).
                Prefer "function_call" for a concrete device action, "command" for
                anything that needs judgement at run time.
            run_at: For "once": the absolute local time in ISO 8601, e.g.
                "2026-07-22T17:30". Resolve relative times ("in 1 hour", "tonight")
                to an absolute time yourself.
            cron: For "recurring": a 5-field cron expression, e.g. "0 8 * * 1-5"
                (08:00 on weekdays).
            command_text: For "command": the natural-language instruction.
            tool_name: For "function_call": the Home Assistant tool to call.
            tool_args_json: For "function_call": that tool's args as a JSON string.
        """
        logger.info("schedule_task: %s [%s]", description, schedule_type)
        try:
            if schedule_type == "once":
                if not run_at:
                    return "Error: run_at is required for a one-time task."
                schedule = {
                    "type": "once",
                    "run_at": _to_aware_iso(run_at),
                    "timezone": AGENT_TZ,
                }
            elif schedule_type == "recurring":
                if not cron:
                    return "Error: cron is required for a recurring task."
                schedule = {"type": "recurring", "cron": cron, "timezone": AGENT_TZ}
            else:
                return f"Error: unknown schedule_type {schedule_type!r}."

            if execution_type == "command":
                if not command_text:
                    return "Error: command_text is required for a command task."
                execution = {"type": "command", "text": command_text}
            elif execution_type == "function_call":
                if not tool_name:
                    return "Error: tool_name is required for a function_call task."
                args = json.loads(tool_args_json) if tool_args_json else {}
                execution = {"type": "function_call", "tool": tool_name, "args": args}
            else:
                return f"Error: unknown execution_type {execution_type!r}."

            task = await _scheduler_request(
                "POST",
                "/tasks",
                {
                    "description": description,
                    "schedule": schedule,
                    "execution": execution,
                },
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
            tasks = await _scheduler_request("GET", "/tasks?active_only=true")
            return json.dumps(tasks, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_scheduled_tasks failed")
            return f"Error listing scheduled tasks: {exc}"

    @function_tool
    async def cancel_scheduled_task(self, task_id: str) -> str:
        """Cancel a scheduled task by its id (get the id from list_scheduled_tasks)."""
        logger.info("cancel_scheduled_task: %s", task_id)
        try:
            task = await _scheduler_request("DELETE", f"/tasks/{task_id}")
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
                    "run_at": _to_aware_iso(run_at),
                    "timezone": AGENT_TZ,
                }
            elif cron is not None:
                payload["schedule"] = {
                    "type": "recurring",
                    "cron": cron,
                    "timezone": AGENT_TZ,
                }
            if not payload:
                return "Error: nothing to update."
            task = await _scheduler_request("PATCH", f"/tasks/{task_id}", payload)
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
        """Invoke an MCP tool through the toolset (not the private client)."""
        await self._mcp_toolset.setup()  # no-op if already connected

        tool_ctx = ToolContext(self._mcp_toolset.tools)
        if fnc_tool := tool_ctx.get_function_tool(name):
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

        YAML is compact and readable for the LLM, and the frontend parses this same
        output to render the status cards (see get_environment_info / get_devices),
        so it flows over the one tool-execution event with no extra data channel.
        """
        return yaml.dump(list(df.to_dict(orient="index").values()), allow_unicode=True)


server = AgentServer()


def _forward_tool_events(ctx: JobContext, session: AgentSession) -> None:
    """Stream the tool-execution lifecycle to the frontend over a data channel.

    Each ``tool_execution_updated`` event (started / updated / ended) is serialized
    and published on ``TOOL_CALL_TOPIC``. A single background consumer preserves
    ordering, and large tool outputs are truncated to stay within the data-packet size
    budget, modeled on the upstream ``async_tool_agent`` example.
    """
    queue: asyncio.Queue[ToolExecutionUpdatedEvent] = asyncio.Queue()

    @session.on("tool_execution_updated")
    def _on_tool_execution_updated(ev: ToolExecutionUpdatedEvent) -> None:
        queue.put_nowait(ev)

    async def _pump() -> None:
        while True:
            ev = await queue.get()
            data = ev.model_dump(mode="json")
            update = data.get("update", {})
            message = update.get("message")
            if isinstance(message, str) and len(message) > MAX_TOOL_OUTPUT_CHARS:
                update["message"] = message[:MAX_TOOL_OUTPUT_CHARS] + "…"
            try:
                await ctx.room.local_participant.publish_data(
                    json.dumps(data, ensure_ascii=False),
                    topic=TOOL_CALL_TOPIC,
                    reliable=True,
                )
            except Exception:
                logger.exception("failed to publish tool event")

    task = asyncio.create_task(_pump())

    async def _cancel_pump() -> None:
        task.cancel()

    ctx.add_shutdown_callback(_cancel_pump)


# --- Headless scheduled execution ---
# When the scheduler service fires a task it dispatches this worker with
# `kind: "scheduled"` job metadata. The entrypoint routes that to run_scheduled_task,
# which executes the task with no mic / STT / TTS / user, reports the outcome to the
# scheduler, notifies Home Assistant, and tears the room down.


def _parse_job_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


async def _notify_ha(title: str, message: str) -> None:
    """Raise a Home Assistant persistent notification (best-effort)."""
    try:
        base = os.environ["HA_URL"].rstrip("/")
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{base}/api/services/persistent_notification/create",
                headers={"Authorization": f"Bearer {os.environ['HA_TOKEN']}"},
                json={"title": title, "message": message},
            )
            resp.raise_for_status()
    except Exception:
        logger.exception("failed to post HA notification")


async def _report_run(run_id: str, status: str, result: str) -> None:
    if not run_id:
        return
    try:
        await _scheduler_request(
            "POST", f"/internal/runs/{run_id}", {"status": status, "result": result}
        )
    except Exception:
        logger.exception("failed to report run %s outcome", run_id)


async def _run_command_turn(
    ctx: JobContext, agent: "HomeAssistantAgent", text: str
) -> str:
    """Execute a natural-language command as one text-only LLM turn (no STT/TTS)."""
    session = AgentSession(llm=inference.LLM(LLM_MODEL), max_tool_steps=8)
    replies: list[str] = []

    @session.on("conversation_item_added")
    def _collect(ev: Any) -> None:
        item = getattr(ev, "item", None)
        if getattr(item, "role", None) == "assistant":
            content = getattr(item, "text_content", None)
            if content:
                replies.append(content)

    await session.start(agent=agent, room=ctx.room)
    # Text-only: no user audio in, and no TTS. Audio output defaults on, so without this
    # the framework invokes tts_node and errors (no TTS is configured).
    try:
        session.input.set_audio_enabled(False)
        session.output.set_audio_enabled(False)
    except Exception:  # noqa: BLE001 - best effort; there is no audio peer anyway
        pass

    async def _await_reply() -> None:
        await session.generate_reply(user_input=text)

    try:
        await asyncio.wait_for(_await_reply(), timeout=SCHEDULED_RUN_TIMEOUT)
    finally:
        await session.aclose()
    return replies[-1] if replies else "Done."


async def run_scheduled_task(ctx: JobContext, meta: dict[str, Any]) -> None:
    task_id = meta.get("task_id", "")
    run_id = meta.get("run_id", "")
    execution = meta.get("execution") or {}
    description = meta.get("description") or "scheduled task"
    logger.info("running scheduled task %s (run %s): %s", task_id, run_id, description)

    await ctx.connect()
    agent = HomeAssistantAgent()
    status = "success"
    result = ""
    try:
        etype = execution.get("type")
        if etype == "function_call":
            args = execution.get("args") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            result = await asyncio.wait_for(
                agent._call_mcp_tool(execution["tool"], args),
                timeout=SCHEDULED_RUN_TIMEOUT,
            )
        elif etype == "command":
            result = await _run_command_turn(ctx, agent, execution.get("text") or "")
        else:
            raise ValueError(f"unknown execution type {etype!r}")
    except Exception as exc:  # noqa: BLE001 - any failure is recorded + notified
        status = "error"
        result = str(exc)
        logger.exception("scheduled task %s failed", task_id)

    result = (result or "").strip()[:MAX_TOOL_OUTPUT_CHARS]
    await _report_run(run_id, status, result)
    if status == "success":
        await _notify_ha("Scheduled task done", f"{description}\n\n{result}".strip())
    else:
        message = f"{description}\n\nError: {result}".strip()
        await _notify_ha("Scheduled task failed", message)

    try:
        await ctx.delete_room()
    except Exception:
        logger.exception("failed to delete room after scheduled task")


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    # A scheduled dispatch (from the scheduler service) runs headlessly and exits: no
    # mic, STT, TTS, or user. Everything below this guard is the interactive path.
    meta = _parse_job_metadata(ctx.job.metadata)
    if meta.get("kind") == "scheduled":
        await run_scheduled_task(ctx, meta)
        return

    # Detector reused whenever we switch to auto; STT instance held so we can detach it
    # (stt=None) and rewire the same object later via Agent.update_options. The VAD is
    # the session's bundled Silero default and stays live throughout (local, unbilled).
    turn_detector = inference.TurnDetector()
    stt = inference.STT(STT_MODEL, language=STT_LANGUAGE)

    agent = HomeAssistantAgent()
    session = AgentSession(
        stt=stt,
        llm=inference.LLM(LLM_MODEL),
        tts=inference.TTS(TTS_MODEL, voice=TTS_VOICE, language=TTS_LANGUAGE),
        turn_handling=TurnHandlingOptions(turn_detection="manual"),
        max_tool_steps=8,
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

    _forward_tool_events(ctx, session)

    # --- Session state the frontend mirrors, and the controls that mutate it. ---
    #
    # Deliberately simple: STT is billed continuously, so it follows the mic (see
    # _set_audio_input) — live while audio input is on, torn down after STT_IDLE_TIMEOUT
    # once gated. TTS (audio output) and text chat are independent, so the agent can run
    # as pure text with zero speech cost. The card drives it all over RPCs; the agent
    # boots dormant + text-only so an idle card costs only its connection. Same in both
    # manual and auto modes.
    stt_enabled = True  # session boots with STT wired; torn down in the boot below
    audio_output_enabled = True  # session default; muted at boot
    stt_timer: asyncio.TimerHandle | None = None
    publish_tasks: set[asyncio.Task[None]] = set()

    def _publish_state() -> None:
        payload = json.dumps(
            {"stt_enabled": stt_enabled, "audio_output": audio_output_enabled}
        )

        async def _send() -> None:
            try:
                await ctx.room.local_participant.publish_data(
                    payload, topic=SESSION_STATE_TOPIC, reliable=True
                )
            except Exception:
                logger.exception("failed to publish session state")

        task = asyncio.create_task(_send())
        publish_tasks.add(task)
        task.add_done_callback(publish_tasks.discard)

    def _cancel_stt_timer() -> None:
        nonlocal stt_timer
        if stt_timer is not None:
            stt_timer.cancel()
            stt_timer = None

    def _enable_stt() -> None:
        nonlocal stt_enabled
        _cancel_stt_timer()
        if stt_enabled:
            return
        agent.update_options(stt=stt)
        stt_enabled = True
        logger.info("STT enabled")
        _publish_state()

    def _disable_stt() -> None:
        nonlocal stt_enabled
        _cancel_stt_timer()
        if not stt_enabled:
            return
        agent.update_options(stt=None)
        stt_enabled = False
        logger.info("STT disabled (mic idle) to save cost")
        _publish_state()

    def _set_audio_input(enabled: bool) -> None:
        """Gate the mic and tie STT to it: enabled while listening, and scheduled for
        teardown STT_IDLE_TIMEOUT after the mic goes quiet (a grace period so quick,
        successive turns don't re-initialise STT)."""
        nonlocal stt_timer
        if enabled:
            _enable_stt()  # before opening the mic so STT is ready for the first words
            session.input.set_audio_enabled(True)
        else:
            session.input.set_audio_enabled(False)
            _cancel_stt_timer()
            if stt_enabled:
                loop = asyncio.get_running_loop()
                stt_timer = loop.call_later(STT_IDLE_TIMEOUT, _disable_stt)

    def _set_audio_output(enabled: bool) -> None:
        """Toggle spoken (TTS) replies. Text replies are unaffected."""
        nonlocal audio_output_enabled
        if enabled == audio_output_enabled:
            return
        session.output.set_audio_enabled(enabled)
        audio_output_enabled = enabled
        logger.info("audio output %s", "enabled" if enabled else "disabled")
        _publish_state()

    async def _set_can_subscribe(identity: str, allow: bool) -> None:
        """Grant/revoke a participant's track-subscribe permission at runtime.

        The card connects with can_subscribe=False so an idle/text connection has no
        receive-audio transceiver (which on iOS grabs the audio session and stops the
        user's music). We allow subscribing only while spoken replies are on.
        """
        if not identity:
            return
        try:
            await ctx.api.room.update_participant(
                api.UpdateParticipantRequest(
                    room=ctx.room.name,
                    identity=identity,
                    permission=api.ParticipantPermission(
                        can_publish=True,
                        can_publish_data=True,
                        can_subscribe=allow,
                    ),
                )
            )
        except Exception:
            logger.exception("failed to update subscribe permission for %s", identity)

    def apply_mode(manual: bool) -> None:
        """Switch turn detection and gate the mic to match. STT follows the mic."""
        session.update_options(turn_detection="manual" if manual else turn_detector)
        if manual:
            session.clear_user_turn()
            _set_audio_input(False)  # idle until start_turn opens a turn
        else:
            _set_audio_input(True)  # auto: mic stays live so the model can detect turns

    # Boot dormant + text-only: mic gated, STT torn down now (no idle wait), TTS muted.
    session.input.set_audio_enabled(False)
    _disable_stt()
    _set_audio_output(False)
    _publish_state()

    @ctx.room.on("participant_connected")
    def _on_participant_connected(_participant: rtc.RemoteParticipant) -> None:
        # Re-assert state so a frontend that joins/reconnects mid-session sees it.
        _publish_state()

    async def _on_shutdown() -> None:
        _cancel_stt_timer()

    ctx.add_shutdown_callback(_on_shutdown)

    @ctx.room.local_participant.register_rpc_method("set_turn_mode")
    async def set_turn_mode(data: rtc.RpcInvocationData) -> str:
        manual = data.payload == "manual"
        logger.info("set turn mode: %s", "manual" if manual else "auto")
        apply_mode(manual)
        return "ok"

    @ctx.room.local_participant.register_rpc_method("set_audio_output")
    async def set_audio_output(data: rtc.RpcInvocationData) -> str:
        # payload "on"/"off" toggles spoken (TTS) replies; text replies still work.
        # Grant subscribe before enabling TTS; revoke after disabling it, so the client
        # only holds the audio session while replies play. No reconnect either way.
        on = data.payload == "on"
        if on:
            await _set_can_subscribe(data.caller_identity, True)
        _set_audio_output(on)
        if not on:
            await _set_can_subscribe(data.caller_identity, False)
        return "ok"

    @ctx.room.local_participant.register_rpc_method("start_turn")
    async def start_turn(data: rtc.RpcInvocationData) -> str:
        session.interrupt()
        session.clear_user_turn()
        # listen only to the participant who started the turn (multi-user rooms)
        session.room_io.set_participant(data.caller_identity)
        _set_audio_input(True)  # opens the mic and (re)enables STT
        return "ok"

    @ctx.room.local_participant.register_rpc_method("end_turn")
    async def end_turn(data: rtc.RpcInvocationData) -> str:
        _set_audio_input(False)  # gate the mic; STT tears down after the idle timeout
        session.commit_user_turn()
        return "ok"

    @ctx.room.local_participant.register_rpc_method("cancel_turn")
    async def cancel_turn(data: rtc.RpcInvocationData) -> str:
        _set_audio_input(False)
        session.clear_user_turn()
        logger.info("cancel turn")
        return "ok"


if __name__ == "__main__":
    cli.run_app(server)
