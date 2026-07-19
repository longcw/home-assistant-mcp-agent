import asyncio
import json
import logging
import os
import time
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from livekit import rtc
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
TTS_MODEL = os.getenv("TTS_MODEL", "cartesia/sonic-3")
TTS_VOICE = os.getenv("TTS_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "zh")

# Explicit-dispatch name; the frontend dispatches this worker by name.
AGENT_NAME = os.getenv("AGENT_NAME", "ha-agent")

# The Home Assistant MCP Server integration exposes a Streamable HTTP endpoint at
# /api/mcp — see https://www.home-assistant.io/integrations/mcp_server/
HA_MCP_PATH = "/api/mcp"

# Tool exposed by Home Assistant that returns the live state of all exposed entities.
LIVE_CONTEXT_TOOL = "GetLiveContext"

# Data-channel topic the frontend listens on (see
# frontend/hooks/use-home-assistant-feed.ts). It carries the tool-execution lifecycle
# so the UI can render tool cards; the state tools return YAML so the same stream also
# powers the device/sensor status cards.
TOOL_CALL_TOPIC = "ha.tool_call"

# Keep forwarded tool outputs under LiveKit's data-packet size budget. Large enough that
# a normal home-state YAML payload stays intact (and stays parseable) for the UI cards.
MAX_TOOL_OUTPUT_CHARS = 12000


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


instructions = """
You are a voice assistant for Home Assistant, designed to help users control their smart home devices.
You control devices by calling the tools exposed by the Home Assistant MCP server (e.g. HassTurnOn, HassTurnOff, HassLightSet).

# Device Control Guidelines
- Before controlling any device, use get_devices or GetLiveContext to confirm the exact device name and current state
- Reuse previous query results when appropriate to avoid redundant status checks
- When a device name contains a comma, use the portion after the comma (the alias) for control
- You MUST use the exact original device name from the system when calling a tool, keep the multiple spaces in the original name

# Handling Ambiguous Requests
- If the requested device cannot be found, identify and suggest similar alternatives, e.g. 书房的射灯 -> 书房照明  射灯 右键 (follow the actual device name)
- Users may mix area references in spoken language (e.g., 厨房/kitchen and 餐厅/dining room) - find the device based on name in possible areas
- When a user asks for a type of device in an area, ask which device in the area they want to control
- When presenting options to users, use natural device names for clarity
- When executing tool calls, ALWAYS use the exact original device name from the system
- When a user asks for devices in a specific area, call get_areas first and match the area name in case of ambiguity

# Communication Style
- Respond conversationally and confirm actions after completion
- Be concise but helpful in your explanations
- Acknowledge when you're checking device status or performing actions
- Use the same language as the user
- Do not use emojis, markdown, or other special characters in your spoken responses
"""  # noqa: E501


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
        super().__init__(instructions=instructions, tools=[toolset])

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


def _push_to_talk_requested(ctx: JobContext) -> bool:
    """Read the input mode the frontend requested via agent dispatch metadata.

    The frontend dispatches this agent with metadata like
    {"input_mode": "push_to_talk"}. Absent or any other value, the agent uses
    automatic turn detection.
    """
    raw = ctx.job.metadata
    if not raw:
        return False
    try:
        meta = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(meta, dict) and meta.get("input_mode") == "push_to_talk"


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


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    push_to_talk = _push_to_talk_requested(ctx)
    logger.info("input mode: %s", "push-to-talk" if push_to_talk else "auto")

    session = AgentSession(
        stt=inference.STT(STT_MODEL, language=STT_LANGUAGE),
        llm=inference.LLM(LLM_MODEL),
        tts=inference.TTS(TTS_MODEL, voice=TTS_VOICE, language=TTS_LANGUAGE),
        turn_handling=TurnHandlingOptions(
            # manual turns for push-to-talk; otherwise let the model detect turns
            turn_detection="manual" if push_to_talk else inference.TurnDetector(),
        ),
        max_tool_steps=8,
    )

    await session.start(agent=HomeAssistantAgent(), room=ctx.room)
    await ctx.connect()

    _forward_tool_events(ctx, session)

    if not push_to_talk:
        # automatic turn detection: the microphone stays live, no RPC wiring needed
        return

    # push-to-talk: disable audio input until the user presses the talk button
    session.input.set_audio_enabled(False)

    @ctx.room.local_participant.register_rpc_method("start_turn")
    async def start_turn(data: rtc.RpcInvocationData) -> str:
        session.interrupt()
        session.clear_user_turn()
        # listen only to the participant who started the turn (multi-user rooms)
        session.room_io.set_participant(data.caller_identity)
        session.input.set_audio_enabled(True)
        return "ok"

    @ctx.room.local_participant.register_rpc_method("end_turn")
    async def end_turn(data: rtc.RpcInvocationData) -> str:
        session.input.set_audio_enabled(False)
        session.commit_user_turn()
        return "ok"

    @ctx.room.local_participant.register_rpc_method("cancel_turn")
    async def cancel_turn(data: rtc.RpcInvocationData) -> str:
        session.input.set_audio_enabled(False)
        session.clear_user_turn()
        logger.info("cancel turn")
        return "ok"


if __name__ == "__main__":
    cli.run_app(server)
