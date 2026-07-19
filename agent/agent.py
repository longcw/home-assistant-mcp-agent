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
        super().__init__(instructions=load_instructions(), tools=[toolset])

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
        # payload: "on" | "off" — toggles spoken (TTS) replies; text replies still work.
        _set_audio_output(data.payload == "on")
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
