import asyncio
import logging
import os
import time

import pandas as pd
import yaml
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    JobContext,
    JobProcess,
    JobRequest,
    UserStateChangedEvent,
    WorkerOptions,
    cli,
    llm,
    mcp,
    utils,
)
from livekit.plugins import openai, silero

logger = logging.getLogger("ha-mcp-agent")

load_dotenv()


instructions = """
You are a voice assistant for Home Assistant, designed to help users control their smart home devices.

# Device Control Guidelines
- Before controlling any device, first query its status to confirm the exact device name
- Reuse previous query results when appropriate to avoid redundant status checks
- When a device name contains a comma, use the portion after the comma (the alias) for control
- You MUST use the exact original device name from the system when calling the tool, keep the multiple spaces in the original name

# Handling Ambiguous Requests
- If the requested device cannot be found, identify and suggest similar alternatives, e.g. 书房的射灯 -> 书房照明  射灯 右键 (follow the actual device name)
- Users may mix area references in spoken language (e.g., 厨房/kitchen and 餐厅/dining room) - find the device based on name in possible areas
- When user ask for a type of device of an area, you should ask which device in the area they want to control
- When presenting options to users, use natural device names for clarity
- When executing tool calls, ALWAYS use the exact original device name from the system
- When user ask for devices in a specific area, get all areas first and match the area name in case of ambiguity

# Communication Style
- Respond conversationally and confirm actions after completion
- Be concise but helpful in your explanations
- Acknowledge when you're checking device status or performing actions
- Use the same language as the user
"""  # noqa: E501


class HomeAssistantAgent(Agent):
    """A LiveKit agent that uses MCP tools"""

    def __init__(self, *, mcp_server: mcp.MCPServer):
        super().__init__(
            instructions=instructions,
            mcp_servers=[mcp_server],
            llm=openai.realtime.RealtimeModel(
                voice="alloy",
                model="gpt-4o-realtime-preview",
                turn_detection=None,  # disable server side turn detection
            ),
        )
        self._mcp = mcp_server
        self._devices: pd.DataFrame | None = None
        self._devices_updated_at: float = 0
        self._devices_timeout_interval = 30

    @llm.function_tool
    async def get_areas(self) -> list[str]:
        """Get all areas in the home"""
        logger.info("get_areas")
        devices = await self.get_home_state()
        return devices["areas"].unique().tolist()

    @llm.function_tool
    async def get_device_domains(self) -> list[str]:
        """Get all device domains in the home"""
        logger.info("get_device_domains")
        devices = await self.get_home_state()
        return devices["domain"].unique().tolist()

    @llm.function_tool
    async def get_devices(self, area: str | list[str]) -> str:
        """Get devices status in the area or areas

        Args:
            area: The area or list of areas to get devices from.
        """  # noqa: E501
        logger.info(f"get_devices: {area}")

        devices = await self.get_home_state(force_update=True)
        if isinstance(area, str):
            area = [area]
        area = [a.strip() for a in area]
        df = devices[devices["areas"].isin(area)]

        logger.info(f"found {len(df)} devices in {area}")
        if len(df) == 0:
            areas = await self.get_areas()
            return f"No devices found in {area}, available areas: {areas}, try to use the current area name"  # noqa: E501
        return self.df_to_str(df)

    @llm.function_tool
    async def get_environment_info(self) -> str:
        """Get the current environment information like temperature, humidity, etc."""
        logger.info("get_environment_info")
        devices = await self.get_home_state(force_update=True)
        return self.df_to_str(devices[devices["domain"] == "sensor"])

    async def get_home_state(self, force_update: bool = False) -> pd.DataFrame:
        if (
            not force_update
            and self._devices is not None
            and time.time() - self._devices_updated_at < self._devices_timeout_interval
        ):
            return self._devices

        result = await self._mcp._client.call_tool("get_home_state")
        content = yaml.safe_load(result.content[0].text)["result"]
        devices = yaml.safe_load(content)
        devices = next(iter(devices.values()))

        self._devices = pd.DataFrame(devices)
        self._devices_updated_at = time.time()
        return self._devices

    def df_to_str(self, df: pd.DataFrame) -> str:
        return yaml.dump(list(df.to_dict(orient="index").values()))


class IdleAgent(Agent):
    """An agent that does nothing"""

    def __init__(self):
        super().__init__(instructions="not needed", tts=None, vad=None)


async def entrypoint(ctx: JobContext):
    ha_url = os.getenv("HOME_ASSISTANT_MCP_URL")
    if not ha_url:
        raise ValueError("HOME_ASSISTANT_MCP_URL is not set")
    ha_token = os.getenv("HOME_ASSISTANT_TOKEN")
    if not ha_token:
        raise ValueError("HOME_ASSISTANT_TOKEN is not set")
    mcp_server = mcp.MCPServerHTTP(
        url=ha_url, headers={"Authorization": f"Bearer {ha_token}"}
    )
    await ctx.connect()

    agent = HomeAssistantAgent(mcp_server=mcp_server)
    idle_agent = IdleAgent()

    # push-to-talk mode: client call the start_turn and end_turn methods
    # vad mode: client call the start_turn (based on keywords), vad call the end_turn
    rparticipant = await utils.wait_for_participant(ctx.room)
    support_ptt = rparticipant.attributes.get("supports-ptt", "0") == "1"
    logger.info(f"supports-ptt: {support_ptt}")

    # create session
    session = AgentSession(
        turn_detection="manual" if support_ptt else "vad",
        vad=ctx.proc.userdata["vad"],
        tts=openai.TTS(voice="alloy"),
    )
    await session.start(
        agent=agent if support_ptt else idle_agent,
        room=ctx.room,
    )
    room_io = session._room_io

    # disable input audio at the start
    session.input.set_audio_enabled(False)

    audio_toggle: asyncio.TimerHandle | None = None

    if not support_ptt:
        loop = asyncio.get_event_loop()
        idle_toggle: asyncio.TimerHandle | None = None

        def reset_idle_timer(delay: float | None = 60):
            nonlocal idle_toggle
            if idle_toggle:
                idle_toggle.cancel()

            if delay is not None:
                logger.info(f"mark agent as idle in {delay} seconds")
                idle_toggle = loop.call_later(delay, session.update_agent, idle_agent)

        @session.on("user_state_changed")
        def on_user_state_changed(ev: UserStateChangedEvent):
            if ev.new_state == "listening":
                reset_idle_timer(60)

        @session.on("agent_state_changed")
        def on_agent_state_changed(ev: AgentStateChangedEvent):
            nonlocal audio_toggle
            if session.input.audio_enabled and ev.new_state == "speaking":
                reset_idle_timer(None)
                if audio_toggle:
                    audio_toggle.cancel()
                session.input.set_audio_enabled(False)

            elif not session.input.audio_enabled and ev.new_state == "listening":
                reset_idle_timer(60)
                if audio_toggle:
                    audio_toggle.cancel()

                audio_toggle = loop.call_later(
                    1.0, session.input.set_audio_enabled, True
                )
                logger.info("enable agent audio input in 1 second")

    @ctx.room.local_participant.register_rpc_method("start_turn")
    async def start_turn(data: rtc.RpcInvocationData):
        if session.current_agent != agent:
            session.update_agent(agent)
            await session._update_activity_atask

        if audio_toggle:
            audio_toggle.cancel()

        session.interrupt()
        session.clear_user_turn()

        # listen to the caller if multi-user
        room_io.set_participant(data.caller_identity)

        if not support_ptt:
            await session.say("我在！")
        else:
            session.input.set_audio_enabled(True)

    @ctx.room.local_participant.register_rpc_method("end_turn")
    async def end_turn(data: rtc.RpcInvocationData):
        session.input.set_audio_enabled(False)
        session.commit_user_turn()

    @ctx.room.local_participant.register_rpc_method("cancel_turn")
    async def cancel_turn(data: rtc.RpcInvocationData):
        session.input.set_audio_enabled(False)
        session.clear_user_turn()
        logger.info("cancel turn")


async def handle_request(request: JobRequest) -> None:
    await request.accept(
        identity="ptt-agent",
        # this attribute communicates to frontend that we support PTT
        attributes={"push-to-talk": "1"},
    )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint, request_fnc=handle_request, prewarm_fnc=prewarm
        )
    )
