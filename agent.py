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
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import openai

from mcp_client import MCPServerSse

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

    def __init__(self, *, tools: list[llm.FunctionTool], mcp_server: MCPServerSse):
        super().__init__(
            instructions=instructions,
            # stt=deepgram.STT(model="nova-2", language="zh-CN"),
            # llm=openai.LLM(model="gpt-4o"),
            # tts=cartesia.TTS(language="zh"),
            # vad=silero.VAD.load(),
            allow_interruptions=True,
            llm=openai.realtime.RealtimeModel(
                model="gpt-4o-realtime-preview", turn_detection=None
            ),
            tools=tools,
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

        result = await self._mcp.call_tool("get_home_state")
        content = yaml.safe_load(result.content[0].text)["result"]
        devices = yaml.safe_load(content)
        devices = next(iter(devices.values()))

        self._devices = pd.DataFrame(devices)
        self._devices_updated_at = time.time()
        return self._devices

    def df_to_str(self, df: pd.DataFrame) -> str:
        return yaml.dump(list(df.to_dict(orient="index").values()))


async def entrypoint(ctx: JobContext):
    mcp_server = MCPServerSse(
        params={
            "url": os.getenv("HOME_ASSISTANT_MCP_URL"),
            "headers": {"Authorization": f"Bearer {os.getenv('HOME_ASSISTANT_TOKEN')}"},
        },
        cache_tools_list=True,
        name="Home Assistant MCP Server",
    )
    await mcp_server.connect()

    agent_tools = await mcp_server.get_agent_tools()
    agent = HomeAssistantAgent(tools=agent_tools, mcp_server=mcp_server)

    await ctx.connect()

    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)
    room_io = session._room_io

    # session.generate_reply(user_input="你好")

    # disable input audio at the start
    session.input.set_audio_enabled(False)

    @ctx.room.local_participant.register_rpc_method("start_turn")
    async def start_turn(data: rtc.RpcInvocationData):
        session.interrupt()
        session.clear_user_turn()

        # listen to the caller if multi-user
        room_io.set_participant(data.caller_identity)
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


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, request_fnc=handle_request))
