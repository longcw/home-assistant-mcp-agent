import os

from dotenv import load_dotenv
from mcp_client import MCPServerSse

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, llm
from livekit.plugins import openai

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

# Communication Style
- Respond conversationally and confirm actions after completion
- Be concise but helpful in your explanations
- Acknowledge when you're checking device status or performing actions
- Use the same language as the user
"""  # noqa: E501


class FunctionAgent(Agent):
    """A LiveKit agent that uses MCP tools"""

    def __init__(self, *, tools: list[llm.FunctionTool]):
        super().__init__(
            instructions=instructions,
            # stt=deepgram.STT(model="nova-2", language="zh-CN"),
            # llm=openai.LLM(model="gpt-4o"),
            # tts=cartesia.TTS(language="zh"),
            # vad=silero.VAD.load(),
            allow_interruptions=True,
            llm=openai.realtime.RealtimeModel(model="gpt-4o-realtime-preview-2024-12-17"),
            tools=tools,
        )


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
    agent = FunctionAgent(tools=agent_tools)

    await ctx.connect()

    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

    session.generate_reply(user_input="你好")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    # asyncio.run(main())
