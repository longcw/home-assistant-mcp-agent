# modified from https://github.com/livekit-examples/basic-mcp/blob/main/mcp_client/server.py

import asyncio
import logging
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from typing import Any, Optional

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from livekit.agents import FunctionTool
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, JSONRPCMessage, Tool as MCPTool
from typing_extensions import NotRequired, TypedDict

from .mcp_utils import mcp_to_function_tool

logger = logging.getLogger()


# Base class for MCP servers
class MCPServer:
    async def connect(self):
        """Connect to the server."""
        raise NotImplementedError

    @property
    def connected(self) -> bool:
        """Whether the server is connected."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """A readable name for the server."""
        raise NotImplementedError

    async def list_tools(self) -> list[MCPTool]:
        """List the tools available on the server."""
        raise NotImplementedError

    async def call_tool(
        self, tool_name: str, arguments: Optional[dict[str, Any]] = None
    ) -> CallToolResult:
        """Invoke a tool on the server."""
        raise NotImplementedError

    async def cleanup(self):
        """Cleanup the server."""
        raise NotImplementedError

    async def get_agent_tools(self) -> list[FunctionTool]:
        """Get the tools available on the server as FunctionTools."""
        tools = await self.list_tools()
        return [mcp_to_function_tool(tool, self.call_tool) for tool in tools]


# Base class for MCP servers that use a ClientSession
class _MCPServerWithClientSession(MCPServer):
    """Base class for MCP servers that use a ClientSession to communicate with the server."""

    def __init__(self, cache_tools_list: bool):
        """
        Args:
            cache_tools_list: Whether to cache the tools list. If True, the tools list will be
            cached and only fetched from the server once. If False, the tools list will be
            fetched from the server on each call to list_tools(). You should set this to True
            if you know the server will not change its tools list, because it can drastically
            improve latency.
        """
        self.session: Optional[ClientSession] = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.cache_tools_list = cache_tools_list

        # The cache is always dirty at startup, so that we fetch tools at least once
        self._cache_dirty = True
        self._tools_list: Optional[list[MCPTool]] = None
        self.logger = logging.getLogger(__name__)

    @property
    def connected(self) -> bool:
        """Whether the server is connected."""
        return self.session is not None

    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        """Create the streams for the server."""
        raise NotImplementedError

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cleanup()

    def invalidate_tools_cache(self):
        """Invalidate the tools cache."""
        self._cache_dirty = True

    async def connect(self):
        """Connect to the server."""
        try:
            transport = await self.exit_stack.enter_async_context(self.create_streams())
            read, write = transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
            self.logger.info(f"Connected to MCP server: {self.name}")
        except Exception as e:
            self.logger.error(f"Error initializing MCP server: {e}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[MCPTool]:
        """List the tools available on the server."""
        if not self.session:
            raise RuntimeError(
                "Server not initialized. Make sure you call connect() first."
            )

        # Return from cache if caching is enabled, we have tools, and the cache is not dirty
        if self.cache_tools_list and not self._cache_dirty and self._tools_list:
            return self._tools_list

        # Reset the cache dirty to False
        self._cache_dirty = False

        try:
            # Fetch the tools from the server
            result = await self.session.list_tools()
            self._tools_list = result.tools
            return self._tools_list
        except Exception as e:
            self.logger.error(f"Error listing tools: {e}")
            raise

    async def call_tool(
        self, tool_name: str, arguments: Optional[dict[str, Any]] = None
    ) -> CallToolResult:
        """Invoke a tool on the server."""
        if not self.session:
            raise RuntimeError(
                "Server not initialized. Make sure you call connect() first."
            )

        arguments = arguments or {}
        try:
            return await self.session.call_tool(tool_name, arguments)
        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
            raise

    async def cleanup(self):
        """Cleanup the server."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
                self.logger.info(f"Cleaned up MCP server: {self.name}")
            except Exception as e:
                self.logger.error(f"Error cleaning up server: {e}")


# Define parameter types for clarity
class MCPServerSseParams(TypedDict):
    url: str
    headers: NotRequired[dict[str, Any]]
    timeout: NotRequired[float]
    sse_read_timeout: NotRequired[float]


# SSE server implementation
class MCPServerSse(_MCPServerWithClientSession):
    """MCP server implementation that uses the HTTP with SSE transport."""

    def __init__(
        self,
        params: MCPServerSseParams,
        cache_tools_list: bool = False,
        name: Optional[str] = None,
    ):
        """Create a new MCP server based on the HTTP with SSE transport.

        Args:
            params: The params that configure the server including the URL, headers,
                   timeout, and SSE read timeout.
            cache_tools_list: Whether to cache the tools list.
            name: A readable name for the server.
        """
        super().__init__(cache_tools_list)
        self.params = params
        self._name = name or f"SSE Server at {self.params.get('url', 'unknown')}"

    def create_streams(
        self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        """Create the streams for the server."""
        return sse_client(
            url=self.params["url"],
            headers=self.params.get("headers"),
            timeout=self.params.get("timeout", 5),
            sse_read_timeout=self.params.get("sse_read_timeout", 60 * 5),
        )

    @property
    def name(self) -> str:
        """A readable name for the server."""
        return self._name
