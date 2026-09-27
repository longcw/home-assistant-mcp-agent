"""Text chat with the Home Assistant agent: an A2A endpoint, and a plain-text bridge.

Built on the draft session-persistence API of livekit-agents (livekit/agents#7465).
Every use of that API lives in this module, so an upgrade is this file plus the pins in
pyproject.toml; docs/text-endpoint.md has the checklist.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from pathlib import Path

from fastapi import Request
from fastapi.responses import PlainTextResponse, Response
from livekit.agents import AgentServer, AgentSession, inference
from livekit.agents.a2a import A2AClient, A2ASessionContext, TaskInput
from livekit.agents.store import LocalStore

from agent import HomeAssistantAgent
from config import settings

logger = logging.getLogger("ha-mcp-agent.text")

A2A_ENDPOINT = "home-assistant"
CHAT_PATH = "/chat"

_data_dir = Path(settings.text_data_dir)
_current_file = _data_dir / "current_conversation"

# one SQLite file per conversation; handed to AgentServer(store=...) in main.py
store = LocalStore(_data_dir / "sessions")


class _Conversation:
    """The one conversation the bridge talks in, kept across restarts until renewed."""

    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self._url = url
        self._headers = headers
        self._client: A2AClient | None = None
        self._conversation_id: str | None = None
        self._lock = asyncio.Lock()

    async def _open(self, renew: bool) -> A2AClient:
        if self._conversation_id is None and _current_file.exists():
            self._conversation_id = _current_file.read_text().strip() or None
        if renew and self._client is not None:
            # the goodbye closes the old context on the server, which saves it
            await self._client.aclose()
            self._client = None
        if renew or self._conversation_id is None:
            self._conversation_id = await store.create_database()
            _current_file.write_text(self._conversation_id)
            logger.info("started text conversation %s", self._conversation_id)
        if self._client is None:
            # the agent is the only one in the conversation, so its session is the front
            # session and the conversation id doubles as the A2A context id
            self._client = A2AClient(
                self._url, context_id=self._conversation_id, headers=self._headers
            )
        return self._client

    async def send(self, text: str, *, renew: bool = False) -> tuple[str, str]:
        """Send one user turn and return (conversation id, the agent's reply)."""
        async with self._lock:
            client = await self._open(renew)
            assert self._conversation_id is not None
            if not text:
                return self._conversation_id, "Started a new conversation."
            task_input = TaskInput(text=text, conversation_id=self._conversation_id)
            said: list[str] = []
            async with client.send(task_input) as stream:
                try:
                    async with asyncio.timeout(settings.text_reply_timeout):
                        async for update in stream:
                            if update.state == "working":
                                if update.text:
                                    said.append(update.text)
                                continue
                            if update.state in ("failed", "canceled"):
                                reply = f"[{update.state}] {update.text}"
                                return self._conversation_id, reply
                            reply = update.text or "\n".join(said)
                            return self._conversation_id, reply
                except TimeoutError:
                    await stream.cancel("the text client stopped waiting")
                    partial = "\n".join(said)
                    return self._conversation_id, partial or "[timeout] still working"
            return self._conversation_id, "\n".join(said)


def mount(server: AgentServer) -> None:
    """Serve the agent over A2A at /home-assistant, and the text bridge at /chat."""
    token = settings.text_api_token
    if not token:
        logger.warning("TEXT_API_TOKEN is not set, so the text chat endpoints are off")
        return
    _data_dir.mkdir(parents=True, exist_ok=True)

    @server.a2a_session(
        endpoint=A2A_ENDPOINT,
        description="Controls the home through Home Assistant.",
        idle_timeout=settings.text_idle_timeout,
    )
    async def serve(ctx: A2ASessionContext) -> None:
        session = AgentSession(llm=inference.LLM(settings.llm_model), max_tool_steps=8)
        # None when the caller names no conversation, which then lives in memory only
        await session.start(agent=HomeAssistantAgent(), persist=ctx.persisted)
        ctx.attach(session)

    @server.http.middleware("http")
    async def require_token(request: Request, call_next):
        if request.url.path.startswith((f"/{A2A_ENDPOINT}", CHAT_PATH)):
            given = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(given.encode(), token.encode()):
                return PlainTextResponse("unauthorized", status_code=401)
        return await call_next(request)

    conversation = _Conversation(
        f"http://127.0.0.1:{settings.http_port}/{A2A_ENDPOINT}",
        headers={"Authorization": f"Bearer {token}"},
    )

    @server.http.post(CHAT_PATH)
    async def chat(request: Request) -> Response:
        """Take `{"text": ..., "new": bool}` or plain text, and reply in plain text."""
        body = (await request.body()).decode().strip()
        text, renew = body, False
        if request.headers.get("content-type", "").startswith("application/json"):
            data = json.loads(body or "{}")
            text = str(data.get("text") or "").strip()
            renew = bool(data.get("new"))
        renew = renew or request.query_params.get("new", "") in ("1", "true", "yes")
        if not text and not renew:
            return PlainTextResponse("no text given", status_code=400)
        try:
            conversation_id, reply = await conversation.send(text, renew=renew)
        except Exception as exc:
            logger.exception("text chat failed")
            return PlainTextResponse(f"[error] {exc}", status_code=502)
        return PlainTextResponse(reply, headers={"X-Conversation-Id": conversation_id})
