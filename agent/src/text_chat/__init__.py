"""Text chat with the Home Assistant agent: an A2A endpoint, and a plain-text bridge.

Built on the draft session-persistence API of livekit-agents (livekit/agents#7465).
The bridge at /chat sends each message over A2A to the agent's own endpoint, which
loads the person's conversation (conversations.py). A request names its person by
``user``, an id; none is no one in particular. docs/text-endpoint.md has the API and
the upgrade checklist.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from livekit.agents import AgentServer, AgentSession
from livekit.agents.a2a import A2ASessionContext

from agent import HomeAssistantAgent
from config import UPDATE_PREFIX, settings
from conversations import A2A_ENDPOINT, conversations
from text_chat.chat import TextChat
from utils import build_llm

logger = logging.getLogger("ha-mcp-agent.text")

CHAT_PATH = "/chat"

_TRUE = ("1", "true", "yes")


def _user_id(value: object) -> str | None:
    """The id a request names its person by; ids are casefolded names."""
    user_id = str(value or "").strip().casefold()
    return user_id or None


async def _json(request: Request) -> dict[str, Any]:
    body = (await request.body()).decode().strip()
    return json.loads(body or "{}")


def mount(server: AgentServer) -> None:
    """Serve the agent over A2A at /home-assistant, and the text bridge at /chat."""
    token = settings.text_api_token
    if not token:
        logger.warning("TEXT_API_TOKEN is not set, so the text chat endpoints are off")
        return

    @server.http.middleware("http")
    async def require_token(request: Request, call_next):
        if request.url.path.startswith((f"/{A2A_ENDPOINT}", CHAT_PATH)):
            given = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(given.encode(), token.encode()):
                return PlainTextResponse("unauthorized", status_code=401)
        return await call_next(request)

    chats: dict[str | None, TextChat] = {}

    def chat_of(user: object) -> TextChat:
        user_id = _user_id(user)
        if user_id not in chats:
            chats[user_id] = TextChat(
                user_id,
                f"http://127.0.0.1:{settings.http_port}/{A2A_ENDPOINT}",
                headers={"Authorization": f"Bearer {token}"},
            )
        return chats[user_id]

    @server.http.post(CHAT_PATH)
    async def chat(request: Request) -> Response:
        """Take `{"text", "new", "steps", "wait", "interrupt", "user"}` JSON or text.

        Answers 202 once the turn has started, and the answer goes to the person's
        phone; with `wait` it answers with the agent's reply instead."""
        query = request.query_params
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await _json(request)
            text = str(data.get("text") or "").strip()
        else:
            data = {}
            text = (await request.body()).decode().strip()
        renew = bool(data.get("new")) or query.get("new", "") in _TRUE
        steps = bool(data.get("steps", True)) and query.get("steps", "") not in (
            "0",
            "false",
            "no",
        )
        wait = bool(data.get("wait")) or query.get("wait", "") in _TRUE
        if not text and not renew:
            return PlainTextResponse("no text given", status_code=400)
        person = chat_of(data.get("user") or query.get("user"))
        try:
            conversation_id, reply = await person.send(
                text,
                renew=renew,
                steps=steps,
                wait=wait,
                interrupt=bool(data.get("interrupt", True)),
            )
        except Exception as exc:
            logger.exception("text chat failed")
            return PlainTextResponse(f"[error] {exc}", status_code=502)
        return PlainTextResponse(
            reply,
            status_code=200 if wait or not text else 202,
            headers={"X-Conversation-Id": conversation_id},
        )

    @server.http.get(f"{CHAT_PATH}/history")
    async def chat_history(
        limit: int = 200, user: str = "", conversation_id: str = ""
    ) -> JSONResponse:
        """A person's current conversation, or one they name, for a UI; `busy` while
        a turn runs."""
        history = await chat_of(user).history(limit, conversation_id or None)
        if history is None:
            return JSONResponse({"detail": "no such conversation"}, status_code=404)
        return JSONResponse(history)

    @server.http.get(f"{CHAT_PATH}/conversations")
    async def chat_conversations(user: str = "") -> JSONResponse:
        """A person's conversations, latest first, and which one is current."""
        return JSONResponse(await chat_of(user).listing())

    @server.http.post(f"{CHAT_PATH}/switch")
    async def chat_switch(request: Request) -> JSONResponse:
        """Make `{"conversation_id", "user"}` the person's current conversation."""
        data = await _json(request)
        target = str(data.get("conversation_id") or "")
        return JSONResponse(
            {"switched": await chat_of(data.get("user")).switch(target)}
        )

    @server.http.post(f"{CHAT_PATH}/delete")
    async def chat_delete(request: Request) -> JSONResponse:
        """Delete `{"conversation_id", "user"}`, a person's past conversation."""
        data = await _json(request)
        target = str(data.get("conversation_id") or "")
        return JSONResponse({"deleted": await chat_of(data.get("user")).delete(target)})

    @server.http.post(f"{CHAT_PATH}/cancel")
    async def chat_cancel(request: Request) -> JSONResponse:
        """Stop a person's running turn's reply, answering at once: `{"task_id",
        "user"}`, the task from history or none for whichever runs; `"force": true`
        stops the tools it started as well."""
        data = await _json(request)
        person = chat_of(data.get("user"))
        task_id = str(data.get("task_id") or "") or None
        cancelled = person.cancel(task_id, force=bool(data.get("force")))
        # a client cancels as the person starts to speak, so their message is coming
        return JSONResponse({"cancelled": cancelled, "warming": person.warm()})

    @server.http.post(f"{CHAT_PATH}/warm")
    async def chat_warm(request: Request) -> JSONResponse:
        """Load a person's current conversation ahead of their message: `{"user"}`."""
        data = await _json(request)
        return JSONResponse({"warming": chat_of(data.get("user")).warm()})

    updates: set[asyncio.Task[Any]] = set()

    @server.http.post(f"{CHAT_PATH}/events")
    async def chat_event(request: Request) -> Response:
        """Take a webhook's `{"source", "text"}` into a person's conversation as a turn
        the agent answers with no tools, e.g. a Claude Code task that finished."""
        data = await _json(request)
        text = str(data.get("text") or "").strip()
        if not text:
            return PlainTextResponse("no text given", status_code=400)
        person = chat_of(request.query_params.get("user"))
        source = str(data.get("source") or "a webhook")
        logger.info("update from %s: %s", source, text[:120])
        # answered in the background, so the sender is not held for a whole turn
        turn = asyncio.create_task(
            person.send(f"{UPDATE_PREFIX} {source}: {text}", interrupt=False)
        )
        updates.add(turn)
        turn.add_done_callback(updates.discard)
        return JSONResponse({"accepted": True}, status_code=202)

    # after the routes above: the A2A binding mounts a catch-all that shadows later ones
    @server.a2a_session(
        endpoint=A2A_ENDPOINT,
        description="Controls the home through Home Assistant.",
        idle_timeout=settings.text_idle_timeout,
    )
    async def serve(ctx: A2ASessionContext) -> None:
        session = AgentSession(llm=build_llm(), max_tool_steps=settings.max_tool_steps)
        agent = HomeAssistantAgent(user_id=conversations.owner(ctx.context_id))
        agent._suggest_replies_cb = conversations.suggested.setdefault(
            ctx.context_id, []
        ).extend
        # None when the caller names no conversation, which then lives in memory only
        await session.start(agent=agent, persist=ctx.persisted)
        await conversations.attach(session, agent, ctx.context_id)
        ctx.attach(session)
