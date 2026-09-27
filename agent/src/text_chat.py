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
import time
from pathlib import Path

from fastapi import Request
from fastapi.responses import PlainTextResponse, Response
from livekit.agents import AgentServer, AgentSession, inference
from livekit.agents.a2a import A2AClient, A2ASessionContext, TaskInput
from livekit.agents.store import LocalStore

import ha
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
    """The one conversation the bridge talks in, kept until renewed or left idle."""

    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self._url = url
        self._headers = headers
        self._client: A2AClient | None = None
        self._conversation_id: str | None = None
        self._lock = asyncio.Lock()

    async def _open(self, renew: bool) -> tuple[A2AClient, str, bool]:
        """The client for this turn, its conversation, and whether that one is new."""
        if self._conversation_id is None and _current_file.exists():
            self._conversation_id = _current_file.read_text().strip() or None
        if self._conversation_id is not None:
            # touched on every turn, so the file's age is how long the user was away
            idle = time.time() - _current_file.stat().st_mtime
            renew = renew or idle > settings.text_renew_after
        if renew and self._client is not None:
            # the goodbye closes the old context on the server, which saves it
            await self._client.aclose()
            self._client = None
        started = renew or self._conversation_id is None
        if started:
            self._conversation_id = await store.create_database()
            _current_file.write_text(self._conversation_id)
            logger.info("started text conversation %s", self._conversation_id)
        assert self._conversation_id is not None
        _current_file.touch()
        if self._client is None:
            # the agent is the only one in the conversation, so its session is the front
            # session and the conversation id doubles as the A2A context id
            self._client = A2AClient(
                self._url, context_id=self._conversation_id, headers=self._headers
            )
        return self._client, self._conversation_id, started

    async def send(
        self, text: str, *, renew: bool = False, steps: bool = True
    ) -> tuple[str, str]:
        """Send one user turn and return (conversation id, the agent's reply).

        With ``steps``, the reply is preceded by one line per tool call the turn made.
        """
        async with self._lock:
            client, conversation_id, started = await self._open(renew)
            if not text:
                return conversation_id, "Started a new conversation."
            live = _LiveActivity.start(text) if settings.text_live_activity else None
            lines = ["(new conversation)"] if steps and started else []
            said: list[str] = []
            answer, ok = "", True
            task_input = TaskInput(text=text, conversation_id=conversation_id)
            async with client.send(task_input) as stream:
                try:
                    async with asyncio.timeout(settings.text_reply_timeout):
                        async for update in stream:
                            item = update.item
                            if item is None or item.type != "function_call":
                                if update.text:
                                    said.append(update.text)
                            elif item.update_of:
                                # a tool's progress report is a call naming the call it
                                # reports on, with the report as the update's text
                                lines.append(f"  … {update.text}")
                                if live:
                                    live.report(update.text)
                            else:
                                try:
                                    args = json.loads(item.arguments or "{}")
                                except ValueError:
                                    args = item.arguments
                                shown = json.dumps(args, ensure_ascii=False)
                                shown = shown.removeprefix("{}")[:120]
                                lines.append(f"→ {item.name}({shown})")
                                if live:
                                    live.step(item.name, args)
                            if update.state != "working":
                                answer = update.text or (said[-1] if said else "")
                                if update.state in ("failed", "canceled"):
                                    answer, ok = f"[{update.state}] {answer}", False
                                break
                except TimeoutError:
                    await stream.cancel("the text client stopped waiting")
                    answer = "\n".join(["[timeout] still working", *said])
                    ok = False
            if live:
                live.finish(answer, ok=ok)
            if not steps:
                lines = []
            elif lines:
                lines.append("")
            return conversation_id, "\n".join([*lines, answer])


class _LiveActivity:
    """One turn's progress as a Live Activity on the phone, updated in place.

    Built on the HA companion app: a notification with ``live_update`` and a ``tag``
    starts it, and each later one with that tag replaces it.
    """

    _last: _LiveActivity | None = None
    _clearing: asyncio.Task[None] | None = None

    def __init__(self, question: str) -> None:
        self._title = question[:60]
        self._tag = f"ha-text-{time.time_ns()}"
        self._steps: list[str] = []
        self._outbox: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        self._sender = asyncio.create_task(self._send_all())

    @classmethod
    def start(cls, question: str) -> _LiveActivity:
        """Open an activity for this question, clearing the previous one."""
        if (last := cls._last) is not None and not last._sender.done():
            last._sender.cancel()
            # held on the class so the task is not collected before it runs
            cls._clearing = asyncio.create_task(
                last._push("clear_notification", "", live=False)
            )
        cls._last = activity = cls(question)
        activity._outbox.put_nowait(("…", "mdi:robot"))
        return activity

    def step(self, tool: str, args: object) -> None:
        values = args.values() if isinstance(args, dict) else [args] if args else []
        shown = [
            "/".join(map(str, v)) if isinstance(v, list) else str(v) for v in values
        ]
        self._steps.append(" · ".join([tool, *shown])[:80])
        done = [f"✓ {s}" for s in self._steps[-3:-1]]
        message = "\n".join([*done, f"… {self._steps[-1]}"])
        if tool.startswith(("HassTurnOff",)):
            icon = "mdi:power-off"
        elif tool.startswith(("HassTurnOn", "HassToggle")):
            icon = "mdi:power"
        elif tool.startswith(("HassLight", "set_")) or "Set" in tool:
            icon = "mdi:tune-variant"
        elif "schedule" in tool:
            icon = "mdi:calendar-clock"
        elif tool == "send_notification":
            icon = "mdi:bell-ring"
        else:
            icon = "mdi:magnify"
        self._outbox.put_nowait((message, icon))

    def report(self, text: str) -> None:
        if text:
            done = [f"✓ {s}" for s in self._steps[-2:]]
            self._outbox.put_nowait(
                ("\n".join([*done, f"… {text}"]), "mdi:progress-clock")
            )

    def finish(self, answer: str, *, ok: bool) -> None:
        icon = "mdi:check-circle" if ok else "mdi:alert-circle"
        self._outbox.put_nowait((answer[:250] or "Done.", icon))
        self._outbox.put_nowait(None)

    async def _push(self, message: str, icon: str, *, live: bool = True) -> None:
        data: dict[str, object] = {"tag": self._tag}
        if live:
            data |= {"live_update": True, "notification_icon": icon, "silent": True}
        await ha.push(settings.text_live_activity, message, data, title=self._title)

    async def _send_all(self) -> None:
        # one sender per activity, so its updates reach the phone in order
        while (update := await self._outbox.get()) is not None:
            await self._push(*update)
        await asyncio.sleep(settings.text_live_clear_after)
        await self._push("clear_notification", "", live=False)


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
        agent = HomeAssistantAgent()
        # None when the caller names no conversation, which then lives in memory only
        await session.start(agent=agent, persist=ctx.persisted)
        # a loaded conversation keeps only its latest items, so the prompt stays bounded
        await agent.update_chat_ctx(
            agent.chat_ctx.copy().truncate(max_items=settings.text_max_items)
        )
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
        """Take `{"text", "new", "steps"}` JSON or plain text; reply in plain text."""
        body = (await request.body()).decode().strip()
        text, renew, steps = body, False, True
        if request.headers.get("content-type", "").startswith("application/json"):
            data = json.loads(body or "{}")
            text = str(data.get("text") or "").strip()
            renew = bool(data.get("new"))
            steps = bool(data.get("steps", True))
        query = request.query_params
        renew = renew or query.get("new", "") in ("1", "true", "yes")
        steps = steps and query.get("steps", "") not in ("0", "false", "no")
        if not text and not renew:
            return PlainTextResponse("no text given", status_code=400)
        try:
            conversation_id, reply = await conversation.send(
                text, renew=renew, steps=steps
            )
        except Exception as exc:
            logger.exception("text chat failed")
            return PlainTextResponse(f"[error] {exc}", status_code=502)
        return PlainTextResponse(reply, headers={"X-Conversation-Id": conversation_id})
