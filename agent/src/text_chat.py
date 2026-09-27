"""Text chat with the Home Assistant agent: an A2A endpoint, and a plain-text bridge.

Built on the draft session-persistence API of livekit-agents (livekit/agents#7465).
Every use of that API lives in this module, so an upgrade is this file plus the pins in
pyproject.toml; docs/text-endpoint.md has the checklist.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import time
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from livekit.agents import AgentServer, AgentSession, inference
from livekit.agents.a2a import A2AClient, A2ASessionContext, TaskInput
from livekit.agents.store import LocalStore, StoreError

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
# the sessions the A2A endpoint has loaded, by conversation, for the history view
_live: dict[str, AgentSession] = {}


class _Conversation:
    """The one conversation the bridge talks in, kept until renewed or left idle."""

    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self._url = url
        self._headers = headers
        self._client: A2AClient | None = None
        self._conversation_id: str | None = None
        self._lock = asyncio.Lock()
        self._progress = _PhoneProgress()

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
            live = self._progress if settings.text_live_activity else None
            if live:
                live.begin(text)
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

    async def history(self, limit: int) -> dict[str, object]:
        """The current conversation's latest ``limit`` messages and tool calls, shaped
        like the card's conversation items."""
        conversation_id = self._conversation_id
        if conversation_id is None and _current_file.exists():
            conversation_id = _current_file.read_text().strip() or None
        items = []
        if conversation_id and (session := _live.get(conversation_id)) is not None:
            items = list(session.history.items)
        elif conversation_id:
            # not loaded, so read what the last close saved; this stamps the row closed
            # again, and leaves a new conversation an empty row that loads as new
            stored = store.session(
                conversation_id, conversation_id, endpoint=A2A_ENDPOINT
            )
            try:
                record = await stored.load()
                items = list(record.history) if record else []
            except StoreError:
                logger.warning("could not read conversation %s", conversation_id)
            finally:
                with contextlib.suppress(StoreError):
                    await stored.release()

        outputs = {i.call_id: i for i in items if i.type == "function_call_output"}
        shown: list[dict[str, object]] = []
        for item in items:
            ts = int(item.created_at * 1000)
            if item.type == "message" and item.role in ("user", "assistant"):
                if text := item.text_content:
                    role = "user" if item.role == "user" else "agent"
                    shown.append(
                        {"kind": "message", "id": item.id, "role": role, "text": text}
                        | {"ts": ts}
                    )
            elif item.type == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except ValueError:
                    args = item.arguments
                out = outputs.get(item.call_id)
                status = (
                    "running" if out is None else "error" if out.is_error else "done"
                )
                shown.append(
                    {"kind": "action", "id": item.id, "ts": ts, "name": item.name}
                    | {"args": args, "status": status}
                )
        return {
            "conversation_id": conversation_id,
            "busy": self._lock.locked(),
            "items": shown[-limit:],
        }


class _PhoneProgress:
    """A text turn's progress on the phone, replaced in place as the turn moves.

    Every push carries one ``tag``, and each replaces the last. In ``notification`` mode
    (the default) that is an ordinary notification with the question as its title; iOS
    alerts once per tag, so each update clears the last and posts anew, steps without
    sound and the answer with it. In ``activity`` mode it is a Live
    Activity, whose title iOS fixes at start and whose push-to-start fails while the
    Companion app is closed (home-assistant/iOS#5766); it clears TEXT_LIVE_CLEAR_AFTER
    seconds after the last turn, since iOS rations how many an app may start.
    """

    TAG = "ha-text"
    ACTIVITY_TITLE = "Home Assistant"

    def __init__(self) -> None:
        self._activity = settings.text_live_mode == "activity"
        self._question = ""
        self._steps: list[str] = []
        # (message, title, data), sent in order by one sender
        self._outbox: asyncio.Queue[tuple[str, str, dict[str, object]]] = (
            asyncio.Queue()
        )
        self._sender: asyncio.Task[None] | None = None
        self._clear: asyncio.Task[None] | None = None

    def begin(self, question: str) -> None:
        if self._clear is not None:
            self._clear.cancel()
        self._question = question if len(question) <= 60 else f"{question[:59]}…"
        self._steps = []
        self._put(["…"], "mdi:robot")

    def step(self, tool: str, args: object) -> None:
        values = args.values() if isinstance(args, dict) else [args] if args else []
        shown = [
            "/".join(map(str, v)) if isinstance(v, list) else str(v) for v in values
        ]
        self._steps.append(" · ".join([tool, *shown])[:80])
        done = [f"✓ {s}" for s in self._steps[-3:-1]]
        if tool.startswith("HassTurnOff"):
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
        self._put([*done, f"… {self._steps[-1]}"], icon)

    def report(self, text: str) -> None:
        if text:
            done = [f"✓ {s}" for s in self._steps[-2:]]
            self._put([*done, f"… {text}"], "mdi:progress-clock")

    def finish(self, answer: str, *, ok: bool) -> None:
        icon = "mdi:check-circle" if ok else "mdi:alert-circle"
        self._put([answer or "Done."], icon, final=True)
        if self._activity:
            self._clear = asyncio.create_task(self._clear_later())

    def _put(self, lines: list[str], icon: str, *, final: bool = False) -> None:
        data: dict[str, object] = {"tag": self.TAG}
        if settings.text_live_url:
            data["url"] = settings.text_live_url
        if self._activity:
            lines = [f"› {self._question}", *lines]
            lines[-1] = lines[-1][:250]
            title = self.ACTIVITY_TITLE
            data |= {"live_update": True, "notification_icon": icon, "silent": True}
        else:
            title = self._question
            data["push"] = {"interruption-level": "active"} | (
                {} if final else {"sound": "none"}
            )
            # a replacement under one tag updates silently, so clear it to pop again
            self._outbox.put_nowait(("clear_notification", "", {"tag": self.TAG}))
        self._outbox.put_nowait(("\n".join(lines), title, data))
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(self._send_all())

    async def _send_all(self) -> None:
        while True:
            message, title, data = await self._outbox.get()
            await ha.push(settings.text_live_activity, message, data, title=title)

    async def _clear_later(self) -> None:
        await asyncio.sleep(settings.text_live_clear_after)
        self._outbox.put_nowait(("clear_notification", "", {"tag": self.TAG}))
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(self._send_all())


def mount(server: AgentServer) -> None:
    """Serve the agent over A2A at /home-assistant, and the text bridge at /chat."""
    token = settings.text_api_token
    if not token:
        logger.warning("TEXT_API_TOKEN is not set, so the text chat endpoints are off")
        return
    _data_dir.mkdir(parents=True, exist_ok=True)

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

    @server.http.get(f"{CHAT_PATH}/history")
    async def chat_history(limit: int = 200) -> JSONResponse:
        """The current conversation for a UI to render; `busy` while a turn runs."""
        return JSONResponse(await conversation.history(limit))

    # after the routes above: the A2A binding mounts a catch-all that shadows later ones
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
        _live[ctx.context_id] = session
        session.on("close", lambda _: _live.pop(ctx.context_id, None))
        # a loaded conversation keeps only its latest items, so the prompt stays bounded
        await agent.update_chat_ctx(
            agent.chat_ctx.copy().truncate(max_items=settings.text_max_items)
        )
        ctx.attach(session)
