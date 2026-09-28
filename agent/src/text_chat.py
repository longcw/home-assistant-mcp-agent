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
import uuid
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
# what the agent's suggest_replies offered in the running turn
_suggested: list[str] = []


class _Conversation:
    """The one conversation the bridge talks in, kept until renewed or left idle."""

    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self._url = url
        self._headers = headers
        self._client: A2AClient | None = None
        self._conversation_id: str | None = None
        self._lock = asyncio.Lock()
        self._progress = _PhoneProgress()
        # the buttons on the latest answer notification: action id -> reply text
        self._actions: dict[str, str] = {}
        self._reply_action = ""
        self._listener: asyncio.Task[None] | None = None
        self._tapped: set[asyncio.Task[object]] = set()

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
                _suggested.clear()
                if self._listener is None or self._listener.done():
                    self._listener = asyncio.create_task(self._listen())
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
                # the suggested replies become buttons, and Reply takes free text
                nonce = uuid.uuid4().hex[:8]
                replies = list(dict.fromkeys(_suggested))[:4]
                self._actions = {
                    f"HATEXT_{nonce}_{i}": r for i, r in enumerate(replies)
                }
                self._reply_action = f"HATEXT_{nonce}_REPLY"
                # a tap can control the house, so it asks for Face ID first
                actions = [
                    {"action": k, "title": r, "authenticationRequired": True}
                    for k, r in self._actions.items()
                ]
                actions.append(
                    {
                        "action": self._reply_action,
                        "title": "Reply",
                        "behavior": "textInput",
                        "textInputButtonTitle": "Send",
                        "authenticationRequired": True,
                    }
                )
                live.finish(answer, ok=ok, actions=actions)
            if not steps:
                lines = []
            elif lines:
                lines.append("")
            return conversation_id, "\n".join([*lines, answer])

    async def _listen(self) -> None:
        """Send a tap on the latest answer's buttons into the conversation."""
        async for event in ha.subscribe("mobile_app_notification_action"):
            action = event.get("action", "")
            if action == self._reply_action:
                reply = (event.get("reply_text") or "").strip()
            else:
                reply = self._actions.get(action, "")
            if reply:
                logger.info("reply from a notification: %s", reply)
                task = asyncio.create_task(self.send(reply))
                self._tapped.add(task)
                task.add_done_callback(self._tapped.discard)

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
        # the quick replies offered since the user last spoke, for a UI's chips
        suggestions: list[str] = []
        for item in items:
            ts = int(item.created_at * 1000)
            if item.type == "message" and item.role in ("user", "assistant"):
                if text := item.text_content:
                    role = "user" if item.role == "user" else "agent"
                    shown.append(
                        {"kind": "message", "id": item.id, "role": role, "text": text}
                        | {"ts": ts}
                    )
                    if role == "user":
                        suggestions = []
            elif item.type == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except ValueError:
                    args = item.arguments
                out = outputs.get(item.call_id)
                status = (
                    "running" if out is None else "error" if out.is_error else "done"
                )
                action: dict[str, object] = {
                    "kind": "action",
                    "id": item.id,
                    "call_id": item.call_id,
                    "ts": ts,
                    "name": item.name,
                    "args": args,
                    "status": status,
                }
                # short outputs only (a scheduled task, say), not a home's device list
                if out is not None and len(out.output) <= 1000:
                    action["output"] = out.output
                shown.append(action)
                if item.name == "suggest_replies" and isinstance(args, dict):
                    suggestions = [str(r) for r in args.get("replies") or []]
        return {
            "conversation_id": conversation_id,
            "busy": self._lock.locked(),
            "items": shown[-limit:],
            "suggestions": suggestions,
        }


class _PhoneProgress:
    """A text turn's progress on the phone, updated in place as the turn moves.

    Every push carries one ``tag``, so each updates the last. In ``activity`` mode it is
    a silent Live Activity under a fixed title (iOS fixes it at start), one line per
    update, each with a title-only alert so it lands without a buzz; the question and
    the answer also come as a regular notification with sound, since iOS refuses
    push-started activities once its allowance is spent. The activity clears
    TEXT_LIVE_CLEAR_AFTER seconds after the last turn. In ``notification`` mode it is a
    notification titled with the question: the question and the answer pop, and each
    step is passive.
    """

    TAG = "ha-text"
    NOTICE_TAG = "ha-text-answer"
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
        self._put(["…"], f"› {self._question}", "…", "mdi:robot", pop=True)
        if self._activity:
            self._notice("…")

    def step(self, tool: str, args: object) -> None:
        values = args.values() if isinstance(args, dict) else [args] if args else []
        shown = [
            "/".join(map(str, v)) if isinstance(v, list) else str(v) for v in values
        ]
        self._steps.append(" · ".join([tool, *shown])[:80])
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
        lines = [f"→ {s}" for s in self._steps]
        self._put(lines, lines[-1], f"Step {len(self._steps)}", icon)

    def report(self, text: str) -> None:
        if text:
            lines = [*(f"→ {s}" for s in self._steps), f"  … {text}"]
            status = f"Step {len(self._steps)}"
            self._put(lines, f"… {text}", status, "mdi:progress-clock")

    def finish(
        self, answer: str, *, ok: bool, actions: list[dict[str, object]]
    ) -> None:
        answer = answer or "Done."
        icon = "mdi:check-circle" if ok else "mdi:alert-circle"
        status = "Done" if ok else "Failed"
        if self._activity:
            self._put([answer], answer, status, icon, pop=True)
            self._notice(answer, actions)
            self._clear = asyncio.create_task(self._clear_later())
        else:
            self._put([answer], answer, status, icon, pop=True, actions=actions)

    def _notice(
        self, message: str, actions: list[dict[str, object]] | None = None
    ) -> None:
        """A sounded notification beside the activity, which reaches the phone even
        when the activity never started; cleared first so it pops, not updates."""
        data: dict[str, object] = {"tag": self.NOTICE_TAG}
        if settings.text_live_url:
            data["url"] = settings.text_live_url
        if actions:
            data["actions"] = actions
        self._outbox.put_nowait(("clear_notification", "", {"tag": self.NOTICE_TAG}))
        self._outbox.put_nowait((message, self._question, data))

    def _put(
        self,
        lines: list[str],
        line: str,
        status: str,
        icon: str,
        *,
        pop: bool = False,
        actions: list[dict[str, object]] | None = None,
    ) -> None:
        """Queue one update: ``lines`` is a notification's body, ``line`` and
        ``status`` a Live Activity's message and short status."""
        data: dict[str, object] = {"tag": self.TAG}
        if settings.text_live_url:
            data["url"] = settings.text_live_url
        if self._activity:
            message, title = line[:250], self.ACTIVITY_TITLE
            data |= {
                "live_update": True,
                "critical_text": status,
                "notification_icon": icon,
            }
            # the relay uses a given alert as is: title-only lands without the buzz
            data["alert"] = {"title": ""}
        else:
            message, title = "\n".join(lines), self._question
            if actions:
                data["actions"] = actions
            if pop:
                # a replacement under one tag updates silently, so clear it to pop again
                self._outbox.put_nowait(("clear_notification", "", {"tag": self.TAG}))
            else:
                # appended to the question's notification, or, once that is gone,
                # delivered without a banner or a sound
                data["push"] = {"interruption-level": "passive", "sound": "none"}
        self._outbox.put_nowait((message, title, data))
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
        agent._suggest_replies_cb = lambda replies: _suggested.extend(replies)
        # None when the caller names no conversation, which then lives in memory only
        await session.start(agent=agent, persist=ctx.persisted)
        _live[ctx.context_id] = session
        session.on("close", lambda _: _live.pop(ctx.context_id, None))
        # a loaded conversation keeps only its latest items, so the prompt stays bounded
        await agent.update_chat_ctx(
            agent.chat_ctx.copy().truncate(max_items=settings.text_max_items)
        )
        ctx.attach(session)
