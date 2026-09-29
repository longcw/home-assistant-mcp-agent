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
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from livekit.agents import AgentServer, AgentSession, inference
from livekit.agents.a2a import A2AClient, A2ASessionContext, TaskInput, TaskStream
from livekit.agents.store import LocalStore, StoreError

import ha
import scheduler_client as scheduler
from agent import HomeAssistantAgent
from config import UPDATE_PREFIX, settings

logger = logging.getLogger("ha-mcp-agent.text")

A2A_ENDPOINT = "home-assistant"
CHAT_PATH = "/chat"

_data_dir = Path(settings.text_data_dir)

# one SQLite file per conversation; handed to AgentServer(store=...) in main.py
store = LocalStore(_data_dir / "sessions")
# the sessions the A2A endpoint has loaded, by conversation, for the history view
_live: dict[str, AgentSession] = {}
# what the agent's suggest_replies offered in each conversation's running turn
_suggested: dict[str, list[str]] = {}
# the person each conversation belongs to, for the A2A endpoint's agent
_owners: dict[str, str | None] = {}
# each conversation's owner, title and last turn, so a person can list and reopen theirs
_index_file = _data_dir / "conversations.json"


def _index() -> dict[str, dict[str, object]]:
    try:
        return json.loads(_index_file.read_text())
    except (OSError, ValueError):
        return {}


def _record(conversation_id: str, user: str | None, title: str = "") -> None:
    """Note a turn in a person's conversation; the first message becomes its title."""
    index = _index()
    entry = index.setdefault(conversation_id, {"created": time.time()})
    entry["user"] = user
    entry["updated"] = time.time()
    if title and not entry.get("title"):
        entry["title"] = title[:80]
    _index_file.write_text(json.dumps(index, ensure_ascii=False, indent=1))


async def _items(conversation_id: str) -> list:
    """A conversation's chat items, from its loaded session or else the store."""
    if (session := _live.get(conversation_id)) is not None:
        return list(session.history.items)
    # not loaded, so read what the last close saved; this stamps the row closed
    # again, and leaves a new conversation an empty row that loads as new
    stored = store.session(conversation_id, conversation_id, endpoint=A2A_ENDPOINT)
    try:
        record = await stored.load()
        return list(record.history) if record else []
    except StoreError:
        logger.warning("could not read conversation %s", conversation_id)
        return []
    finally:
        with contextlib.suppress(StoreError):
            await stored.release()


class _Turn:
    """One message's turn, read to its end in the background."""

    def __init__(self, stream: TaskStream, text: str) -> None:
        self.stream = stream
        # what the person sent, shown until the session has recorded it
        self.text = text
        self.sent_at = time.time()
        # a newer message or a stop took over: nothing it does reaches the phone
        self.quiet = False
        # its reply was stopped, while the tools it started may run on
        self.interrupted = False
        # a tool result arrived after that, so the answer it brings is worth a push
        self.late = False
        self.reader: asyncio.Task[None] | None = None
        # its step lines and its answer, for a sender that waits for them
        self.reply: asyncio.Future[tuple[list[str], str]] = (
            asyncio.get_running_loop().create_future()
        )

    @property
    def running(self) -> bool:
        return self.reader is not None and not self.reader.done()

    async def acked(self, timeout: float = 5) -> None:
        """Wait until the server has named the task, which its first event does."""
        for _ in range(int(timeout / 0.1)):
            if self.stream.task_id or not self.running:
                return
            await asyncio.sleep(0.1)


class _Conversation:
    """One person's conversation, kept until renewed or left idle."""

    def __init__(self, url: str, headers: dict[str, str], user: str | None) -> None:
        self._url = url
        self._headers = headers
        self._user = user
        # no one in particular keeps the file from before there were people
        name = "current_conversation"
        self._current_file = _data_dir / (
            f"{name}.{quote(user, safe='')}" if user else name
        )
        self._client: A2AClient | None = None
        self._conversation_id: str | None = None
        self._lock = asyncio.Lock()
        self._progress = _PhoneProgress()
        # the buttons on the latest answer notification: action id -> reply text
        self._actions: dict[str, str] = {}
        self._reply_action = ""
        self._listener: asyncio.Task[None] | None = None
        # background work: notification taps, and the cancels of superseded turns
        self._chores: set[asyncio.Task[object]] = set()
        # the latest turn; its task id is set by its first event
        self._turn: _Turn | None = None

    async def _open(self, renew: bool) -> tuple[A2AClient, str, bool]:
        """The client for this turn, its conversation, and whether that one is new."""
        if self._conversation_id is None and self._current_file.exists():
            self._conversation_id = self._current_file.read_text().strip() or None
        if self._conversation_id is not None and settings.text_renew_after > 0:
            # touched on every turn, so the file's age is how long the user was away
            idle = time.time() - self._current_file.stat().st_mtime
            renew = renew or idle > settings.text_renew_after
        if renew and self._client is not None:
            # the goodbye closes the old context on the server, which saves it
            await self._client.aclose()
            self._client = None
        started = renew or self._conversation_id is None
        if started:
            self._conversation_id = await store.create_database()
            self._current_file.write_text(self._conversation_id)
            _record(self._conversation_id, self._user)
            logger.info("started text conversation %s", self._conversation_id)
        assert self._conversation_id is not None
        _owners[self._conversation_id] = self._user
        self._current_file.touch()
        if self._client is None:
            # the agent is the only one in the conversation, so its session is the front
            # session and the conversation id doubles as the A2A context id
            self._client = A2AClient(
                self._url, context_id=self._conversation_id, headers=self._headers
            )
        return self._client, self._conversation_id, started

    async def send(
        self,
        text: str,
        *,
        renew: bool = False,
        steps: bool = True,
        wait: bool = False,
        interrupt: bool = True,
    ) -> tuple[str, str]:
        """Start one user turn and return (conversation id, what to answer the sender).

        The turn is read to its end in the background, and its answer goes to the
        person's phone. With ``wait``, the answer is returned too, preceded with
        ``steps`` by one line per tool call. With ``interrupt``, the turn replaces a
        running one; without, it starts once that one has ended.
        """
        while not interrupt and self._turn is not None and self._turn.running:
            # an update never cuts the person off: it waits for their turn to end
            await asyncio.wait([self._turn.reader])
        async with self._lock:
            superseding = False
            if (previous := self._turn) is not None:
                if interrupt and previous.running and not previous.quiet:
                    # the new turn interrupts it on the server before it runs
                    previous.interrupted = superseding = True
                # a context's messages are ordered only once the previous one has a task
                await previous.acked()
            client, conversation_id, started = await self._open(renew)
            if not text:
                return conversation_id, "Started a new conversation."
            _record(conversation_id, self._user, text)
            _suggested.setdefault(conversation_id, []).clear()
            self._progress.target = await scheduler.phone(self._user)
            if self._progress.target:
                self._progress.begin(text)
                if self._listener is None or self._listener.done():
                    self._listener = asyncio.create_task(self._listen())
            task_input = TaskInput(
                text=text,
                conversation_id=conversation_id,
                interrupting=[] if superseding else None,
            )
            turn = self._turn = _Turn(client.send(task_input), text)
            turn.reader = asyncio.create_task(self._read(turn, conversation_id))
        if not wait:
            return conversation_id, "accepted"
        try:
            lines, answer = await asyncio.wait_for(
                asyncio.shield(turn.reply), settings.text_reply_timeout
            )
        except TimeoutError:
            # only the waiting stops: the turn runs on and its answer still goes out
            return conversation_id, "[timeout] still working"
        if not steps:
            return conversation_id, answer
        lines = ["(new conversation)", *lines] if started else lines
        return conversation_id, "\n".join([*lines, "", answer] if lines else [answer])

    async def _read(self, turn: _Turn, conversation_id: str) -> None:
        """Read a turn's updates to its end: its steps, then its answer."""
        phone = self._progress if self._progress.target else None
        lines: list[str] = []
        said: list[str] = []
        answer, ok = "", True
        try:
            async with turn.stream as stream:
                async for update in stream:
                    item = update.item
                    if (
                        turn.interrupted
                        and item
                        and item.type == "function_call_output"
                    ):
                        turn.late = True
                    silent = turn.quiet or (turn.interrupted and not turn.late)
                    live = None if silent else phone
                    if item is None or item.type != "function_call":
                        if update.text:
                            said.append(update.text)
                    elif item.update_of:
                        # a tool's progress report is a call naming the call it reports
                        # on, with the report as the update's text
                        lines.append(f"  … {update.text}")
                        if live:
                            live.report(update.text)
                    else:
                        try:
                            args = json.loads(item.arguments or "{}")
                        except ValueError:
                            args = item.arguments
                        shown = json.dumps(args, ensure_ascii=False)
                        lines.append(f"→ {item.name}({shown.removeprefix('{}')[:120]})")
                        if live:
                            live.step(item.name, args)
                    if update.state != "working":
                        answer = update.text or (said[-1] if said else "")
                        if update.state in ("failed", "canceled"):
                            answer, ok = f"[{update.state}] {answer}", False
                        break
                else:
                    # a cancelled task can close its stream with no last word
                    state = "canceled" if turn.quiet else "ended"
                    answer = "\n".join([f"[{state}]", *said])
                    ok = False
        except Exception as exc:  # noqa: BLE001 - a failed turn still answers
            logger.exception("text turn failed")
            answer, ok = f"[error] {exc}", False
        finally:
            if not turn.reply.done():
                turn.reply.set_result((lines, answer))
        silent = turn.quiet or (turn.interrupted and not turn.late)
        if silent or self._progress.target == "":
            return
        # the suggested replies become buttons, and Reply takes free text
        nonce = uuid.uuid4().hex[:8]
        replies = list(dict.fromkeys(_suggested.get(conversation_id, [])))[:4]
        self._actions = {f"HATEXT_{nonce}_{i}": r for i, r in enumerate(replies)}
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
        self._progress.finish(answer, ok=ok, actions=actions)

    def _stop(self, turn: _Turn | None, reason: str, *, force: bool) -> bool:
        """Stop a running turn's reply in the background; with ``force``, the tools it
        started as well."""
        if turn is None or not turn.running or turn.quiet:
            return False
        if force:
            turn.quiet = True
        elif turn.interrupted:
            return False
        else:
            turn.interrupted = True
        chore = asyncio.create_task(self._halt(turn, reason, force=force))
        self._chores.add(chore)
        chore.add_done_callback(self._chores.discard)
        return True

    async def _halt(self, turn: _Turn, reason: str, *, force: bool) -> None:
        # the task has no id to stop it by until its first event arrives
        await turn.acked()
        if turn.running and turn.stream.task_id:
            with contextlib.suppress(Exception):
                if force:
                    await turn.stream.cancel(reason)
                else:
                    await turn.stream.interrupt()

    def warm(self) -> bool:
        """Start the current conversation's session in the background, with no turn, so
        the next message does not wait for it; False when there is nothing to start."""
        conversation_id = self._current()
        if conversation_id is None or conversation_id in _live:
            return False
        if settings.text_renew_after > 0 and self._current_file.exists():
            idle = time.time() - self._current_file.stat().st_mtime
            if idle > settings.text_renew_after:
                return False  # the next message starts a new conversation instead
        _owners[conversation_id] = self._user
        chore = asyncio.create_task(self._warm(conversation_id))
        self._chores.add(chore)
        chore.add_done_callback(self._chores.discard)
        return True

    async def _warm(self, conversation_id: str) -> None:
        logger.info("warming text conversation %s", conversation_id)
        async with self._lock:
            if self._client is None and self._current() == conversation_id:
                self._client = A2AClient(
                    self._url, context_id=conversation_id, headers=self._headers
                )
            client = self._client
        if client is None:
            return
        try:
            await client.prewarm(conversation_id=conversation_id)
        except Exception:
            logger.exception("could not warm text conversation %s", conversation_id)

    def cancel(self, task_id: str | None, *, force: bool = False) -> bool:
        """Stop the running turn's reply, and with ``force`` the tools it started too;
        naming a task never stops a newer one."""
        turn = self._turn
        if turn is None or (task_id and task_id != turn.stream.task_id):
            return False
        return self._stop(turn, "stopped by the user", force=force)

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
                self._chores.add(task)
                task.add_done_callback(self._chores.discard)

    def _current(self) -> str | None:
        if self._conversation_id is None and self._current_file.exists():
            return self._current_file.read_text().strip() or None
        return self._conversation_id

    def _owns(self, conversation_id: str) -> bool:
        entry = _index().get(conversation_id)
        return entry is not None and entry.get("user") == self._user

    async def conversations(self) -> dict[str, object]:
        """This person's conversations, latest first, titled by their first message."""
        index = _index()
        mine = {k: v for k, v in index.items() if v.get("user") == self._user}
        for conversation_id, entry in mine.items():
            if "title" in entry:
                continue
            # recorded before titles were, or never spoken in: read the first message
            first = next(
                (
                    i.text_content
                    for i in await _items(conversation_id)
                    if i.type == "message" and i.role == "user" and i.text_content
                ),
                None,
            )
            # an empty title marks one never spoken in, so it is read only once
            entry["title"] = (first or "")[:80]
            index[conversation_id] = entry
            _index_file.write_text(json.dumps(index, ensure_ascii=False, indent=1))
        rows = [
            {
                "id": k,
                "title": v.get("title") or "",
                "created": int(float(v.get("created") or 0) * 1000),
                "updated": int(float(v.get("updated") or 0) * 1000),
            }
            for k, v in mine.items()
        ]
        rows.sort(key=lambda r: r["updated"], reverse=True)
        return {"current": self._current(), "conversations": rows}

    async def switch(self, conversation_id: str) -> bool:
        """Make one of this person's conversations the current one."""
        busy = self._turn is not None and self._turn.running
        if not self._owns(conversation_id) or busy or self._lock.locked():
            return False
        async with self._lock:
            if conversation_id == self._current():
                return True
            if self._client is not None:
                # the goodbye closes the old context on the server, which saves it
                await self._client.aclose()
                self._client = None
            self._conversation_id = conversation_id
            self._current_file.write_text(conversation_id)
            _record(conversation_id, self._user)
            logger.info("switched to text conversation %s", conversation_id)
            return True

    async def delete(self, conversation_id: str) -> bool:
        """Delete one of this person's past conversations that is not live."""
        if not self._owns(conversation_id) or conversation_id == self._current():
            return False
        if conversation_id in _live:
            return False
        # the store keeps a connection per database it has read
        if (database := store._databases.pop(conversation_id, None)) is not None:
            await database.aclose()
        for path in (_data_dir / "sessions").glob(f"{conversation_id}.sqlite*"):
            path.unlink(missing_ok=True)
        index = _index()
        index.pop(conversation_id, None)
        _index_file.write_text(json.dumps(index, ensure_ascii=False, indent=1))
        _suggested.pop(conversation_id, None)
        _owners.pop(conversation_id, None)
        logger.info("deleted text conversation %s", conversation_id)
        return True

    async def history(
        self, limit: int, conversation_id: str | None = None
    ) -> dict[str, object] | None:
        """A conversation's latest ``limit`` messages and tool calls, shaped like the
        card's conversation items: the current one, or another of this person's."""
        current = self._current()
        if conversation_id and conversation_id != current:
            if not self._owns(conversation_id):
                return None
        else:
            conversation_id = current
        items = await _items(conversation_id) if conversation_id else []
        live = conversation_id == current
        turn = self._turn if self._turn is not None and self._turn.running else None

        outputs = {i.call_id: i for i in items if i.type == "function_call_output"}
        shown: list[dict[str, object]] = []
        # the quick replies offered since the user last spoke, for a UI's chips
        suggestions: list[str] = []
        for item in items:
            ts = int(item.created_at * 1000)
            text = item.text_content if item.type == "message" else None
            if text and item.role == "user" and text.startswith(UPDATE_PREFIX):
                # an update is not the person: shown as a closed row, like a tool's
                source, _, body = (
                    text.removeprefix(UPDATE_PREFIX).strip().partition(": ")
                )
                shown.append(
                    {"kind": "action", "id": item.id, "call_id": item.id, "ts": ts}
                    | {"name": "update_from", "args": {"name": source}}
                    | {"status": "done", "output": body[:1000]}
                )
                continue
            if item.type == "message" and item.role in ("user", "assistant"):
                if text:
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
                # clipped, so a home's device list does not ride on every poll
                if out is not None:
                    action["output"] = out.output[:1000]
                    if len(out.output) > 1000:
                        action["output_chars"] = len(out.output)
                shown.append(action)
                if item.name == "suggest_replies" and isinstance(args, dict):
                    suggestions = [str(r) for r in args.get("replies") or []]
        if (
            live
            and turn is not None
            # an update turn's message is not the person's, and becomes a row
            and not turn.text.startswith(UPDATE_PREFIX)
            and not any(
                i["kind"] == "message"
                and i["role"] == "user"
                and i["text"] == turn.text
                and int(i["ts"]) >= int(turn.sent_at * 1000) - 1000
                for i in shown
            )
        ):
            # a conversation still loading has not recorded the message yet
            shown.append(
                {"kind": "message", "id": f"pending-{int(turn.sent_at * 1000)}"}
                | {"role": "user", "text": turn.text, "ts": int(turn.sent_at * 1000)}
            )
            suggestions = []
        return {
            "conversation_id": conversation_id,
            "current": live,
            "busy": live and turn is not None,
            # the running turn's task, which POST /chat/cancel takes
            "task_id": turn.stream.task_id or None if live and turn else None,
            "items": shown[-limit:],
            "suggestions": suggestions,
        }


class _PhoneProgress:
    """A text turn's progress on the phone, sent in order as the turn moves.

    In ``activity`` mode each phase goes to the livekit_voice integration's progress
    API, which runs inside Home Assistant and can see whether the phone's Live Activity
    is running: it starts or updates the activity, and sends the answer as a
    notification with sound only when the activity cannot carry it. In
    ``notification`` mode it is one notification titled with the question, under a
    fixed tag: the question and the answer pop with sound, and each step is appended
    as a passive update.
    """

    TAG = "ha-text"

    def __init__(self) -> None:
        # the notify service of the phone showing it, set before each turn
        self.target = ""
        self._activity = settings.text_live_mode == "activity"
        self._question = ""
        self._steps: list[str] = []
        # sent in order by one sender: (message, title, data) or a progress API body
        self._outbox: asyncio.Queue[tuple[str, str, dict[str, object]] | dict] = (
            asyncio.Queue()
        )
        self._sender: asyncio.Task[None] | None = None

    def begin(self, question: str) -> None:
        self._question = question if len(question) <= 60 else f"{question[:59]}…"
        self._steps = []
        if self._activity:
            self._phase("start", f"› {self._question}", "…", "mdi:robot")
        else:
            self._notify(["…"], pop=True)

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
        if self._activity:
            self._phase("progress", lines[-1], f"Step {len(self._steps)}", icon)
        else:
            self._notify(lines)

    def report(self, text: str) -> None:
        if not text:
            return
        if self._activity:
            status = f"Step {len(self._steps)}"
            self._phase("progress", f"… {text}", status, "mdi:progress-clock")
        else:
            self._notify([*(f"→ {s}" for s in self._steps), f"  … {text}"])

    def finish(
        self, answer: str, *, ok: bool, actions: list[dict[str, object]]
    ) -> None:
        answer = answer or "Done."
        if self._activity:
            icon = "mdi:check-circle" if ok else "mdi:alert-circle"
            self._phase(
                "final",
                answer,
                "Done" if ok else "Failed",
                icon,
                answer=answer,
                question=self._question,
                actions=actions,
                clear_after=settings.text_live_clear_after,
            )
        else:
            self._notify([answer], pop=True, actions=actions)

    def _phase(
        self, phase: str, message: str, status: str, icon: str, **extra: object
    ) -> None:
        body: dict[str, object] = {
            "phase": phase,
            "target": self.target,
            "tag": self.TAG,
            "message": message,
            "status": status,
            "icon": icon,
            **extra,
        }
        if settings.text_live_url:
            body["url"] = settings.text_live_url
        self._queue(body)

    def _notify(
        self,
        lines: list[str],
        *,
        pop: bool = False,
        actions: list[dict[str, object]] | None = None,
    ) -> None:
        data: dict[str, object] = {"tag": self.TAG}
        if settings.text_live_url:
            data["url"] = settings.text_live_url
        if actions:
            data["actions"] = actions
        if pop:
            # a replacement under one tag updates silently, so clear it to pop again
            self._queue(("clear_notification", "", {"tag": self.TAG}))
        else:
            # appended to the question's notification, or, once that is gone,
            # delivered without a banner or a sound
            data["push"] = {"interruption-level": "passive", "sound": "none"}
        self._queue(("\n".join(lines), self._question, data))

    def _queue(self, item: tuple[str, str, dict[str, object]] | dict) -> None:
        self._outbox.put_nowait(item)
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(self._send_all())

    async def _send_all(self) -> None:
        while True:
            item = await self._outbox.get()
            if isinstance(item, dict):
                await ha.progress(item)
            else:
                message, title, data = item
                await ha.push(self.target, message, data, title=title)


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

    conversations: dict[str | None, _Conversation] = {}

    async def conversation_for(name: object, ha_user_id: object) -> _Conversation:
        """The conversation of the person a request names, or its HA login's."""
        user = await scheduler.resolve_user(
            str(name) if name else None, str(ha_user_id) if ha_user_id else None
        )
        if user not in conversations:
            conversations[user] = _Conversation(
                f"http://127.0.0.1:{settings.http_port}/{A2A_ENDPOINT}",
                headers={"Authorization": f"Bearer {token}"},
                user=user,
            )
        return conversations[user]

    @server.http.post(CHAT_PATH)
    async def chat(request: Request) -> Response:
        """Take `{"text", "new", "steps", "wait", "user"}` JSON or plain text.

        Answers 202 once the turn has started, and the answer goes to the person's
        phone; with `wait` it answers with the agent's reply instead."""
        body = (await request.body()).decode().strip()
        text, renew, steps, wait = body, False, True, False
        data: dict[str, object] = {}
        if request.headers.get("content-type", "").startswith("application/json"):
            data = json.loads(body or "{}")
            text = str(data.get("text") or "").strip()
            renew = bool(data.get("new"))
            steps = bool(data.get("steps", True))
            wait = bool(data.get("wait"))
        query = request.query_params
        conversation = await conversation_for(
            data.get("user") or query.get("user"),
            data.get("ha_user_id") or query.get("ha_user_id"),
        )
        renew = renew or query.get("new", "") in ("1", "true", "yes")
        steps = steps and query.get("steps", "") not in ("0", "false", "no")
        wait = wait or query.get("wait", "") in ("1", "true", "yes")
        if not text and not renew:
            return PlainTextResponse("no text given", status_code=400)
        try:
            conversation_id, reply = await conversation.send(
                text, renew=renew, steps=steps, wait=wait
            )
        except Exception as exc:
            logger.exception("text chat failed")
            return PlainTextResponse(f"[error] {exc}", status_code=502)
        status = 200 if wait or not text else 202
        headers = {"X-Conversation-Id": conversation_id}
        return PlainTextResponse(reply, status_code=status, headers=headers)

    @server.http.get(f"{CHAT_PATH}/history")
    async def chat_history(
        limit: int = 200,
        user: str = "",
        ha_user_id: str = "",
        conversation_id: str = "",
    ) -> JSONResponse:
        """A person's current conversation, or one they name, for a UI; `busy` while
        a turn runs."""
        conversation = await conversation_for(user, ha_user_id)
        history = await conversation.history(limit, conversation_id or None)
        if history is None:
            return JSONResponse({"detail": "no such conversation"}, status_code=404)
        return JSONResponse(history)

    @server.http.get(f"{CHAT_PATH}/conversations")
    async def chat_conversations(user: str = "", ha_user_id: str = "") -> JSONResponse:
        """A person's conversations, latest first, and which one is current."""
        conversation = await conversation_for(user, ha_user_id)
        return JSONResponse(await conversation.conversations())

    @server.http.post(f"{CHAT_PATH}/switch")
    async def chat_switch(request: Request) -> JSONResponse:
        """Make `{"conversation_id", "user"}` the person's current conversation."""
        body = (await request.body()).decode().strip()
        data = json.loads(body or "{}")
        conversation = await conversation_for(data.get("user"), data.get("ha_user_id"))
        target = str(data.get("conversation_id") or "")
        return JSONResponse({"switched": await conversation.switch(target)})

    @server.http.post(f"{CHAT_PATH}/delete")
    async def chat_delete(request: Request) -> JSONResponse:
        """Delete `{"conversation_id", "user"}`, a person's past conversation."""
        body = (await request.body()).decode().strip()
        data = json.loads(body or "{}")
        conversation = await conversation_for(data.get("user"), data.get("ha_user_id"))
        target = str(data.get("conversation_id") or "")
        return JSONResponse({"deleted": await conversation.delete(target)})

    @server.http.post(f"{CHAT_PATH}/cancel")
    async def chat_cancel(request: Request) -> JSONResponse:
        """Stop a person's running turn's reply, answering at once: `{"task_id",
        "user"}`, the task from history or none for whichever runs; `"force": true`
        stops the tools it started as well."""
        body = (await request.body()).decode().strip()
        data = json.loads(body or "{}")
        conversation = await conversation_for(data.get("user"), data.get("ha_user_id"))
        task_id = str(data.get("task_id") or "") or None
        cancelled = conversation.cancel(task_id, force=bool(data.get("force")))
        # a client cancels as the person starts to speak, so their message is coming
        return JSONResponse({"cancelled": cancelled, "warming": conversation.warm()})

    @server.http.post(f"{CHAT_PATH}/warm")
    async def chat_warm(request: Request) -> JSONResponse:
        """Load a person's current conversation ahead of their message: `{"user"}`."""
        body = (await request.body()).decode().strip()
        data = json.loads(body or "{}")
        conversation = await conversation_for(data.get("user"), data.get("ha_user_id"))
        return JSONResponse({"warming": conversation.warm()})

    updates: set[asyncio.Task[object]] = set()

    @server.http.post(f"{CHAT_PATH}/events")
    async def chat_event(request: Request) -> Response:
        """Take a webhook's `{"source", "text"}` into a person's conversation as a turn
        the agent answers with no tools, e.g. a Claude Code task that finished."""
        data = json.loads((await request.body()).decode() or "{}")
        text = str(data.get("text") or "").strip()
        if not text:
            return PlainTextResponse("no text given", status_code=400)
        query = request.query_params
        user, ha_user_id = query.get("user"), query.get("ha_user_id")
        conversation = await conversation_for(user, ha_user_id)
        source = str(data.get("source") or "a webhook")
        logger.info("update from %s: %s", source, text[:120])
        # answered in the background, so the sender is not held for a whole turn
        turn = asyncio.create_task(
            conversation.send(f"{UPDATE_PREFIX} {source}: {text}", interrupt=False)
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
        session = AgentSession(llm=inference.LLM(settings.llm_model), max_tool_steps=8)
        agent = HomeAssistantAgent(user=_owners.get(ctx.context_id))
        suggested = _suggested.setdefault(ctx.context_id, [])
        agent._suggest_replies_cb = suggested.extend
        # None when the caller names no conversation, which then lives in memory only
        await session.start(agent=agent, persist=ctx.persisted)
        # a loaded conversation keeps only its latest items, so the prompt stays bounded
        await agent.update_chat_ctx(
            agent.chat_ctx.copy().truncate(max_items=settings.text_max_items)
        )
        ctx.attach(session)
        _live[ctx.context_id] = session
        session.on("close", lambda _: _live.pop(ctx.context_id, None))
