"""One person's text chat: their current conversation over A2A, its running turn, and
that turn's progress for their phone."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, Literal

from livekit.agents.a2a import A2AClient, TaskInput, TaskStream

from config import settings
from conversations import conversations
from history import render
from scheduler_client import scheduler
from utils import parse_arguments

logger = logging.getLogger("ha-mcp-agent.text")


class Progress:
    """Forwards a person's turn phases to their phone, in order, never holding up
    the turn."""

    def __init__(self, user_id: str | None) -> None:
        self._user_id = user_id
        self._outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._sender: asyncio.Task[None] | None = None

    def start(self, question: str) -> None:
        self._send(phase="start", text=question)

    def step(self, tool: str, args: object) -> None:
        self._send(phase="step", tool=tool, args=args)

    def report(self, text: str) -> None:
        self._send(phase="report", text=text)

    def final(self, answer: str, *, ok: bool, replies: list[str]) -> None:
        self._send(phase="final", text=answer, ok=ok, replies=replies)

    def _send(self, **event: Any) -> None:
        self._outbox.put_nowait({"user": self._user_id, **event})
        if self._sender is None or self._sender.done():
            self._sender = asyncio.create_task(self._send_all())

    async def _send_all(self) -> None:
        while not self._outbox.empty():
            await scheduler.progress(self._outbox.get_nowait())


class Turn:
    """One message's turn, read to its end in the background."""

    def __init__(self, stream: TaskStream, text: str) -> None:
        self.stream = stream
        # what the person sent, shown until the session has recorded it
        self.text = text
        self.sent_at = time.time()
        # "interrupted": its reply was stopped, while the tools it started may run on;
        # "cancelled": it was stopped with its tools, and nothing it does is shown
        self.stopped: Literal["interrupted", "cancelled"] | None = None
        # a tool result arrived after an interrupt, so the answer it brings is shown
        self.late = False
        self.reader: asyncio.Task[None] | None = None
        # its step lines and its answer, for a sender that waits for them
        self.reply: asyncio.Future[tuple[list[str], str]] = (
            asyncio.get_running_loop().create_future()
        )

    @property
    def running(self) -> bool:
        return self.reader is not None and not self.reader.done()

    @property
    def silent(self) -> bool:
        """Whether what it does no longer reaches the phone."""
        return self.stopped == "cancelled" or (
            self.stopped == "interrupted" and not self.late
        )

    async def acked(self, timeout: float = 5) -> None:
        """Wait until the server has named the task, which its first event does."""
        for _ in range(int(timeout / 0.1)):
            if self.stream.task_id or not self.running:
                return
            await asyncio.sleep(0.1)

    async def read(self, progress: Progress) -> tuple[str, bool]:
        """Read the turn's updates to its end, showing them as they come; returns its
        answer and whether it succeeded."""
        lines: list[str] = []
        said: list[str] = []
        answer, ok = "", True
        try:
            async with self.stream as stream:
                async for update in stream:
                    item = update.item
                    if self.stopped and item and item.type == "function_call_output":
                        self.late = True
                    if item is None or item.type != "function_call":
                        if update.text:
                            said.append(update.text)
                    elif item.update_of:
                        # a tool's progress report is a call naming the call it reports
                        # on, with the report as the update's text
                        lines.append(f"  … {update.text}")
                        if not self.silent and update.text:
                            progress.report(update.text)
                    else:
                        args = parse_arguments(item.arguments)
                        shown = json.dumps(args, ensure_ascii=False)
                        lines.append(f"→ {item.name}({shown.removeprefix('{}')[:120]})")
                        if not self.silent:
                            progress.step(item.name, args)
                    if update.state != "working":
                        answer = update.text or (said[-1] if said else "")
                        if update.state in ("failed", "canceled"):
                            answer, ok = f"[{update.state}] {answer}", False
                        break
                else:
                    # a cancelled task can close its stream with no last word
                    state = "canceled" if self.stopped == "cancelled" else "ended"
                    answer, ok = "\n".join([f"[{state}]", *said]), False
        except Exception as exc:  # noqa: BLE001 - a failed turn still answers
            logger.exception("text turn failed")
            answer, ok = f"[error] {exc}", False
        finally:
            if not self.reply.done():
                self.reply.set_result((lines, answer))
        return answer, ok


class TextChat:
    def __init__(self, user_id: str | None, url: str, headers: dict[str, str]) -> None:
        self.user_id = user_id
        self._url = url
        self._headers = headers
        # bound to the current conversation; closed when another becomes current
        self._client: A2AClient | None = None
        self._lock = asyncio.Lock()
        self._progress = Progress(user_id)
        # the latest turn; its task id is set by its first event
        self._turn: Turn | None = None
        # background work, such as stopping a turn once the server has named it
        self._chores: set[asyncio.Task[Any]] = set()

    @property
    def busy(self) -> bool:
        return self._turn is not None and self._turn.running

    def _spawn(self, coro: Any) -> None:
        chore = asyncio.create_task(coro)
        self._chores.add(chore)
        chore.add_done_callback(self._chores.discard)

    async def _close_client(self) -> None:
        if self._client is not None:
            # the goodbye closes the old context on the server, which saves it
            await self._client.aclose()
            self._client = None

    async def _open(self, renew: bool) -> tuple[A2AClient, str, bool]:
        """The client for this turn, its conversation, and whether that one is new."""
        current = conversations.current(self.user_id)
        renew = renew or current is None or conversations.stale(current)
        if renew:
            await self._close_client()
            current = await conversations.create(self.user_id)
        assert current is not None
        if self._client is None:
            # the agent is the only one in the conversation, so its session is the front
            # session and the conversation id doubles as the A2A context id
            self._client = A2AClient(
                self._url, context_id=current, headers=self._headers
            )
        return self._client, current, renew

    async def send(
        self,
        text: str,
        *,
        renew: bool = False,
        steps: bool = True,
        wait: bool = False,
        interrupt: bool = True,
    ) -> tuple[str, str]:
        """Start one turn and return (conversation id, what to answer the sender).

        The turn is read to its end in the background, and its progress and answer go to
        the person's phone. With ``wait``, the answer is returned too, preceded with
        ``steps`` by one line per tool call. With ``interrupt``, the turn replaces a
        running one; without, it starts once that one has ended.
        """
        while not interrupt and self._turn is not None and self._turn.running:
            # an update never cuts the person off: it waits for their turn to end
            await asyncio.wait([self._turn.reader])
        async with self._lock:
            superseding = False
            if (previous := self._turn) is not None:
                if interrupt and previous.running and previous.stopped != "cancelled":
                    # the new turn interrupts it on the server before it runs
                    previous.stopped = "interrupted"
                    superseding = True
                # a context's messages are ordered only once the previous one has a task
                await previous.acked()
            client, conversation_id, started = await self._open(renew)
            if not text:
                return conversation_id, "Started a new conversation."
            conversations.record(conversation_id, text)
            conversations.suggested.setdefault(conversation_id, []).clear()
            self._progress.start(text)
            task_input = TaskInput(
                text=text,
                conversation_id=conversation_id,
                interrupting=[] if superseding else None,
            )
            turn = self._turn = Turn(client.send(task_input), text)
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

    async def _read(self, turn: Turn, conversation_id: str) -> None:
        answer, ok = await turn.read(self._progress)
        if not turn.silent:
            replies = list(
                dict.fromkeys(conversations.suggested.get(conversation_id, []))
            )
            self._progress.final(answer, ok=ok, replies=replies)

    def cancel(self, task_id: str | None, *, force: bool = False) -> bool:
        """Stop the running turn's reply, and with ``force`` the tools it started too;
        naming a task never stops a newer one."""
        turn = self._turn
        if turn is None or not turn.running or turn.stopped == "cancelled":
            return False
        if task_id and task_id != turn.stream.task_id:
            return False
        if force:
            turn.stopped = "cancelled"
        elif turn.stopped == "interrupted":
            return False
        else:
            turn.stopped = "interrupted"
        self._spawn(self._halt(turn, force=force))
        return True

    async def _halt(self, turn: Turn, *, force: bool) -> None:
        # the task has no id to stop it by until its first event arrives
        await turn.acked()
        if turn.running and turn.stream.task_id:
            with contextlib.suppress(Exception):
                if force:
                    await turn.stream.cancel("stopped by the user")
                else:
                    await turn.stream.interrupt()

    def warm(self) -> bool:
        """Start the current conversation's session in the background, with no turn, so
        the next message does not wait for it; False when there is nothing to start."""
        current = conversations.current(self.user_id)
        if current is None or current in conversations.live:
            return False
        if conversations.stale(current):
            return False  # the next message starts a new conversation instead
        self._spawn(self._warm(current))
        return True

    async def _warm(self, conversation_id: str) -> None:
        logger.info("warming text conversation %s", conversation_id)
        async with self._lock:
            if self._client is None and conversations.current(self.user_id) == (
                conversation_id
            ):
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

    async def switch(self, conversation_id: str) -> bool:
        """Make one of this person's conversations the current one."""
        if not conversations.owns(self.user_id, conversation_id):
            return False
        if self.busy or self._lock.locked():
            return False
        async with self._lock:
            if conversation_id != conversations.current(self.user_id):
                await self._close_client()
                conversations.set_current(self.user_id, conversation_id)
                logger.info("switched to text conversation %s", conversation_id)
            return True

    async def delete(self, conversation_id: str) -> bool:
        """Delete one of this person's past conversations that is not loaded."""
        if not conversations.owns(self.user_id, conversation_id):
            return False
        if conversation_id == conversations.current(self.user_id):
            return False
        if conversation_id in conversations.live:
            return False
        await conversations.delete(conversation_id)
        return True

    async def listing(self) -> dict[str, Any]:
        """This person's conversations, latest first, and which one is current."""
        return {
            "current": conversations.current(self.user_id),
            "conversations": await conversations.listing(self.user_id),
        }

    async def history(
        self, limit: int, conversation_id: str | None = None
    ) -> dict[str, Any] | None:
        """A conversation's latest ``limit`` messages and tool calls, shaped like the
        card's conversation items: the current one, or another of this person's."""
        current = conversations.current(self.user_id)
        if conversation_id and conversation_id != current:
            if not conversations.owns(self.user_id, conversation_id):
                return None
        else:
            conversation_id = current
        live = conversation_id is not None and conversation_id == current
        turn = self._turn if live and self.busy else None
        items = await conversations.items(conversation_id) if conversation_id else []
        pending = (turn.text, turn.sent_at) if turn else None
        shown, suggestions = render(items, pending)
        return {
            "conversation_id": conversation_id,
            "current": live,
            "busy": turn is not None,
            # the running turn's task, which POST /chat/cancel takes
            "task_id": (turn.stream.task_id or None) if turn else None,
            "items": shown[-limit:],
            "suggestions": suggestions,
            # the LLM tokens it has used, cached input counted within input
            "usage": conversations.usage(conversation_id) if conversation_id else None,
        }
