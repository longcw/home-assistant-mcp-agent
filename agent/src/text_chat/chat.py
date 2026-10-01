"""One person's text chat: their current conversation over A2A, and its running turn."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from livekit.agents.a2a import A2AClient, TaskInput

from config import settings
from conversations import conversations
from history import render
from text_chat.turn import Progress, Turn

logger = logging.getLogger("ha-mcp-agent.text")


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
