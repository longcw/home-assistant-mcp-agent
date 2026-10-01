"""One message's turn, read from its A2A task stream, and its progress for the phone."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Literal

from livekit.agents.a2a import TaskStream

import scheduler_client as scheduler
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
