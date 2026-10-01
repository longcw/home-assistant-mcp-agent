"""Every person's conversations: one SQLite session file each, and one index.

A conversation belongs to no mode: whichever session loads it, text or voice, joins
it through ``attach``. The index, ``conversations.json``, holds each conversation's
owner, title, last turn and LLM token totals, and each person's current conversation,
so a person can list and reopen theirs. People are only ids here; no one in
particular is the empty id.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from livekit.agents import Agent, AgentSession, SessionUsageUpdatedEvent
from livekit.agents.metrics import LLMModelUsage
from livekit.agents.store import LocalStore, StoredSession, StoreError

from config import settings

logger = logging.getLogger("ha-mcp-agent.text")

# the endpoint every stored session is filed under, the text chat's A2A one
A2A_ENDPOINT = "home-assistant"
# a conversation's token totals, by their key in the index and LLMModelUsage's field
_USAGE_FIELDS = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cached": "input_cached_tokens",
}


class Conversations:
    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir
        self._file = data_dir / "conversations.json"
        # one SQLite file per conversation; handed to AgentServer(store=...) in main.py
        self.store = LocalStore(data_dir / "sessions")
        # the sessions the A2A endpoint has loaded, by conversation
        self.live: dict[str, AgentSession] = {}
        # what the agent's suggest_replies offered in each conversation's running turn
        self.suggested: dict[str, list[str]] = {}
        # unloaded conversations as last read, with their files' stamp then
        self._read: dict[str, tuple[tuple[int, ...], list]] = {}
        self._data: dict[str, Any] | None = None

    def _index(self) -> dict[str, Any]:
        if self._data is not None:
            return self._data
        try:
            data = json.loads(self._file.read_text())
        except (OSError, ValueError):
            data = {}
        if "conversations" not in data:
            # the flat index, with each person's current one in a file of its own
            data = {"conversations": data, "current": {}}
            old = list(self._dir.glob("current_conversation*"))
            for path in old:
                _, _, quoted = path.name.partition(".")
                if conversation_id := path.read_text().strip():
                    data["current"][unquote(quoted)] = conversation_id
            self._data = data
            self._save()
            for path in old:
                path.unlink(missing_ok=True)
        self._data = data
        return data

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self._index(), ensure_ascii=False, indent=1)
        self._file.write_text(text)

    def _entry(self, conversation_id: str) -> dict[str, Any]:
        return self._index()["conversations"].setdefault(
            conversation_id, {"created": time.time()}
        )

    def current(self, user_id: str | None) -> str | None:
        return self._index()["current"].get(user_id or "")

    def set_current(self, user_id: str | None, conversation_id: str) -> None:
        self._index()["current"][user_id or ""] = conversation_id
        self._entry(conversation_id)["updated"] = time.time()
        self._save()

    async def create(self, user_id: str | None) -> str:
        """Start a person's new conversation, which becomes their current one."""
        conversation_id = await self.store.create_database()
        self._entry(conversation_id)["user"] = user_id
        self.set_current(user_id, conversation_id)
        logger.info("started text conversation %s", conversation_id)
        return conversation_id

    def record(self, conversation_id: str, title: str) -> None:
        """Note a turn; the first message becomes the conversation's title."""
        entry = self._entry(conversation_id)
        entry["updated"] = time.time()
        if title and not entry.get("title"):
            entry["title"] = title[:80]
        self._save()

    def owner(self, conversation_id: str) -> str | None:
        entry = self._index()["conversations"].get(conversation_id) or {}
        return entry.get("user")

    def owns(self, user_id: str | None, conversation_id: str) -> bool:
        """Whether it is this person's; one missing from the index is no one's."""
        index = self._index()["conversations"]
        return (
            conversation_id in index and index[conversation_id].get("user") == user_id
        )

    def stale(self, conversation_id: str) -> bool:
        """Whether the person has been away long enough to start a new conversation."""
        if settings.text_renew_after <= 0:
            return False
        entry = self._index()["conversations"].get(conversation_id) or {}
        updated = float(entry.get("updated") or 0)
        return time.time() - updated > settings.text_renew_after

    def usage(self, conversation_id: str) -> dict[str, int] | None:
        entry = self._index()["conversations"].get(conversation_id) or {}
        return entry.get("usage")

    def set_usage(self, conversation_id: str, usage: dict[str, int]) -> None:
        self._entry(conversation_id)["usage"] = usage
        self._save()

    def stored(self, conversation_id: str) -> StoredSession:
        """A conversation's stored session, for a session to start with ``persist=``."""
        return self.store.session(
            conversation_id, conversation_id, endpoint=A2A_ENDPOINT
        )

    async def attach(
        self, session: AgentSession, agent: Agent, conversation_id: str
    ) -> None:
        """Join a started session to its conversation: listed as loaded, saved as each
        item arrives, and its LLM tokens added to the conversation's totals."""
        # a loaded conversation keeps only its latest items, so the prompt stays bounded
        await agent.update_chat_ctx(
            agent.chat_ctx.copy().truncate(max_items=settings.text_max_items)
        )
        self.live[conversation_id] = session
        session.on("close", lambda _: self.live.pop(conversation_id, None))

        saves: set[asyncio.Task[None]] = set()

        def on_item(_: object) -> None:
            # each save writes only what changed, so the store is never far behind
            task = asyncio.create_task(session.save())
            saves.add(task)
            task.add_done_callback(saves.discard)

        session.on("conversation_item_added", on_item)

        # a loaded session counts from zero, so add it to what earlier loads used
        earlier = dict(self.usage(conversation_id) or {})

        def on_usage(ev: SessionUsageUpdatedEvent) -> None:
            llms = [u for u in ev.usage.model_usage if isinstance(u, LLMModelUsage)]
            usage = {
                key: int(earlier.get(key, 0)) + sum(getattr(u, field) for u in llms)
                for key, field in _USAGE_FIELDS.items()
            }
            self.set_usage(conversation_id, usage)

        session.on("session_usage_updated", on_usage)

    async def items(self, conversation_id: str) -> list:
        """A conversation's chat items, from its loaded session or else the store."""
        if (session := self.live.get(conversation_id)) is not None:
            return list(session.history.items)
        # nothing writes to an unloaded conversation, so the last read holds until its
        # files change, as loading and closing it does
        cached = self._read.get(conversation_id)
        if cached is not None and cached[0] == self._stamp(conversation_id):
            return cached[1]
        # not loaded, so read what was saved; this stamps the row closed again, and
        # leaves a new conversation an empty row that loads as new
        stored = self.stored(conversation_id)
        items: list = []
        try:
            record = await stored.load()
            items = list(record.history) if record else []
        except StoreError:
            logger.warning("could not read conversation %s", conversation_id)
            return []
        finally:
            with contextlib.suppress(StoreError):
                await stored.release()
        # stamped after the release, which writes to the file too
        self._read[conversation_id] = (self._stamp(conversation_id), items)
        return items

    def _stamp(self, conversation_id: str) -> tuple[int, ...]:
        paths = sorted((self._dir / "sessions").glob(f"{conversation_id}.sqlite*"))
        return tuple(n for p in paths for n in (p.stat().st_mtime_ns, p.stat().st_size))

    async def listing(self, user_id: str | None) -> list[dict[str, Any]]:
        """A person's conversations, latest first, titled by their first message."""
        mine = {
            k: v
            for k, v in self._index()["conversations"].items()
            if v.get("user") == user_id
        }
        for conversation_id, entry in mine.items():
            if "title" in entry:
                continue
            # never spoken in, or recorded before titles were: read the first message
            first = next(
                (
                    i.text_content
                    for i in await self.items(conversation_id)
                    if i.type == "message" and i.role == "user" and i.text_content
                ),
                None,
            )
            # an empty title marks one never spoken in, so it is read only once
            entry["title"] = (first or "")[:80]
            self._save()
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
        return rows

    async def delete(self, conversation_id: str) -> None:
        """Remove a conversation's files and its entry."""
        # the store keeps a connection per database it has read
        if (database := self.store._databases.pop(conversation_id, None)) is not None:
            await database.aclose()
        for path in (self._dir / "sessions").glob(f"{conversation_id}.sqlite*"):
            path.unlink(missing_ok=True)
        self._index()["conversations"].pop(conversation_id, None)
        self._save()
        self.suggested.pop(conversation_id, None)
        self._read.pop(conversation_id, None)
        logger.info("deleted text conversation %s", conversation_id)


conversations = Conversations(Path(settings.text_data_dir))
