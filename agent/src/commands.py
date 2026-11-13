"""Slash commands: a text message `/name args` runs a command instead of a turn.

A command is a markdown file in ``COMMANDS_DIR`` shaped like a Claude Code skill: YAML
frontmatter with ``name``, ``description`` and ``argument-hint``, and the body as its
prompt, so a skill file works unchanged. A command runs on its own, with no tools and
none of the conversation but its own earlier exchanges; what was typed and the reply
are then added to the conversation, so the person and the agent both see them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from livekit.agents import llm

from config import settings
from utils import build_llm

logger = logging.getLogger("ha-mcp-agent.commands")

# the `extra` key marking a conversation's items as a command's, with its name
COMMAND_KEY = "command"
# earlier exchanges of the same command a run sees, so `/en 3` alone can redo the last
_EARLIER = 3
_INVOCATION = re.compile(r"/([\w-]+)(?:\s+(.*))?", re.DOTALL)


@dataclass(frozen=True)
class Command:
    name: str
    description: str
    argument_hint: str
    prompt: str


def load() -> dict[str, Command]:
    """The commands in ``COMMANDS_DIR``, read fresh so an edit applies at once."""
    found: dict[str, Command] = {}
    for path in sorted(Path(settings.commands_dir).glob("*.md")):
        if path.name.startswith("."):
            continue  # e.g. the ._ files a Mac leaves on a network volume
        text = path.read_text(encoding="utf-8")
        head, sep, body = text.removeprefix("---\n").partition("\n---\n")
        if not text.startswith("---\n") or not sep:
            continue
        try:
            meta = yaml.safe_load(head) or {}
        except yaml.YAMLError:
            logger.warning("skipping command %s: bad frontmatter", path.name)
            continue
        name = str(meta.get("name") or "")
        if not name or not body.strip():
            continue
        found[name] = Command(
            name=name,
            description=str(meta.get("description") or ""),
            argument_hint=str(meta.get("argument-hint") or ""),
            prompt=body.strip(),
        )
    return found


def parse(text: str) -> Command | None:
    """The command a message invokes; an unknown name is an ordinary message."""
    match = _INVOCATION.fullmatch(text.strip())
    return load().get(match.group(1)) if match else None


async def run(command: Command, typed: str, earlier: list[tuple[str, str]]) -> str:
    """Answer ``typed`` with the command's prompt, after its ``earlier`` exchanges."""
    chat_ctx = llm.ChatContext.empty()
    chat_ctx.add_message(role="system", content=command.prompt)
    for asked, answered in earlier[-_EARLIER:]:
        chat_ctx.add_message(role="user", content=asked)
        chat_ctx.add_message(role="assistant", content=answered)
    chat_ctx.add_message(role="user", content=typed)
    parts: list[str] = []
    model = build_llm(settings.commands_llm_extra_body)
    async with model.chat(chat_ctx=chat_ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                parts.append(chunk.delta.content)
    return "".join(parts).strip()


def exchanges(items: list, name: str) -> list[tuple[str, str]]:
    """A conversation's earlier exchanges of one command, oldest first."""
    pairs: list[tuple[str, str]] = []
    asked: str | None = None
    for item in items:
        if item.type != "message" or item.extra.get(COMMAND_KEY) != name:
            continue
        if item.role == "user":
            asked = item.text_content
        elif asked is not None:
            pairs.append((asked, item.text_content or ""))
            asked = None
    return pairs
