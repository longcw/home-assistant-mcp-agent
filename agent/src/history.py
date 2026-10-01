"""A conversation's chat items, shaped like the card's conversation items."""

from __future__ import annotations

from typing import Any

from config import UPDATE_PREFIX
from utils import parse_arguments


def render(
    items: list, pending: tuple[str, float] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """The card's rows for ``items``, and the quick replies offered since the person
    last spoke; ``pending`` is the running turn's message and when it was sent, which
    a session still loading has not recorded."""
    outputs = {i.call_id: i for i in items if i.type == "function_call_output"}
    shown: list[dict[str, Any]] = []
    suggestions: list[str] = []
    for item in items:
        ts = int(item.created_at * 1000)
        text = item.text_content if item.type == "message" else None
        if text and item.role == "user" and text.startswith(UPDATE_PREFIX):
            # an update is not the person: shown as a closed row, like a tool's
            source, _, body = text.removeprefix(UPDATE_PREFIX).strip().partition(": ")
            shown.append(
                {"kind": "action", "id": item.id, "call_id": item.id, "ts": ts}
                | {"name": "update_from", "args": {"name": source}}
                | {"status": "done", "output": body[:1000]}
            )
        elif item.type == "message" and item.role in ("user", "assistant"):
            if text:
                role = "user" if item.role == "user" else "agent"
                shown.append(
                    {"kind": "message", "id": item.id, "role": role, "text": text}
                    | {"ts": ts}
                )
                if role == "user":
                    suggestions = []
        elif item.type == "function_call":
            args = parse_arguments(item.arguments)
            out = outputs.get(item.call_id)
            status = "running" if out is None else "error" if out.is_error else "done"
            action: dict[str, Any] = {
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
    if pending is None:
        return shown, suggestions
    text, sent_at = pending
    sent_ms = int(sent_at * 1000)
    if not text.startswith(UPDATE_PREFIX) and not any(
        i["kind"] == "message"
        and i["role"] == "user"
        and i["text"] == text
        and int(i["ts"]) >= sent_ms - 1000
        for i in shown
    ):
        # an update turn's message is not the person's, and becomes a row above
        shown.append(
            {"kind": "message", "id": f"pending-{sent_ms}", "role": "user"}
            | {"text": text, "ts": sent_ms}
        )
        suggestions = []
    return shown, suggestions
