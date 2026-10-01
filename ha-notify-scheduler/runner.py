"""Carry out a task when it fires: send its notification, or hand its instruction to the agent.

An instruction goes to the agent's text chat endpoint as a message in its owner's
conversation, so the agent resolves devices then, sees a failing tool and can retry, and
the person can follow up.
"""

from __future__ import annotations

import logging

import httpx

import ha
from config import Config

logger = logging.getLogger("scheduler.runner")

# how the chat endpoint answers a turn that did not finish
_UNFINISHED = ("[failed]", "[canceled]", "[ended]", "[error]")


async def run(
    cfg: Config,
    description: str,
    execution: dict,
    user: str | None,
    targets: list[str],
    *,
    shown: bool,
) -> tuple[str, str]:
    """Run a fired task and tell its owner how it went; returns (status, result).

    ``shown`` is whether a phone shows the owner's turns, and so the instruction's answer.
    """
    if notification := execution.get("notification"):
        title = notification.get("title") or description
        if await ha.notify(cfg, notification["message"], title, targets):
            return "success", "Notification sent."
        return "error", "Failed to send the notification."

    status = "success"
    try:
        if not (instruction := execution.get("instruction")):
            raise RuntimeError("the task has no notification or instruction")
        if not cfg.chat_token:
            raise RuntimeError("TEXT_API_TOKEN is not set, so the agent's text chat is off")
        async with httpx.AsyncClient(timeout=cfg.run_timeout) as client:
            resp = await client.post(
                cfg.chat_url,
                headers={"Authorization": f"Bearer {cfg.chat_token}"},
                json={
                    "text": f"[Scheduled task due: {description}] {instruction}",
                    "user": user,
                    "steps": False,
                    "wait": True,
                    # a task firing mid-conversation waits for the person's turn
                    "interrupt": False,
                },
            )
            resp.raise_for_status()
        result = resp.text.strip()
        if result.startswith(_UNFINISHED):
            raise RuntimeError(result)
    except Exception as exc:  # noqa: BLE001 - any failure is the run's result
        logger.exception("scheduled instruction failed: %s", description)
        status, result = "error", str(exc)
        # a turn that ended badly reached the phone, and one that never ran did not
        shown = shown and result.startswith(_UNFINISHED)

    if not shown:
        ok = status == "success"
        title = "Scheduled task done" if ok else "Scheduled task failed"
        body = result if ok else f"Error: {result}"
        await ha.notify(cfg, f"{description}\n\n{body}", title, targets)
    return status, result

