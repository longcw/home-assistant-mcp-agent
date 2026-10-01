"""A text turn's progress on its person's phone, and the reply buttons on its answer.

The agent posts each phase of a turn: ``start`` (the question), ``step`` (a tool call),
``report`` (a tool's progress report) and ``final`` (the answer, with the quick replies it
offered). ``TEXT_LIVE_MODE`` picks how they show:

- ``notification``: one notification titled with the question, under a fixed tag. The
  question and the answer pop with sound, and each step is appended as a passive update.
- ``activity``: each phase goes to the livekit_voice integration's progress API, which runs
  inside Home Assistant and can see whether the phone's Live Activity is running.

The answer carries a button per quick reply and a free-text Reply button; a tap sends that
text into the person's conversation as their next message.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

import ha
from config import Config
from schemas import ProgressEvent

logger = logging.getLogger("scheduler.progress")

TAG = "ha-text"


@dataclass
class _Shown:
    """What one person's phone shows of their latest turn."""

    phone: str = ""
    question: str = ""
    steps: list[str] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class PhoneProgress:
    def __init__(self, cfg: Config, phone_of: Callable[[str | None], str]) -> None:
        self.cfg = cfg
        # the notify service of a person's progress phone; "" for none
        self._phone_of = phone_of
        self._shown: dict[str | None, _Shown] = {}
        # the latest answer's buttons: action id -> (person, reply text, or None to type one)
        self._actions: dict[str, tuple[str | None, str | None]] = {}

    async def update(self, event: ProgressEvent) -> bool:
        """Show one phase on the person's phone; False when they have none."""
        if not (phone := self._phone_of(event.user)):
            return False
        shown = self._shown.setdefault(event.user, _Shown())
        async with shown.lock:
            shown.phone = phone
            if event.phase == "start":
                await self._start(shown, event.text)
            elif event.phase == "step":
                await self._step(shown, event.tool, event.args)
            elif event.phase == "report" and event.text:
                await self._report(shown, event.text)
            elif event.phase == "final":
                await self._final(shown, event)
        return True

    async def _start(self, shown: _Shown, question: str) -> None:
        shown.question = question if len(question) <= 60 else f"{question[:59]}…"
        shown.steps = []
        if self._activity:
            await self._phase(shown, "start", f"› {shown.question}", "…", "mdi:robot")
        else:
            await self._notify(shown, ["…"], pop=True)

    async def _step(self, shown: _Shown, tool: str, args: Any) -> None:
        values = args.values() if isinstance(args, dict) else [args] if args else []
        parts = ["/".join(map(str, v)) if isinstance(v, list) else str(v) for v in values]
        shown.steps.append(" · ".join([tool, *parts])[:80])
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
        lines = [f"→ {s}" for s in shown.steps]
        if self._activity:
            await self._phase(shown, "progress", lines[-1], f"Step {len(shown.steps)}", icon)
        else:
            await self._notify(shown, lines)

    async def _report(self, shown: _Shown, text: str) -> None:
        if self._activity:
            status = f"Step {len(shown.steps)}"
            await self._phase(shown, "progress", f"… {text}", status, "mdi:progress-clock")
        else:
            await self._notify(shown, [*(f"→ {s}" for s in shown.steps), f"  … {text}"])

    async def _final(self, shown: _Shown, event: ProgressEvent) -> None:
        answer = event.text or "Done."
        user = event.user
        # only the latest answer's buttons stay live
        self._actions = {k: v for k, v in self._actions.items() if v[0] != user}
        nonce = uuid.uuid4().hex[:8]
        replies = list(dict.fromkeys(event.replies))[:4]
        # a tap can control the house, so it asks for Face ID first
        actions: list[dict[str, Any]] = []
        for i, reply in enumerate(replies):
            action = f"HATEXT_{nonce}_{i}"
            self._actions[action] = (user, reply)
            actions.append({"action": action, "title": reply, "authenticationRequired": True})
        self._actions[f"HATEXT_{nonce}_REPLY"] = (user, None)
        actions.append(
            {
                "action": f"HATEXT_{nonce}_REPLY",
                "title": "Reply",
                "behavior": "textInput",
                "textInputButtonTitle": "Send",
                "authenticationRequired": True,
            }
        )
        if self._activity:
            await self._phase(
                shown,
                "final",
                answer,
                "Done" if event.ok else "Failed",
                "mdi:check-circle" if event.ok else "mdi:alert-circle",
                answer=answer,
                question=shown.question,
                actions=actions,
                clear_after=self.cfg.live_clear_after,
            )
        else:
            await self._notify(shown, [answer], pop=True, actions=actions)

    @property
    def _activity(self) -> bool:
        return self.cfg.live_mode == "activity"

    async def _phase(
        self, shown: _Shown, phase: str, message: str, status: str, icon: str, **extra: Any
    ) -> None:
        body: dict[str, Any] = {
            "phase": phase,
            "target": shown.phone,
            "tag": TAG,
            "message": message,
            "status": status,
            "icon": icon,
            **extra,
        }
        if self.cfg.live_url:
            body["url"] = self.cfg.live_url
        await ha.live_activity(self.cfg, body)

    async def _notify(
        self,
        shown: _Shown,
        lines: list[str],
        *,
        pop: bool = False,
        actions: list[dict[str, Any]] | None = None,
    ) -> None:
        data: dict[str, Any] = {"tag": TAG}
        if self.cfg.live_url:
            data["url"] = self.cfg.live_url
        if actions:
            data["actions"] = actions
        if pop:
            # a replacement under one tag updates silently, so clear it to pop again
            await ha.push(self.cfg, shown.phone, "clear_notification", {"tag": TAG})
        else:
            # appended to the question's notification, or, once that is gone, delivered
            # without a banner or a sound
            data["push"] = {"interruption-level": "passive", "sound": "none"}
        await ha.push(self.cfg, shown.phone, "\n".join(lines), data, title=shown.question)

    async def listen(self) -> None:
        """Send a tap on an answer's buttons into its person's conversation."""
        async for event in ha.subscribe(self.cfg, "mobile_app_notification_action"):
            if (entry := self._actions.get(event.get("action", ""))) is None:
                continue
            user, reply = entry
            text = (event.get("reply_text") or "").strip() if reply is None else reply
            if not text:
                continue
            logger.info("reply from a notification: %s", text)
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(
                        self.cfg.chat_url,
                        headers={"Authorization": f"Bearer {self.cfg.chat_token}"},
                        json={"text": text, "user": user},
                    )
                    resp.raise_for_status()
            except Exception:  # noqa: BLE001 - one lost tap leaves the listener running
                logger.exception("failed to send a notification reply")
