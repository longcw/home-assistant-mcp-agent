"""Home Assistant's REST and WebSocket APIs, as this service uses them."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx
import websockets

from config import Config

logger = logging.getLogger("scheduler.ha")


async def _post(cfg: Config, path: str, payload: dict[str, Any]) -> bool:
    """POST to HA's API (best-effort); True when it succeeded."""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{cfg.ha_url.rstrip('/')}/api/{path}",
                headers={"Authorization": f"Bearer {cfg.ha_token}"},
                json=payload,
            )
            resp.raise_for_status()
        return True
    except Exception:  # noqa: BLE001 - a failed send is reported by the return value
        logger.exception("failed HA call %s", path)
        return False


async def notify(cfg: Config, message: str, title: str, targets: list[str]) -> bool:
    """Send to each channel; True if any send succeeded.

    ``"persistent_notification"`` raises an in-HA notification, and any other target is a
    ``notify.<service>``, e.g. a phone through the Companion app.
    """
    payload = {"message": message.strip(), "title": title}
    ok = False
    for target in targets:
        if target == "persistent_notification":
            path = "services/persistent_notification/create"
        else:
            path = f"services/notify/{target.removeprefix('notify.')}"
        ok = await _post(cfg, path, payload) or ok
    return ok


async def push(
    cfg: Config, service: str, message: str, data: dict[str, Any], title: str = ""
) -> bool:
    """Send to one ``notify.<service>`` with a Companion-app ``data`` payload."""
    payload: dict[str, Any] = {"message": message, "data": data}
    if title:
        payload["title"] = title
    return await _post(cfg, f"services/notify/{service.removeprefix('notify.')}", payload)


async def live_activity(cfg: Config, body: dict[str, Any]) -> bool:
    """Post one turn phase to the livekit_voice integration's progress API."""
    return await _post(cfg, "livekit_voice/progress", body)


async def subscribe(cfg: Config, event_type: str) -> AsyncIterator[dict[str, Any]]:
    """Yield the data of each HA event of ``event_type``, reconnecting when dropped."""
    url = f"{cfg.ha_url.rstrip('/').replace('http', 'ws', 1)}/api/websocket"
    while True:
        try:
            async with websockets.connect(url, ping_interval=30) as ws:
                await ws.recv()  # auth_required
                await ws.send(json.dumps({"type": "auth", "access_token": cfg.ha_token}))
                if json.loads(await ws.recv()).get("type") != "auth_ok":
                    raise RuntimeError("Home Assistant refused the token")
                await ws.send(
                    json.dumps({"id": 1, "type": "subscribe_events", "event_type": event_type})
                )
                async for raw in ws:
                    data = json.loads(raw)
                    if data.get("type") == "event":
                        yield data["event"]["data"]
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the subscription outlives a dropped socket
            logger.warning("lost the %s subscription; retrying", event_type, exc_info=True)
        await asyncio.sleep(10)
