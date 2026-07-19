"""LiveKit access-token endpoint for the dashboard card.

The card POSTs to ``/api/livekit_voice/token`` (via ``hass.callApi``, which attaches the
user's HA auth), and this view mints a short-lived LiveKit token that also explicitly
dispatches the voice worker (agent/agent.py) into a fresh room. Mirrors the token route
in frontend/app/api/token/route.ts, but auth is enforced by Home Assistant.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import timedelta

from aiohttp import web
from livekit import api

from homeassistant.components.http import HomeAssistantView
from homeassistant.core import HomeAssistant

from .const import (
    CONF_AGENT_NAME,
    CONF_API_KEY,
    CONF_API_SECRET,
    CONF_LIVEKIT_URL,
    DATA_CONFIG,
    DEFAULT_AGENT_NAME,
    DOMAIN,
    TOKEN_URL,
)

_LOGGER = logging.getLogger(__name__)

# Turn modes the card may request; forwarded to the worker as dispatch metadata so it
# configures turn detection to match (see agent/agent.py `_push_to_talk_requested`).
_VALID_INPUT_MODES = ("auto", "push_to_talk")
_TOKEN_TTL = timedelta(minutes=15)


class LiveKitTokenView(HomeAssistantView):
    """Mints a LiveKit token + agent dispatch for the authenticated HA user."""

    url = TOKEN_URL
    name = "api:livekit_voice:token"
    requires_auth = True

    def __init__(self, hass: HomeAssistant) -> None:
        self._hass = hass

    async def post(self, request: web.Request) -> web.Response:
        config = self._hass.data.get(DOMAIN, {}).get(DATA_CONFIG)
        if not config:
            return self.json_message(
                "LiveKit Voice is not configured", status_code=503
            )

        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError):
            body = {}

        input_mode = "auto"
        if isinstance(body, dict) and body.get("input_mode") in _VALID_INPUT_MODES:
            input_mode = body["input_mode"]

        agent_name = config.get(CONF_AGENT_NAME) or DEFAULT_AGENT_NAME
        identity = f"ha-user-{secrets.token_hex(4)}"
        room_name = f"ha-voice-{secrets.token_hex(4)}"
        metadata = json.dumps({"input_mode": input_mode})

        try:
            token = (
                api.AccessToken(config[CONF_API_KEY], config[CONF_API_SECRET])
                .with_identity(identity)
                .with_name("Home Assistant")
                .with_ttl(_TOKEN_TTL)
                .with_grants(
                    api.VideoGrants(
                        room_join=True,
                        room=room_name,
                        can_publish=True,
                        can_publish_data=True,
                        can_subscribe=True,
                    )
                )
                .with_room_config(
                    api.RoomConfiguration(
                        agents=[
                            api.RoomAgentDispatch(
                                agent_name=agent_name, metadata=metadata
                            )
                        ]
                    )
                )
                .to_jwt()
            )
        except Exception:  # noqa: BLE001 - surface any signing/config error as 500
            _LOGGER.exception("failed to mint LiveKit token")
            return self.json_message("failed to mint token", status_code=500)

        return self.json(
            {
                "serverUrl": config[CONF_LIVEKIT_URL],
                "roomName": room_name,
                "participantName": "Home Assistant",
                "participantToken": token,
            }
        )
