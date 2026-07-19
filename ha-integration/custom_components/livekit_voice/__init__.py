"""The LiveKit Voice Assistant integration.

This is deliberately thin glue: the voice "brain" is the standalone LiveKit worker
(agent/agent.py), unchanged. This integration only

  1. mints LiveKit access tokens for logged-in Home Assistant users (token.py), and
  2. serves + registers the dashboard card (../card) so it loads in the HA frontend.

The card renders device tiles from live ``hass.states`` and controls them with
``hass.callService`` — see ../README.md for the full architecture.
"""

from __future__ import annotations

import logging
from pathlib import Path

from homeassistant.components.frontend import add_extra_js_url
from homeassistant.components.http import StaticPathConfig
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import CARD_FILENAME, CARD_URL, DATA_CONFIG, DATA_REGISTERED, DOMAIN
from .token import LiveKitTokenView

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up LiveKit Voice from a config entry."""
    store = hass.data.setdefault(DOMAIN, {})
    store[DATA_CONFIG] = entry.data

    # The HTTP view, static path, and frontend module can only be registered once per
    # HA process (they cannot be cleanly torn down), so guard behind a flag that
    # survives entry reloads but not a full restart.
    if not store.get(DATA_REGISTERED):
        card_path = Path(__file__).parent / "frontend" / CARD_FILENAME
        await hass.http.async_register_static_paths(
            [StaticPathConfig(CARD_URL, str(card_path), cache_headers=False)]
        )
        add_extra_js_url(hass, CARD_URL)
        hass.http.register_view(LiveKitTokenView(hass))
        store[DATA_REGISTERED] = True
        _LOGGER.debug("registered token view + card at %s", CARD_URL)

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Keep the token view reading the latest config after an options update."""
    hass.data[DOMAIN][DATA_CONFIG] = entry.data


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the config entry.

    The registered HTTP view / static path / frontend module remain until HA restarts
    (Home Assistant offers no runtime unregister for them); we only drop the config so
    the token endpoint reports "not configured" until set up again.
    """
    hass.data.get(DOMAIN, {}).pop(DATA_CONFIG, None)
    return True
