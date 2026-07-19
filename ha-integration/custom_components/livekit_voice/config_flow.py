"""Config flow for the LiveKit Voice Assistant integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant.config_entries import ConfigFlow, ConfigFlowResult

from .const import (
    CONF_AGENT_NAME,
    CONF_API_KEY,
    CONF_API_SECRET,
    CONF_LIVEKIT_URL,
    DEFAULT_AGENT_NAME,
    DOMAIN,
)


class LiveKitVoiceConfigFlow(ConfigFlow, domain=DOMAIN):
    """Collect the LiveKit connection details for the token endpoint."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial setup step."""
        if user_input is not None:
            return self.async_create_entry(
                title="LiveKit Voice Assistant", data=user_input
            )

        schema = vol.Schema(
            {
                vol.Required(CONF_LIVEKIT_URL): str,
                vol.Required(CONF_API_KEY): str,
                vol.Required(CONF_API_SECRET): str,
                vol.Optional(CONF_AGENT_NAME, default=DEFAULT_AGENT_NAME): str,
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema)
