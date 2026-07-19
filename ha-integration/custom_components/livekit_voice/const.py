"""Constants for the LiveKit Voice Assistant integration."""

DOMAIN = "livekit_voice"

# Config-entry keys.
CONF_LIVEKIT_URL = "livekit_url"
CONF_API_KEY = "api_key"
CONF_API_SECRET = "api_secret"
CONF_AGENT_NAME = "agent_name"

# Matches AGENT_NAME in agent/agent.py (the worker's explicit-dispatch name).
DEFAULT_AGENT_NAME = "ha-agent"

# Served card bundle (built from ../card via esbuild) and its public URL.
CARD_FILENAME = "livekit-voice-card.js"
CARD_URL = "/livekit_voice/livekit-voice-card.js"

# Token endpoint the card calls via `hass.callApi('POST', 'livekit_voice/token', ...)`.
TOKEN_URL = "/api/livekit_voice/token"

# hass.data[DOMAIN] keys.
DATA_CONFIG = "config"
DATA_REGISTERED = "registered"
