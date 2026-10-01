"""Central configuration: env-derived settings plus fixed protocol constants.

Every environment knob lives in the frozen ``settings`` singleton. The constants after
the dataclass are fixed contract values shared with the scheduler and the frontend.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# prompt.yaml and mcp.yaml live at the agent root, one level up from src/.
_AGENT_ROOT = os.path.dirname(os.path.dirname(__file__))


@dataclass(frozen=True)
class Settings:
    # Models served by LiveKit Inference (auth via LIVEKIT_INFERENCE_* / LIVEKIT_* env).
    stt_model: str
    stt_language: str
    llm_model: str
    # OpenAI-compatible endpoint serving llm_model instead of Inference; empty = off.
    llm_base_url: str
    llm_api_key: str
    # extra fields merged into each request to that endpoint, from JSON
    llm_extra_body: dict[str, Any]
    # LLM calls one turn may chain through tools before it must answer.
    max_tool_steps: int
    tts_model: str
    tts_voice: str
    tts_language: str
    # Explicit-dispatch name; the frontend and scheduler dispatch this worker by name.
    agent_name: str
    # System prompt file (YAML `instructions:`), bind-mounted and re-read per session.
    prompt_file: str
    # Seconds after the mic is gated before STT (billed continuously) is torn down.
    stt_idle_timeout: float
    # Scheduler service base URL and optional bearer token (empty = no auth).
    scheduler_url: str
    scheduler_token: str
    # Home timezone: resolves relative times and stamps schedules with an offset.
    agent_tz: str
    # Home Assistant base URL and long-lived token.
    ha_url: str
    ha_token: str
    # MCP servers beyond Home Assistant's (YAML `servers:`); a missing file adds none.
    mcp_config: str
    # this agent's HTTP app as MCP servers reach it, for their callbacks; empty = none.
    callback_base_url: str
    # Port of the agent server's HTTP app, which serves the text chat endpoints.
    http_port: int
    # Bearer token for the text chat endpoints; empty leaves them unmounted.
    text_api_token: str
    # Directory holding the persisted text conversations and the current one's id.
    text_data_dir: str
    # Seconds a quiet text conversation stays loaded before it is saved and dropped.
    text_idle_timeout: float
    # Seconds idle after which the next text turn starts a new conversation; 0 = never.
    text_renew_after: float
    # Chat items a loaded text conversation keeps; older ones are dropped.
    text_max_items: int
    # notify service of the phone that shows each text turn's progress; empty = off.
    text_live_activity: str
    # "notification" (one tagged notification, replaced in place) or "activity".
    text_live_mode: str
    # Page the phone opens in the HA app when the progress is tapped, e.g. the Text tab.
    text_live_url: str
    # Seconds after the last text turn before a Live Activity is cleared.
    text_live_clear_after: float
    # Seconds a waiting /chat request waits for its reply, under the proxy's 90 s.
    text_reply_timeout: float


def load_settings() -> Settings:
    return Settings(
        stt_model=os.getenv("STT_MODEL", "assemblyai/universal-3-5-pro"),
        stt_language=os.getenv("STT_LANGUAGE", "multi"),
        llm_model=os.getenv("LLM_MODEL", "google/gemma-4-31b-it"),
        llm_base_url=os.getenv("LLM_BASE_URL", ""),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        llm_extra_body=json.loads(os.getenv("LLM_EXTRA_BODY") or "{}"),
        max_tool_steps=int(os.getenv("MAX_TOOL_STEPS", "6")),
        tts_model=os.getenv("TTS_MODEL", "fishaudio/s2.1-pro"),
        tts_voice=os.getenv("TTS_VOICE", "5c353fdb312f4888836a9a5680099ef0"),
        tts_language=os.getenv("TTS_LANGUAGE", ""),
        agent_name=os.getenv("AGENT_NAME", "ha-agent"),
        prompt_file=os.getenv("PROMPT_FILE", os.path.join(_AGENT_ROOT, "prompt.yaml")),
        stt_idle_timeout=float(os.getenv("STT_IDLE_TIMEOUT", "120")),
        scheduler_url=os.getenv("SCHEDULER_URL", "http://ha-notify-scheduler:8080"),
        scheduler_token=os.getenv("SCHEDULER_TOKEN", ""),
        agent_tz=os.getenv("AGENT_TZ") or os.getenv("TZ") or "UTC",
        ha_url=os.getenv("HA_URL", ""),
        ha_token=os.getenv("HA_TOKEN", ""),
        mcp_config=os.getenv("MCP_CONFIG", os.path.join(_AGENT_ROOT, "mcp.yaml")),
        callback_base_url=os.getenv("CALLBACK_BASE_URL", ""),
        http_port=int(os.getenv("HTTP_PORT", "8081")),
        text_api_token=os.getenv("TEXT_API_TOKEN", ""),
        text_data_dir=os.getenv("TEXT_DATA_DIR", "/data"),
        text_idle_timeout=float(os.getenv("TEXT_IDLE_TIMEOUT", "1800")),
        text_renew_after=float(os.getenv("TEXT_RENEW_AFTER", "0")),
        text_max_items=int(os.getenv("TEXT_MAX_ITEMS", "100")),
        text_live_activity=os.getenv("TEXT_LIVE_ACTIVITY", ""),
        text_live_mode=os.getenv("TEXT_LIVE_MODE", "notification"),
        text_live_url=os.getenv("TEXT_LIVE_URL", ""),
        text_live_clear_after=float(os.getenv("TEXT_LIVE_CLEAR_AFTER", "900")),
        text_reply_timeout=float(os.getenv("TEXT_REPLY_TIMEOUT", "80")),
    )


settings = load_settings()


# --- Fixed protocol constants (not env-configurable) ---
# HA's MCP Server integration exposes Streamable HTTP at /api/mcp.
MCP_PATH = "/api/mcp"
# How a webhook's update opens the user message that carries it into a conversation.
UPDATE_PREFIX = "[update]"
# HA tool returning the live state of all exposed entities.
LIVE_CONTEXT_TOOL = "GetLiveContext"
# Data-channel topic carrying the tool-execution lifecycle: powers the frontend's tool
# cards and (since state tools return YAML) the device/sensor status cards.
TOOL_CALL_TOPIC = "ha.tool_call"
# Data-channel topic mirroring session state (stt_enabled / audio_output booleans).
SESSION_STATE_TOPIC = "ha.speech_state"
# Data-channel topic carrying one-tap quick replies for the card ({"replies": [...]}).
SUGGESTIONS_TOPIC = "ha.suggestions"
# Keep forwarded tool outputs under LiveKit's data-packet size budget, while leaving a
# normal home-state YAML payload intact and parseable for the UI cards.
MAX_TOOL_OUTPUT_CHARS = 12000
