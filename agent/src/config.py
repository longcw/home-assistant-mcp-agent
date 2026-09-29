"""Central configuration: env-derived settings plus fixed protocol constants.

Every environment knob lives in the frozen ``settings`` singleton. The constants after
the dataclass are fixed contract values shared with the scheduler and the frontend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# prompt.yaml lives at the agent root, one level up from src/.
_DEFAULT_PROMPT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "prompt.yaml"
)


@dataclass(frozen=True)
class Settings:
    # Models served by LiveKit Inference (auth via LIVEKIT_INFERENCE_* / LIVEKIT_* env).
    stt_model: str
    stt_language: str
    llm_model: str
    tts_model: str
    tts_voice: str
    tts_language: str
    # Explicit-dispatch name; the frontend and scheduler dispatch this worker by name.
    agent_name: str
    # System prompt file (YAML `instructions:`), bind-mounted and re-read per session.
    prompt_file: str
    # Seconds after the mic is gated before STT (billed continuously) is torn down.
    stt_idle_timeout: float
    # Upper bound on a single headless scheduled execution before it's abandoned.
    scheduled_run_timeout: float
    # Scheduler service base URL and optional bearer token (empty = no auth).
    scheduler_url: str
    scheduler_token: str
    # Home timezone: resolves relative times and stamps schedules with an offset.
    agent_tz: str
    # Home Assistant base URL and long-lived token.
    ha_url: str
    ha_token: str
    # Parallel Search MCP endpoint (web_search / web_fetch); empty turns web search off.
    web_search_url: str
    # Parallel API key for higher limits; empty uses the anonymous free tier.
    parallel_api_key: str
    # Mem0 API key for long-term memory across conversations; empty turns memory off.
    mem0_api_key: str
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
    # Upper bound on one text reply, kept under the reverse proxy's 90 s read timeout.
    text_reply_timeout: float


def load_settings() -> Settings:
    return Settings(
        stt_model=os.getenv("STT_MODEL", "assemblyai/universal-3-5-pro"),
        stt_language=os.getenv("STT_LANGUAGE", "multi"),
        llm_model=os.getenv("LLM_MODEL", "google/gemma-4-31b-it"),
        tts_model=os.getenv("TTS_MODEL", "fishaudio/s2.1-pro"),
        tts_voice=os.getenv("TTS_VOICE", "5c353fdb312f4888836a9a5680099ef0"),
        tts_language=os.getenv("TTS_LANGUAGE", ""),
        agent_name=os.getenv("AGENT_NAME", "ha-agent"),
        prompt_file=os.getenv("PROMPT_FILE", _DEFAULT_PROMPT_FILE),
        stt_idle_timeout=float(os.getenv("STT_IDLE_TIMEOUT", "120")),
        scheduled_run_timeout=float(os.getenv("SCHEDULED_RUN_TIMEOUT", "120")),
        scheduler_url=os.getenv("SCHEDULER_URL", "http://scheduler:8080"),
        scheduler_token=os.getenv("SCHEDULER_TOKEN", ""),
        agent_tz=os.getenv("AGENT_TZ") or os.getenv("TZ") or "UTC",
        ha_url=os.getenv("HA_URL", ""),
        ha_token=os.getenv("HA_TOKEN", ""),
        web_search_url=os.getenv("WEB_SEARCH_URL", "https://search.parallel.ai/mcp"),
        parallel_api_key=os.getenv("PARALLEL_API_KEY", ""),
        mem0_api_key=os.getenv("MEM0_API_KEY", ""),
        http_port=int(os.getenv("HTTP_PORT", "8081")),
        text_api_token=os.getenv("TEXT_API_TOKEN", ""),
        text_data_dir=os.getenv("TEXT_DATA_DIR", "/data"),
        text_idle_timeout=float(os.getenv("TEXT_IDLE_TIMEOUT", "300")),
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
# Mem0's hosted MCP server; each person's memories are their own user scope there.
MEM0_MCP_URL = "https://mcp.mem0.ai/mcp"
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
