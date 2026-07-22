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
    )


settings = load_settings()


# --- Fixed protocol constants (not env-configurable) ---
# HA's MCP Server integration exposes Streamable HTTP at /api/mcp.
MCP_PATH = "/api/mcp"
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
