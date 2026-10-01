"""Small shared helpers: LLM construction, time formatting and job-metadata parsing."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from livekit.agents import inference, llm
from livekit.plugins import openai

from config import settings


def build_llm() -> llm.LLM:
    """The session LLM: LiveKit Inference, or the OpenAI-compatible endpoint if set."""
    if not settings.llm_base_url:
        return inference.LLM(settings.llm_model)
    return openai.LLM(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        extra_body=settings.llm_extra_body,
    )


def to_aware_iso(run_at: str, tz: str) -> str:
    """Normalise a local datetime string to offset-aware ISO in `tz`."""
    dt = datetime.fromisoformat(run_at.strip().replace(" ", "T"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.isoformat()


def current_time_text(tz: str, at: float) -> str:
    """The local-time line injected before the user message sent at ``at``."""
    sent = datetime.fromtimestamp(at, ZoneInfo(tz))
    return (
        f"The user's last message was sent at {sent.isoformat()} ({tz} local time). "
        'Use it to resolve relative times such as "in 1 hour" or "tonight".'
    )


def parse_job_metadata(raw: str | None) -> dict[str, Any]:
    """Parse LiveKit job metadata JSON; return {} on anything unexpected."""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def truncate(text: str, limit: int) -> str:
    """Clip text to `limit` chars, adding an ellipsis when clipped."""
    return text if len(text) <= limit else text[:limit] + "…"


def parse_arguments(raw: str | None) -> Any:
    """A tool call's JSON arguments, or the raw text when they do not parse."""
    try:
        return json.loads(raw or "{}")
    except ValueError:
        return raw
