"""Small shared helpers: time formatting and job-metadata parsing."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo


def to_aware_iso(run_at: str, tz: str) -> str:
    """Normalise a local datetime string to offset-aware ISO in `tz`."""
    dt = datetime.fromisoformat(run_at.strip().replace(" ", "T"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.isoformat()


def current_time_text(tz: str) -> str:
    """The 'current local time' line injected fresh into each LLM turn."""
    now = datetime.now(ZoneInfo(tz))
    return (
        f"The current local time is {now.isoformat()} ({tz}). "
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
