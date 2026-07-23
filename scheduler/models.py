"""Persistence models: a ``Task`` and its ``Run`` history.

``tasks`` is the source of truth for the list UI and for rehydrating the in-memory APScheduler
on boot. Timestamps are stored as ISO-8601 strings rather than SQLite DATETIME: ``run_at`` is
timezone-aware (the exact instant the user asked for, offset included), and SQLite cannot
round-trip tzinfo through its native DATETIME type. Keeping them as strings makes the stored
instant unambiguous.
"""

from __future__ import annotations

from sqlalchemy import ForeignKey, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    description: Mapped[str] = mapped_column(Text)
    schedule_type: Mapped[str] = mapped_column(String(16))  # "once" | "recurring"
    # Exactly one of run_at / cron is set, per schedule_type.
    run_at: Mapped[str | None] = mapped_column(String(40), nullable=True)  # aware ISO, "once"
    cron: Mapped[str | None] = mapped_column(String(120), nullable=True)  # 5-field, "recurring"
    timezone: Mapped[str] = mapped_column(String(64))
    # {"type": "function_call", "tool": str, "args": dict} | {"type": "instruction", "text": str}
    execution: Mapped[dict] = mapped_column(JSON)
    # "scheduled" (live) | "completed" (a once task fired) | "missed"
    status: Mapped[str] = mapped_column(String(16), default="scheduled")
    enabled: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[str] = mapped_column(String(40))

    runs: Mapped[list["Run"]] = relationship(
        back_populates="task",
        cascade="all, delete-orphan",
        order_by="Run.fired_at",
    )


class Settings(Base):
    """Singleton (``id=1``) app settings shared by the card and the worker."""

    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(primary_key=True)
    # notify.* services to also push notifications to, e.g. ["mobile_app_iphone"].
    notify_targets: Mapped[list] = mapped_column(JSON, default=list)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    task_id: Mapped[str] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), index=True
    )
    fired_at: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(16))  # "pending" | "success" | "error"
    result: Mapped[str | None] = mapped_column(Text, nullable=True)

    task: Mapped["Task"] = relationship(back_populates="runs")
