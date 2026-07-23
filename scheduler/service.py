"""Scheduling engine: owns the APScheduler instance and the task/run persistence.

Design: the ``tasks`` table is the single source of truth. APScheduler runs purely in memory
and is rehydrated from the table on boot, so there is only one durable store to reason about
(no risk of the job store and the task table drifting apart). One-shot tasks use a
``DateTrigger``; recurring tasks a cron ``CronTrigger``. When a trigger fires we dispatch the
worker (dispatch.py) and record a ``Run``; the worker later reports the outcome back.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

import dispatch
from config import Config
from models import Run, Settings, Task
from schemas import RunOut, SettingsOut, SettingsUpdate, TaskCreate, TaskOut, TaskUpdate

logger = logging.getLogger("scheduler.service")

ACTIVE = "scheduled"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


class SchedulerService:
    def __init__(self, cfg: Config, session_factory: sessionmaker) -> None:
        self.cfg = cfg
        self._Session = session_factory
        default_tz = ZoneInfo(cfg.default_tz) if cfg.default_tz else timezone.utc
        self.scheduler = AsyncIOScheduler(timezone=default_tz)

    # --- lifecycle -----------------------------------------------------------------

    def start(self) -> None:
        self.scheduler.start()
        self._rehydrate()

    async def stop(self) -> None:
        self.scheduler.shutdown(wait=False)

    # --- public CRUD ---------------------------------------------------------------

    def create_task(self, req: TaskCreate) -> TaskOut:
        run_at_iso, cron = self._validate_schedule(req.schedule)
        with self._Session() as s:
            task = Task(
                id=_new_id(),
                description=req.description,
                schedule_type=req.schedule.type,
                run_at=run_at_iso,
                cron=cron,
                timezone=req.schedule.timezone,
                execution=req.execution.model_dump(),
                status=ACTIVE,
                enabled=True,
                created_at=_utcnow_iso(),
            )
            s.add(task)
            s.commit()
            s.refresh(task)
            self._add_job(task)
            return self._to_out(task)

    def list_tasks(self, active_only: bool = True) -> list[TaskOut]:
        with self._Session() as s:
            stmt = select(Task)
            if active_only:
                stmt = stmt.where(Task.status == ACTIVE)
            tasks = list(s.scalars(stmt))
            out = [self._to_out(t) for t in tasks]
        # Soonest fire first; tasks with no upcoming run sort last.
        out.sort(key=lambda t: (t.next_run_at is None, t.next_run_at or ""))
        return out

    def get_task(self, task_id: str) -> TaskOut | None:
        with self._Session() as s:
            task = s.get(Task, task_id)
            return self._to_out(task) if task else None

    def update_task(self, task_id: str, req: TaskUpdate) -> TaskOut | None:
        with self._Session() as s:
            task = s.get(Task, task_id)
            if task is None:
                return None
            if req.description is not None:
                task.description = req.description
            if req.execution is not None:
                task.execution = req.execution.model_dump()
            if req.schedule is not None:
                run_at_iso, cron = self._validate_schedule(req.schedule)
                task.schedule_type = req.schedule.type
                task.run_at = run_at_iso
                task.cron = cron
                task.timezone = req.schedule.timezone
                # Re-scheduling revives a task that already fired / was cancelled.
                if task.status != ACTIVE:
                    task.status = ACTIVE
            if req.enabled is not None:
                task.enabled = req.enabled
            s.commit()
            s.refresh(task)
            self._remove_job(task.id)
            if task.enabled and task.status == ACTIVE:
                self._add_job(task)
            return self._to_out(task)

    def delete_task(self, task_id: str) -> TaskOut | None:
        """Hard-delete a task and its run history, and unschedule it."""
        with self._Session() as s:
            task = s.get(Task, task_id)
            if task is None:
                return None
            out = self._to_out(task)  # snapshot before removal
            s.delete(task)  # runs cascade via the relationship
            s.commit()
        self._remove_job(task_id)
        return out

    def record_run(self, run_id: str, status: str, result: str | None) -> bool:
        """Called from the worker's report endpoint once a task has executed."""
        with self._Session() as s:
            run = s.get(Run, run_id)
            if run is None:
                return False
            run.status = status
            run.result = result
            s.commit()
            return True

    # --- settings ------------------------------------------------------------------

    def get_settings(self) -> SettingsOut:
        with self._Session() as s:
            row = s.get(Settings, 1)
            # No row yet → default to the in-HA persistent notification being enabled.
            targets = list(row.notify_targets) if row else ["persistent_notification"]
        return SettingsOut(notify_targets=targets)

    def update_settings(self, req: SettingsUpdate) -> SettingsOut:
        with self._Session() as s:
            row = s.get(Settings, 1)
            if row is None:
                row = Settings(id=1, notify_targets=[])
                s.add(row)
            if req.notify_targets is not None:
                row.notify_targets = req.notify_targets
            s.commit()
            s.refresh(row)
            return SettingsOut(notify_targets=list(row.notify_targets))

    # --- firing --------------------------------------------------------------------

    async def _fire(self, task_id: str) -> None:
        """APScheduler callback. Records a pending run then dispatches the worker."""
        with self._Session() as s:
            task = s.get(Task, task_id)
            if task is None or not task.enabled or task.status == "cancelled":
                logger.info("skip firing task %s (missing/disabled/cancelled)", task_id)
                return
            run_id = _new_id()
            s.add(Run(id=run_id, task_id=task_id, fired_at=_utcnow_iso(), status="pending"))
            # A one-shot won't fire again, so it's done as far as scheduling goes; the run row
            # tracks its actual outcome.
            if task.schedule_type == "once":
                task.status = "completed"
            description = task.description
            execution = dict(task.execution)
            s.commit()

        room = f"sched-{task_id[:8]}-{run_id[:8]}"
        try:
            await dispatch.dispatch_scheduled(
                self.cfg,
                task_id=task_id,
                description=description,
                execution=execution,
                run_id=run_id,
                room=room,
            )
        except Exception as exc:  # noqa: BLE001 - dispatch failure is a run failure
            logger.exception("failed to dispatch task %s", task_id)
            self.record_run(run_id, "error", f"dispatch failed: {exc}")

    # --- helpers -------------------------------------------------------------------

    def _validate_schedule(self, schedule) -> tuple[str | None, str | None]:
        """Validate a schedule and return the (run_at_iso, cron) to persist.

        Raises ValueError on a bad timezone, a past one-shot time, or an invalid cron.
        """
        try:
            tz = ZoneInfo(schedule.timezone)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid timezone {schedule.timezone!r}: {exc}") from exc

        if schedule.type == "once":
            try:
                dt = datetime.fromisoformat(schedule.run_at)
            except ValueError as exc:
                raise ValueError(f"invalid run_at {schedule.run_at!r}") from exc
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)
            if dt.astimezone(timezone.utc) <= _utcnow():
                raise ValueError("run_at is in the past")
            return dt.isoformat(), None

        # recurring: validate the cron expression by building the trigger.
        try:
            CronTrigger.from_crontab(schedule.cron, timezone=tz)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"invalid cron {schedule.cron!r}: {exc}") from exc
        return None, schedule.cron

    def _trigger(self, task: Task):
        tz = ZoneInfo(task.timezone)
        if task.schedule_type == "recurring":
            return CronTrigger.from_crontab(task.cron, timezone=tz)
        if task.run_at is None:
            raise ValueError(f"once task {task.id} has no run_at")
        dt = datetime.fromisoformat(task.run_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        return DateTrigger(run_date=dt)

    def _add_job(self, task: Task) -> None:
        self.scheduler.add_job(
            self._fire,
            trigger=self._trigger(task),
            args=[task.id],
            id=task.id,
            replace_existing=True,
            misfire_grace_time=self.cfg.misfire_grace_seconds,
            coalesce=True,
        )

    def _remove_job(self, task_id: str) -> None:
        if self.scheduler.get_job(task_id) is not None:
            self.scheduler.remove_job(task_id)

    def _next_run_at(self, task: Task) -> str | None:
        job = self.scheduler.get_job(task.id)
        # next_run_time is a __slots__ field left unassigned until a *running* scheduler
        # processes the job, so read it defensively (getattr, not attribute access).
        next_run_time = getattr(job, "next_run_time", None) if job is not None else None
        if next_run_time is not None:
            return next_run_time.isoformat()
        if task.schedule_type == "once" and task.status == ACTIVE:
            return task.run_at
        return None

    def _to_out(self, task: Task) -> TaskOut:
        return TaskOut(
            id=task.id,
            description=task.description,
            schedule_type=task.schedule_type,
            run_at=task.run_at,
            cron=task.cron,
            timezone=task.timezone,
            execution=task.execution,
            status=task.status,
            enabled=task.enabled,
            created_at=task.created_at,
            next_run_at=self._next_run_at(task),
            # Most-recent runs only: bounds the payload so a long-lived recurring task's
            # history can't blow past the worker's tool-feed size budget (runs are asc).
            runs=[
                RunOut(id=r.id, fired_at=r.fired_at, status=r.status, result=r.result)
                for r in task.runs[-10:]
            ],
        )

    def _rehydrate(self) -> None:
        """Re-arm live tasks from the DB after a restart.

        Recurring tasks are simply re-added. A one-shot whose time is still in the future is
        re-armed; one whose time passed while we were down runs once now if within the misfire
        grace window, otherwise it's marked "missed" so the user can see it didn't run.
        """
        now = _utcnow()
        with self._Session() as s:
            tasks = list(
                s.scalars(select(Task).where(Task.status == ACTIVE, Task.enabled.is_(True)))
            )
            for task in tasks:
                if task.schedule_type == "recurring":
                    self._add_job(task)
                    continue
                dt = datetime.fromisoformat(task.run_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo(task.timezone))
                dt_utc = dt.astimezone(timezone.utc)
                if dt_utc > now:
                    self._add_job(task)
                elif (now - dt_utc).total_seconds() <= self.cfg.misfire_grace_seconds:
                    self.scheduler.add_job(
                        self._fire,
                        trigger=DateTrigger(run_date=now),
                        args=[task.id],
                        id=task.id,
                        replace_existing=True,
                    )
                else:
                    task.status = "missed"
                    s.add(
                        Run(
                            id=_new_id(),
                            task_id=task.id,
                            fired_at=_utcnow_iso(),
                            status="error",
                            result="missed while the scheduler was offline",
                        )
                    )
            s.commit()
        logger.info("rehydrated %d active task(s)", len(tasks))
