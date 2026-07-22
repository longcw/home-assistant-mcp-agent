"""Unit tests for SchedulerService CRUD, validation, firing, and rehydration.

Dispatch (the only external dependency) is monkeypatched, so these run without LiveKit.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import service as service_module
from config import Config
from db import make_engine, make_session_factory
from models import Task
from schemas import ExecutionSpec, ScheduleSpec, TaskCreate, TaskUpdate
from service import SchedulerService


def make_service(tmp_path) -> SchedulerService:
    cfg = Config(
        livekit_url="wss://example",
        livekit_api_key="k",
        livekit_api_secret="s",
        agent_name="ha-agent",
        db_path=str(tmp_path / "s.db"),
        default_tz="UTC",
        misfire_grace_seconds=3600,
        port=8080,
        auth_token="",
    )
    engine = make_engine(cfg.db_path)
    return SchedulerService(cfg, make_session_factory(engine))


def future_iso(**delta) -> str:
    return (datetime.now(timezone.utc) + timedelta(**delta)).isoformat()


def test_create_once_task(tmp_path):
    svc = make_service(tmp_path)
    out = svc.create_task(
        TaskCreate(
            description="turn off AC",
            schedule=ScheduleSpec(type="once", run_at=future_iso(hours=1), timezone="UTC"),
            execution=ExecutionSpec(
                steps=[{"tool": "HassTurnOff", "args": {"name": "AC"}}]
            ),
        )
    )
    assert out.status == "scheduled"
    assert out.schedule_type == "once"
    assert out.next_run_at is not None  # falls back to run_at
    assert svc.get_task(out.id).description == "turn off AC"
    assert len(svc.list_tasks()) == 1


def test_create_multi_step_task(tmp_path):
    svc = make_service(tmp_path)
    out = svc.create_task(
        TaskCreate(
            description="turn on the fan and set it to 50%",
            schedule=ScheduleSpec(type="once", run_at=future_iso(hours=1), timezone="UTC"),
            execution=ExecutionSpec(
                steps=[
                    {"tool": "HassTurnOn", "args": {"name": "fan"}},
                    {"tool": "HassSetPosition", "args": {"name": "fan", "position": 50}},
                ]
            ),
        )
    )
    stored = svc.get_task(out.id).execution
    assert [s["tool"] for s in stored["steps"]] == ["HassTurnOn", "HassSetPosition"]
    assert stored["instruction"] is None


def test_create_steps_plus_instruction(tmp_path):
    svc = make_service(tmp_path)
    out = svc.create_task(
        TaskCreate(
            description="fetch weather then summarize",
            schedule=ScheduleSpec(type="once", run_at=future_iso(hours=1), timezone="UTC"),
            execution=ExecutionSpec(
                steps=[{"tool": "GetWeather", "args": {}}],
                instruction="tell me tomorrow's weather in one sentence",
            ),
        )
    )
    stored = svc.get_task(out.id).execution
    assert len(stored["steps"]) == 1
    assert stored["instruction"] == "tell me tomorrow's weather in one sentence"


def test_execution_requires_steps_or_instruction():
    with pytest.raises(ValueError):
        ExecutionSpec()


def test_reject_past_time(tmp_path):
    svc = make_service(tmp_path)
    with pytest.raises(ValueError):
        svc.create_task(
            TaskCreate(
                description="past",
                schedule=ScheduleSpec(type="once", run_at=future_iso(hours=-1), timezone="UTC"),
                execution=ExecutionSpec(instruction="do it"),
            )
        )


def test_reject_bad_cron(tmp_path):
    svc = make_service(tmp_path)
    with pytest.raises(ValueError):
        svc.create_task(
            TaskCreate(
                description="bad",
                schedule=ScheduleSpec(type="recurring", cron="not a cron", timezone="UTC"),
                execution=ExecutionSpec(instruction="do it"),
            )
        )


async def test_recurring_task_next_run(tmp_path):
    svc = make_service(tmp_path)
    svc.scheduler.start()  # running loop present (async test) so next_run_time is computed
    try:
        out = svc.create_task(
            TaskCreate(
                description="every morning",
                schedule=ScheduleSpec(type="recurring", cron="0 8 * * *", timezone="UTC"),
                execution=ExecutionSpec(instruction="good morning"),
            )
        )
        assert out.schedule_type == "recurring"
        assert out.next_run_at is not None
    finally:
        svc.scheduler.shutdown(wait=False)


def test_delete_task(tmp_path):
    svc = make_service(tmp_path)
    out = svc.create_task(
        TaskCreate(
            description="x",
            schedule=ScheduleSpec(type="once", run_at=future_iso(hours=1), timezone="UTC"),
            execution=ExecutionSpec(instruction="x"),
        )
    )
    assert svc.delete_task(out.id) is not None
    assert svc.get_task(out.id) is None
    assert svc.list_tasks(active_only=False) == []


def test_update_reschedule(tmp_path):
    svc = make_service(tmp_path)
    out = svc.create_task(
        TaskCreate(
            description="x",
            schedule=ScheduleSpec(type="once", run_at=future_iso(hours=1), timezone="UTC"),
            execution=ExecutionSpec(instruction="x"),
        )
    )
    new_at = future_iso(hours=3)
    updated = svc.update_task(
        out.id,
        TaskUpdate(schedule=ScheduleSpec(type="once", run_at=new_at, timezone="UTC")),
    )
    assert updated.run_at == new_at


async def test_fire_dispatches_and_records(tmp_path, monkeypatch):
    svc = make_service(tmp_path)
    calls = []

    async def fake_dispatch(cfg, *, task_id, description, execution, run_id, room):
        calls.append((task_id, run_id, execution))

    monkeypatch.setattr(service_module.dispatch, "dispatch_scheduled", fake_dispatch)

    out = svc.create_task(
        TaskCreate(
            description="fire me",
            schedule=ScheduleSpec(type="once", run_at=future_iso(hours=1), timezone="UTC"),
            execution=ExecutionSpec(
                steps=[{"tool": "HassTurnOff", "args": {"name": "AC"}}]
            ),
        )
    )
    await svc._fire(out.id)

    assert len(calls) == 1
    task = svc.get_task(out.id)
    assert task.status == "completed"  # one-shot is done after firing
    assert len(task.runs) == 1 and task.runs[0].status == "pending"

    run_id = calls[0][1]
    assert svc.record_run(run_id, "success", "done") is True
    assert svc.get_task(out.id).runs[0].status == "success"


def test_rehydrate_marks_missed(tmp_path):
    svc = make_service(tmp_path)
    # Insert a long-past once task directly, bypassing create's future-time validation.
    with svc._Session() as s:
        s.add(
            Task(
                id="deadbeef",
                description="old",
                schedule_type="once",
                run_at=(datetime.now(timezone.utc) - timedelta(days=2)).isoformat(),
                cron=None,
                timezone="UTC",
                execution={"steps": [], "instruction": "x"},
                status="scheduled",
                enabled=True,
                created_at=service_module._utcnow_iso(),
            )
        )
        s.commit()
    svc._rehydrate()
    assert svc.get_task("deadbeef").status == "missed"
