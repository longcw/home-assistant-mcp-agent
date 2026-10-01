"""Unit tests for SchedulerService CRUD, validation, firing, and rehydration.

The runner's HTTP calls go to a mock transport, so these run without Home Assistant or
the agent.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

import ha
import service as service_module
from config import Config
from db import make_engine, make_session_factory
from models import Task
from schemas import (
    ExecutionSpec,
    ProgressEvent,
    ScheduleSpec,
    SettingsUpdate,
    TaskCreate,
    TaskUpdate,
    UserSettings,
)
from service import SchedulerService


def make_service(tmp_path) -> SchedulerService:
    cfg = Config(
        ha_url="http://ha",
        ha_token="ha-token",
        chat_url="http://agent/chat",
        chat_token="chat-token",
        run_timeout=5,
        default_phone="",
        live_mode="notification",
        live_url="",
        live_clear_after=900,
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
            schedule=ScheduleSpec(
                type="once", run_at=future_iso(hours=1), timezone="UTC"
            ),
            execution=ExecutionSpec(instruction="turn off the AC"),
        )
    )
    assert out.status == "scheduled"
    assert out.schedule_type == "once"
    assert out.next_run_at is not None  # falls back to run_at
    assert svc.get_task(out.id).description == "turn off AC"
    assert len(svc.list_tasks()) == 1


def test_create_reminder(tmp_path):
    svc = make_service(tmp_path)
    out = svc.create_task(
        TaskCreate(
            description="buy soap",
            schedule=ScheduleSpec(
                type="once", run_at=future_iso(hours=1), timezone="UTC"
            ),
            execution=ExecutionSpec(notification={"message": "time to buy soap"}),
        )
    )
    assert svc.get_task(out.id).execution == {
        "notification": {"message": "time to buy soap"}
    }


def test_execution_needs_exactly_one_kind():
    with pytest.raises(ValueError):
        ExecutionSpec()
    with pytest.raises(ValueError):
        ExecutionSpec(instruction="  ")
    with pytest.raises(ValueError):
        ExecutionSpec(notification={"message": "x"}, instruction="do it")


def test_reject_past_time(tmp_path):
    svc = make_service(tmp_path)
    with pytest.raises(ValueError):
        svc.create_task(
            TaskCreate(
                description="past",
                schedule=ScheduleSpec(
                    type="once", run_at=future_iso(hours=-1), timezone="UTC"
                ),
                execution=ExecutionSpec(instruction="do it"),
            )
        )


def test_reject_bad_cron(tmp_path):
    svc = make_service(tmp_path)
    with pytest.raises(ValueError):
        svc.create_task(
            TaskCreate(
                description="bad",
                schedule=ScheduleSpec(
                    type="recurring", cron="not a cron", timezone="UTC"
                ),
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
                schedule=ScheduleSpec(
                    type="recurring", cron="0 8 * * *", timezone="UTC"
                ),
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
            schedule=ScheduleSpec(
                type="once", run_at=future_iso(hours=1), timezone="UTC"
            ),
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
            schedule=ScheduleSpec(
                type="once", run_at=future_iso(hours=1), timezone="UTC"
            ),
            execution=ExecutionSpec(instruction="x"),
        )
    )
    new_at = future_iso(hours=3)
    updated = svc.update_task(
        out.id,
        TaskUpdate(schedule=ScheduleSpec(type="once", run_at=new_at, timezone="UTC")),
    )
    assert updated.run_at == new_at


def mock_http(monkeypatch, handler) -> list[httpx.Request]:
    """Route the service's HTTP calls to ``handler``; returns the requests it got."""
    requests: list[httpx.Request] = []
    real = httpx.AsyncClient

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    def client(**kwargs) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(record), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return requests


def fire_once(svc: SchedulerService, execution: ExecutionSpec, user: str | None) -> str:
    return svc.create_task(
        TaskCreate(
            description="fire me",
            schedule=ScheduleSpec(
                type="once", run_at=future_iso(hours=1), timezone="UTC"
            ),
            execution=execution,
            user=user,
        )
    ).id


async def test_fire_sends_a_reminder(tmp_path, monkeypatch):
    svc = make_service(tmp_path)
    svc.update_settings(
        SettingsUpdate(users=[UserSettings(name="Alice", notify_targets=["mobile_app_a"])])
    )
    requests = mock_http(monkeypatch, lambda _: httpx.Response(200, json=[]))
    task_id = fire_once(svc, ExecutionSpec(notification={"message": "soap"}), "alice")
    await svc._fire(task_id)

    assert [r.url.path for r in requests] == ["/api/services/notify/mobile_app_a"]
    assert requests[0].headers["authorization"] == "Bearer ha-token"
    task = svc.get_task(task_id, owner="alice")
    assert task.status == "completed"  # one-shot is done after firing
    assert [(r.status, r.result) for r in task.runs] == [("success", "Notification sent.")]


async def test_fire_sends_an_instruction_to_the_agent(tmp_path, monkeypatch):
    svc = make_service(tmp_path)
    svc.update_settings(
        SettingsUpdate(users=[UserSettings(name="Alice", notify_targets=["mobile_app_a"])])
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/chat":
            return httpx.Response(200, text="AC is off.")
        return httpx.Response(200, json=[])

    requests = mock_http(monkeypatch, handler)
    task_id = fire_once(svc, ExecutionSpec(instruction="turn off the AC"), "alice")
    await svc._fire(task_id)

    # alice's phone shows her turns, so nothing else is sent
    assert [r.url.path for r in requests] == ["/chat"]
    sent = requests[0]
    assert sent.headers["authorization"] == "Bearer chat-token"
    body = json.loads(sent.content)
    assert body["text"] == "[Scheduled task due: fire me] turn off the AC"
    assert body["user"] == "alice" and body["wait"] and not body["interrupt"]
    runs = svc.get_task(task_id, owner="alice").runs
    assert [(r.status, r.result) for r in runs] == [("success", "AC is off.")]


async def test_fire_reports_a_failed_turn(tmp_path, monkeypatch):
    svc = make_service(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/chat":
            return httpx.Response(200, text="[failed] no such device")
        return httpx.Response(200, json=[])

    requests = mock_http(monkeypatch, handler)
    task_id = fire_once(svc, ExecutionSpec(instruction="turn off the AC"), None)
    await svc._fire(task_id)

    # no phone showed it, so HA's own notification tells of the failure
    assert [r.url.path for r in requests] == [
        "/chat",
        "/api/services/persistent_notification/create",
    ]
    assert json.loads(requests[1].content)["title"] == "Scheduled task failed"
    run = svc.get_task(task_id).runs[0]
    assert (run.status, run.result) == ("error", "[failed] no such device")


async def test_progress_on_the_phone_and_a_tapped_reply(tmp_path, monkeypatch):
    svc = make_service(tmp_path)
    svc.update_settings(
        SettingsUpdate(users=[UserSettings(name="Alice", notify_targets=["mobile_app_a"])])
    )
    requests = mock_http(monkeypatch, lambda _: httpx.Response(200, json=[]))
    for event in (
        ProgressEvent(user="alice", phase="start", text="AC off?"),
        ProgressEvent(user="alice", phase="step", tool="HassTurnOff", args={"name": "AC"}),
        ProgressEvent(user="alice", phase="final", text="Off.", replies=["Thanks"]),
    ):
        assert await svc.progress.update(event)
    sent = [json.loads(r.content) for r in requests]
    assert [s["message"] for s in sent] == [
        "clear_notification",
        "…",
        "→ HassTurnOff · AC",
        "clear_notification",
        "Off.",
    ]
    assert sent[2]["title"] == "AC off?"
    assert sent[2]["data"]["push"]["interruption-level"] == "passive"
    buttons = sent[4]["data"]["actions"]
    assert [b["title"] for b in buttons] == ["Thanks", "Reply"]
    # no one in particular has no phone here, so nothing shows
    assert not await svc.progress.update(ProgressEvent(phase="start", text="hi"))

    async def taps():
        yield {"action": buttons[0]["action"]}
        yield {"action": buttons[1]["action"], "reply_text": " later "}
        yield {"action": "someone else's"}

    monkeypatch.setattr(ha, "subscribe", lambda cfg, event_type: taps())
    requests.clear()
    await svc.progress.listen()
    assert [json.loads(r.content) for r in requests] == [
        {"text": "Thanks", "user": "alice"},
        {"text": "later", "user": "alice"},
    ]


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
                execution={"instruction": "x"},
                status="scheduled",
                enabled=True,
                created_at=service_module._utcnow_iso(),
            )
        )
        s.commit()
    svc._rehydrate()
    assert svc.get_task("deadbeef").status == "missed"


def test_settings_users(tmp_path):
    svc = make_service(tmp_path)
    assert svc.get_settings().users == []
    alice = UserSettings(name=" Alice ", notify_targets=["mobile_app_alice"])
    out = svc.update_settings(SettingsUpdate(users=[alice]))
    assert out.users[0].name == "Alice"
    # updating one field keeps the other
    out = svc.update_settings(SettingsUpdate(notify_targets=["mobile_app_x"]))
    assert out.users[0].notify_targets == ["mobile_app_alice"]
    # a restricted MCP server is off for a person until listed
    assert out.users[0].servers == []
    alice.servers = ["herdr"]
    svc.update_settings(SettingsUpdate(users=[alice]))
    assert svc.get_settings().users[0].servers == ["herdr"]
    with pytest.raises(ValueError):
        SettingsUpdate(users=[alice, UserSettings(name="alice")])


def test_user_ids(tmp_path):
    svc = make_service(tmp_path)
    # an older record has no id, and gets its casefolded name
    svc.update_settings(
        SettingsUpdate(users=[UserSettings(name="Alice", ha_user_id="ha1")])
    )
    assert svc.user("alice").name == "Alice"
    assert svc.owner(None, "ha1") == "alice"
    # a set id stays when the name changes
    svc.update_settings(
        SettingsUpdate(users=[UserSettings(id="alice", name="Ally", ha_user_id="ha1")])
    )
    assert svc.user("alice").name == "Ally"
    assert svc.user("ally") is None and svc.user(None) is None


def test_adds_columns_to_an_old_database(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE settings (id INTEGER PRIMARY KEY, notify_targets JSON)")
    con.execute("INSERT INTO settings VALUES (1, '[\"mobile_app_x\"]')")
    con.commit()
    con.close()
    engine = make_engine(str(db))
    svc = SchedulerService(make_service(tmp_path).cfg, make_session_factory(engine))
    assert svc.get_settings().notify_targets == ["mobile_app_x"]
    assert svc.get_settings().users == []


def test_tasks_belong_to_their_owner(tmp_path):
    svc = make_service(tmp_path)
    svc.update_settings(
        SettingsUpdate(users=[UserSettings(name="Alice", ha_user_id="ha1")])
    )
    req = TaskCreate(
        description="mine",
        schedule=ScheduleSpec(type="once", run_at=future_iso(hours=1), timezone="UTC"),
        execution=ExecutionSpec(instruction="say hi"),
    )
    alice = svc.owner(None, "ha1")
    assert alice == svc.owner(" ALICE ", None) == "alice"
    out = svc.create_task(req, owner=alice)
    assert out.user == "alice"
    assert [t.id for t in svc.list_tasks(owner="alice")] == [out.id]
    # no one else sees, edits or deletes it
    assert svc.list_tasks(owner="bob") == [] and svc.list_tasks() == []
    assert svc.get_task(out.id, owner="bob") is None
    assert svc.update_task(out.id, TaskUpdate(enabled=False), owner="bob") is None
    assert svc.delete_task(out.id) is None
    assert svc.delete_task(out.id, owner="alice") is not None
