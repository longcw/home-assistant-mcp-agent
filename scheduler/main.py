"""FastAPI surface for the scheduler service.

The worker's scheduling tools call the ``/tasks`` CRUD endpoints over the compose network,
and the Home Assistant integration reads tasks and settings through the published port.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException

from config import load_config
from db import make_engine, make_session_factory
from schemas import SettingsOut, SettingsUpdate, TaskCreate, TaskOut, TaskUpdate
from service import SchedulerService

logging.basicConfig(level=logging.INFO)

load_dotenv()

cfg = load_config()
engine = make_engine(cfg.db_path)
Session = make_session_factory(engine)
service = SchedulerService(cfg, Session)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.start()
    yield
    await service.stop()


def require_token(authorization: str | None = Header(default=None)) -> None:
    """Guard every non-health route once the service is exposed beyond the compose net.

    No-op when SCHEDULER_TOKEN is unset (purely internal deployment).
    """
    if not cfg.auth_token:
        return
    if authorization != f"Bearer {cfg.auth_token}":
        raise HTTPException(status_code=401, detail="unauthorized")


app = FastAPI(title="HA Agent Scheduler", lifespan=lifespan)
guard = [Depends(require_token)]


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


def owner(user: str | None = None, ha_user_id: str | None = None) -> str | None:
    """The person a task request acts for; other people's tasks are not found."""
    return service.owner(user, ha_user_id)


Owner = Annotated[str | None, Depends(owner)]


@app.post("/tasks", response_model=TaskOut, status_code=201, dependencies=guard)
def create_task(req: TaskCreate, who: Owner) -> TaskOut:
    try:
        return service.create_task(req, owner=who)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tasks", response_model=list[TaskOut], dependencies=guard)
def list_tasks(who: Owner, active_only: bool = True) -> list[TaskOut]:
    return service.list_tasks(active_only=active_only, owner=who)


@app.get("/tasks/{task_id}", response_model=TaskOut, dependencies=guard)
def get_task(task_id: str, who: Owner) -> TaskOut:
    out = service.get_task(task_id, owner=who)
    if out is None:
        raise HTTPException(status_code=404, detail="task not found")
    return out


@app.patch("/tasks/{task_id}", response_model=TaskOut, dependencies=guard)
def update_task(task_id: str, req: TaskUpdate, who: Owner) -> TaskOut:
    try:
        out = service.update_task(task_id, req, owner=who)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if out is None:
        raise HTTPException(status_code=404, detail="task not found")
    return out


@app.delete("/tasks/{task_id}", response_model=TaskOut, dependencies=guard)
def delete_task(task_id: str, who: Owner) -> TaskOut:
    out = service.delete_task(task_id, owner=who)
    if out is None:
        raise HTTPException(status_code=404, detail="task not found")
    return out


@app.get("/settings", response_model=SettingsOut, dependencies=guard)
def get_settings() -> SettingsOut:
    return service.get_settings()


@app.put("/settings", response_model=SettingsOut, dependencies=guard)
def update_settings(req: SettingsUpdate) -> SettingsOut:
    return service.update_settings(req)
