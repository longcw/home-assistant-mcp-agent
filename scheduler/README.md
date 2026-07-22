# Scheduler service

A small FastAPI + [APScheduler](https://apscheduler.readthedocs.io/) service that lets the
voice agent schedule Home Assistant tasks to run later — one-shot ("turn off the AC in 1
hour") or recurring ("every weekday at 8am") — even after the user closes the connection.

## How it fits together

```
agent worker ──REST (create/list/cancel/report)──► scheduler ──LiveKit dispatch──► agent worker
                                                    (SQLite)      (fire time, headless)
```

- The worker exposes `schedule_task` / `list_scheduled_tasks` / `cancel_scheduled_task` /
  `update_scheduled_task` function tools that call this service over the compose network.
- At fire time the service **dispatches the worker into a fresh room** (`kind: "scheduled"`
  metadata). The worker runs the task headlessly (no STT/TTS), reports the outcome back to
  `/internal/runs/{run_id}`, and raises a Home Assistant persistent notification.
- SQLite (`SCHEDULER_DB`, on a volume) is the source of truth; APScheduler runs in memory and
  is rehydrated from the table on boot.

This service holds **no** Home Assistant credentials — all HA access stays in the worker.

## Execution kinds

- `function_call` — a frozen Home Assistant MCP tool + args, replayed deterministically at
  fire time (no LLM). Best for concrete device actions.
- `command` — a natural-language instruction the LLM re-interprets at fire time. Best for
  conditional/complex tasks.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/tasks` | Create a task (see `schemas.TaskCreate`). |
| `GET` | `/tasks?active_only=true` | List tasks (soonest first). |
| `GET` | `/tasks/{id}` | One task + its run history. |
| `PATCH` | `/tasks/{id}` | Modify time / execution / enabled. |
| `DELETE` | `/tasks/{id}` | Cancel a task. |
| `POST` | `/internal/runs/{run_id}` | Worker reports a run's outcome. |
| `GET` | `/healthz` | Liveness. |

## Config (env, shared `.env`)

| Var | Purpose |
| --- | --- |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | Dispatch the worker at fire time. |
| `AGENT_NAME` | Worker dispatch name (must match the worker). |
| `AGENT_TZ` (or `TZ`) | Default timezone for schedules. |
| `SCHEDULER_DB` | SQLite path (default `/data/scheduler.db`). |
| `MISFIRE_GRACE_SECONDS` | Run a one-shot missed during an outage if within this window (default 3600). |

## Develop

```bash
cd scheduler
uv sync
uv run uvicorn main:app --reload --port 8080
uv run pytest
```
