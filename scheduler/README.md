# Scheduler service

A small FastAPI + [APScheduler](https://apscheduler.readthedocs.io/) service that lets the
voice agent schedule Home Assistant tasks to run later — one-shot ("turn off the AC in 1
hour") or recurring ("every weekday at 8am") — even after the user closes the connection.

## How it fits together

```
agent worker ──REST (create/list/cancel)──► scheduler ──notify service──► Home Assistant
                                             (SQLite)  ──POST /chat────► agent worker
```

- The worker exposes `schedule_task` / `list_scheduled_tasks` / `cancel_scheduled_task` /
  `update_scheduled_task` function tools that call this service over the compose network.
- At fire time the service carries the task out itself (`runner.py`) and stores the outcome
  on the task's run history. No LiveKit room or job is involved.
- SQLite (`SCHEDULER_DB`, on a volume) is the source of truth; APScheduler runs in memory and
  is rehydrated from the table on boot.

## Execution kinds

A task's `execution` holds exactly one of:

- `notification` — `{message, title?}`, a reminder sent to the owner's Home Assistant
  channels (the Settings tab's devices for a person, HA's own notification for anyone else).
  The title defaults to the task's description.
- `instruction` — the action to do at fire time, sent as a message in the owner's text
  conversation (`POST /chat` with `wait`), so the agent resolves devices then, sees a
  failing tool and can retry. The reply is the run's result. When no phone showed the turn
  (no `X-Phone` header), the reply or the failure is also sent as a notification.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/tasks` | Create a task (see `schemas.TaskCreate`). |
| `GET` | `/tasks?active_only=true` | List tasks (soonest first). |
| `GET` | `/tasks/{id}` | One task + its run history. |
| `PATCH` | `/tasks/{id}` | Modify time / execution / enabled. |
| `DELETE` | `/tasks/{id}` | Cancel a task. |
| `GET` | `/healthz` | Liveness. |

## Config (env, shared `.env`)

| Var | Purpose |
| --- | --- |
| `HA_URL` / `HA_TOKEN` | Send reminders and run notifications through Home Assistant. |
| `AGENT_CHAT_URL` | The agent's text chat endpoint (default `http://agent:8081/chat`). |
| `TEXT_API_TOKEN` | Bearer token for that endpoint; unset, instructions fail. |
| `SCHEDULED_RUN_TIMEOUT` | Seconds an instruction may take, waiting out a running turn included (default 200). |
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
