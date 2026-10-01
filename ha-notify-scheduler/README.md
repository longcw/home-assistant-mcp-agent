# ha-notify-scheduler

A small FastAPI + [APScheduler](https://apscheduler.readthedocs.io/) service that holds the
home's people and what the house does for them: tasks scheduled to run later — one-shot
("turn off the AC in 1 hour") or recurring ("every weekday at 8am") — and the Home Assistant
notifications that reach each person. The agent knows a person only by their id; this
service maps that id to their name, HA login, devices and allowed MCP servers.

## How it fits together

```
agent worker ──REST (tasks, users)──► ha-notify-scheduler ──notify service──► Home Assistant
                                          (SQLite)        ──POST /chat────► agent worker
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
  failing tool and can retry. The reply is the run's result. When the owner has no phone
  showing their turns, the reply or the failure is also sent as a notification.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/tasks` | Create a task (see `schemas.TaskCreate`). |
| `GET` | `/tasks?active_only=true` | List tasks (soonest first). |
| `GET` | `/tasks/{id}` | One task + its run history. |
| `PATCH` | `/tasks/{id}` | Modify time / execution / enabled. |
| `DELETE` | `/tasks/{id}` | Cancel a task. |
| `GET` / `PUT` | `/settings` | The people and the default notify channels (the card's Settings tab). |
| `GET` | `/users/{id}` | One person's settings. |
| `POST` | `/notify` | `{user, message, title}`: a notification to the person's channels (HA's own for no one). |
| `POST` | `/progress` | `{user, phase, text, tool, args, ok, replies}`: one phase of a text turn, on the person's phone (see `progress.py`). A tap on the answer's buttons is sent to the agent's `/chat` as the person's next message. |
| `GET` | `/users/resolve?ha_user_id=` | The id of the person linked to an HA login. |
| `GET` | `/healthz` | Liveness. |

## Config (env, shared `.env`)

| Var | Purpose |
| --- | --- |
| `HA_URL` / `HA_TOKEN` | Send reminders and run notifications through Home Assistant. |
| `AGENT_CHAT_URL` | The agent's text chat endpoint (default `http://agent:8081/chat`). |
| `TEXT_API_TOKEN` | Bearer token for that endpoint; unset, instructions fail. |
| `TEXT_LIVE_ACTIVITY` | The phone showing text-turn progress for no one in particular; a person's is the first phone among their devices. |
| `TEXT_LIVE_MODE` / `TEXT_LIVE_URL` / `TEXT_LIVE_CLEAR_AFTER` | How progress shows (`notification` or `activity`), the page a tap opens, and when a Live Activity clears. |
| `SCHEDULED_RUN_TIMEOUT` | Seconds an instruction may take, waiting out a running turn included (default 200). |
| `AGENT_TZ` (or `TZ`) | Default timezone for schedules. |
| `SCHEDULER_DB` | SQLite path (default `/data/scheduler.db`). |
| `MISFIRE_GRACE_SECONDS` | Run a one-shot missed during an outage if within this window (default 3600). |

## Develop

```bash
cd ha-notify-scheduler
uv sync
uv run uvicorn main:app --reload --port 8080
uv run pytest
```
