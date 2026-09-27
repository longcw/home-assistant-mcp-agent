# Text chat endpoint

The agent can also be reached by text over HTTP, for clients such as an iPhone Shortcut. The voice path is untouched: it runs the same worker and never persists.

Text runs on the **draft** session-persistence API of livekit-agents ([livekit/agents#7465](https://github.com/livekit/agents/pull/7465)), with sessions stored in local SQLite rather than agent-db. The API may still change, so everything that touches it is kept in one place; the upgrade checklist is at the end.

## Using it

```bash
curl -X POST http://192.168.100.121:8952/chat \
     -H "Authorization: Bearer $TEXT_API_TOKEN" \
     -H 'content-type: application/json' \
     -d '{"text": "客厅的灯开着吗？"}'
```

- The body is either JSON `{"text": "...", "new": false, "steps": true}` or plain text. `?new=1` and `?steps=0` work too.
- The reply is plain text: one line per tool call the turn made (`→ HassTurnOn({"name": "客厅 灯"})`, plus `  … ` lines for a tool's progress reports), a blank line, then the agent's answer. `"steps": false` returns the answer alone, which suits *Speak Text*. A turn that started a new conversation begins with `(new conversation)`. The `X-Conversation-Id` header names the conversation.
- The reply comes back whole, not streamed: an iPhone Shortcut's *Get Contents of URL* waits for the full body, so streamed output would show no sooner there. The step lines are how a turn's work is shown, and the phone progress below is how it is shown while it runs.
- Requests go into one conversation, so the agent remembers earlier turns, across restarts too. The next request after `TEXT_RENEW_AFTER` (8 h) without one starts a new conversation on its own, so an evening's chat is not the next morning's. `"new": true` starts one now; with no `text` it only does that.
- A loaded conversation keeps only its latest `TEXT_MAX_ITEMS` (100) chat items in the agent's context, cut with `ChatContext.truncate()` each time it loads. The session's own history, which the model never reads, is not cut, and a renewal bounds it anyway.
- A reply is capped at `TEXT_REPLY_TIMEOUT` (80 s), under the reverse proxy's 90 s read timeout. A turn that runs out is cancelled and returns whatever the agent had said so far.
- Both endpoints need `TEXT_API_TOKEN` as a bearer token; with it unset, neither is mounted.

### History, and the card's Text tab

`GET /chat/history?limit=200` (same bearer token) returns `{"conversation_id", "busy", "items"}`: the current conversation's messages and tool calls, shaped like the card's conversation items (`{"kind": "message", "role": "user"|"agent", "text"}` and `{"kind": "action", "name", "args", "status"}`). It reads the loaded session while the conversation is live, and the store otherwise; `busy` is true while a turn runs.

The HA integration (`ha-livekit-agent-frontend`) proxies it and `POST /chat` as `/api/livekit_voice/chat[/history]`, configured with `chat_url` (here `http://127.0.0.1:8952`, since HA runs on the host network) and `chat_token` (`TEXT_API_TOKEN`). The card's **Text** tab renders that conversation, polls it every second while a turn runs, and sends into it; `?lk_tab=text` on a dashboard URL opens the tab, which is what `TEXT_LIVE_URL` points the phone progress at.

The chat routes are registered before `@server.a2a_session`: the A2A binding mounts a catch-all that shadows GET routes added after it.

### Progress on the phone

With `TEXT_LIVE_ACTIVITY` set to a phone's notify service (`mobile_app_long_s_iphone_air`), each turn's progress also goes to that phone, with no unlock needed to read it. Every push carries the tag `ha-text`, so each replaces the last instead of piling up. Steps show as `✓` done and `…` running, with arguments as plain values (`HassTurnOff · 背景灯 电视 左键`), then the answer. Tapping opens `TEXT_LIVE_URL`, the card's Text tab. `TEXT_LIVE_MODE` picks how:

- `notification` (default): an ordinary notification with the question as its title. Step updates are `interruption-level: passive`, so they replace it quietly; the answer is `active`, so the phone lights up when it is done. Reliable with the app closed, since it is a plain remote push.
- `activity`: a Live Activity on the lock screen and Dynamic Island ([docs](https://companion.home-assistant.io/docs/notifications/live-activities/)), titled "Home Assistant" (iOS fixes the title at start) with `› <question>` over the steps, and an icon per tool. Two limits made it not the default. Push-to-start does not start an activity while the Companion app is closed ([home-assistant/iOS#5766](https://github.com/home-assistant/iOS/issues/5766), open as of 2026-09); HA holds no activity token afterwards (`.storage/mobile_app` → `live_activity_tokens`), which is how to tell it never started. And iOS rations push-to-start, refilling over minutes to hours, so all turns share one activity, cleared `TEXT_LIVE_CLEAR_AFTER` (900) seconds after the last; a first version started one per turn and ran dry after about twenty. Switch back once #5766 is fixed.

### iPhone Shortcut

`scripts/make_shortcut.py` builds a signed `.shortcut` file to import (it needs a Mac, for the `shortcuts sign` CLI):

```bash
uv run --no-project python scripts/make_shortcut.py -o "Ask Home.shortcut" --token "$TEXT_API_TOKEN"
```

AirDrop the file to the iPhone, or open it on a Mac signed in to the same iCloud account, and import it. The import asks for the URL and the token, prefilled with `--url` (default `http://192.168.100.121:8952/chat`, reachable at home or over Tailscale) and `--token`. Without `--token` the file holds no secret and can be shared. The text it sends is its Shortcut Input, and the reply (tool calls, then the answer) is its output. From another shortcut: *Dictate Text* → *Run Shortcut* "Ask Home" with the dictated text as input → *Show Result* on its output. Run on its own, it asks for the text and shows the reply.

## How it works

```
POST /chat  ──►  bridge (A2AClient)  ──►  /home-assistant  (A2A endpoint)  ──►  AgentSession + HomeAssistantAgent
   plain text        TaskInput JSON           one context per conversation            persist=ctx.persisted → LocalStore
```

- `/home-assistant` is `@server.a2a_session(...)` on the agent server's own HTTP app (port 8081 in the container, published as 8952). It runs in the worker's main process, not in a job process, and has no room: text in, text out, LLM only.
- `/chat` is a FastAPI route on the same app. It turns one line into a `TaskInput`, sends it to `/home-assistant` over loopback with the framework's `A2AClient`, and turns the `TaskUpdate` stream back into one line. So the JSON wire format is the framework's business, not this repo's.
- One conversation is one SQLite file, `agent-data/sessions/<conversation id>.sqlite`, created with `LocalStore.create_database()`. The id of the current one is in `agent-data/current_conversation`, whose modification time is the last turn's, which the renewal reads.
- The agent is the only one in the conversation, so its session is the conversation's *front session*: the bridge uses the conversation id as the A2A context id, as the branch's `examples/voice_agents/delegation/chat.py` does.

### When a conversation is saved

The framework saves a session **once, when it closes**. Here that is:

- after `TEXT_IDLE_TIMEOUT` seconds (300) with no request, when the endpoint drops the context;
- when `"new": true` or the renewal after `TEXT_RENEW_AFTER` closes it (the bridge sends the A2A goodbye first);
- on a graceful shutdown (`docker compose stop`/`restart`; `stop_grace_period: 30s` in the compose file).

A crash loses the turns since the last save. The next request after a drop or a restart rehydrates the session from SQLite (`rehydrated a persisted session` in the log).

## Where the draft API is used

| What | Where |
| --- | --- |
| `livekit-agents` and `livekit-protocol` pinned to commits (`[tool.uv.sources]`) | `agent/pyproject.toml`, `agent/uv.lock` |
| `git` in the image, for those pins | `agent/Dockerfile` |
| `AgentServer(store=...)` | `agent/src/main.py`, one line |
| `livekit.agents.store.LocalStore`, `create_database()`, `session(...).load()`/`.release()` and `SessionRecord.history` (the history view of an unloaded conversation), `StoreError` | `agent/src/text_chat.py` |
| `@server.a2a_session(endpoint=, description=, idle_timeout=)`, `A2ASessionContext.persisted`, `ctx.attach()` | `agent/src/text_chat.py` |
| `AgentSession.start(persist=)` | `agent/src/text_chat.py` |
| `livekit.agents.a2a.A2AClient(url, context_id=, headers=)`, `.send()`, `.aclose()`, `TaskInput(text=, conversation_id=)`, `TaskUpdate.state`/`.text`/`.item` (a `FunctionCall` with `update_of` for a progress report) | `agent/src/text_chat.py` |

Nothing else in the repo imports from `livekit.agents.store` or `livekit.agents.a2a`.

## Upgrading

1. Move the `livekit-agents` rev in `agent/pyproject.toml` to the new commit of the branch (or drop the source entirely once the feature is released, and raise the version floor instead). Do the same for `livekit-protocol` once a release carries `livekit.protocol.agentdb`; the `livekit-protocol` dependency line exists only so the git pin applies, and can go with it.
2. `cd agent && uv lock && uv sync`, then `uv run python -c "import main"` from `src/` catches renamed imports.
3. Read the branch's `PERSISTENCE_HANDOFF.md` "Public API" section and diff it against the table above. The names that moved most during the draft were the store types (`Store` → `SessionStore`, `Session` → `StoredSession`) and the A2A context/conversation arguments.
4. Test locally before deploying. Use another `AGENT_NAME` so the local worker cannot take voice dispatches meant for the deployed one:

   ```bash
   cd agent
   AGENT_NAME=ha-agent-dev TEXT_API_TOKEN=devtoken TEXT_DATA_DIR=/tmp/textdata \
     HTTP_PORT=8952 TEXT_IDLE_TIMEOUT=30 uv run src/main.py start
   ```

   Then check: a turn answers; a follow-up remembers it; after 30 s idle the SQLite file has rows (`select owner, count(*) from chat_items group by owner`) and the next turn logs `rehydrated a persisted session`; a restart keeps memory; `?new=1` answers with no memory of the old conversation.
5. If the stored schema changed, the old SQLite files may no longer load (`SchemaVersionError`, or a failed turn). They hold only chat history, so start over with `?new=1`, or move `agent-data/` aside.
6. If agent-db replaces SQLite, only `store = LocalStore(...)` and its `create_database()` call change; `AgentDB()` takes its URL and key from the environment.

## Deploying (fnOS)

The stack lives at `/dockers/home-assistant-mcp-agent` on `192.168.100.121`. Check out the branch there, make sure `.env` has `TEXT_API_TOKEN`, then `docker compose up -d --build agent`. The build fetches the pinned commits from GitHub.
