# Text chat endpoint

The agent can also be reached by text over HTTP, for clients such as an iPhone Shortcut. The voice path is untouched: it runs the same worker and never persists.

Text runs on the **draft** session-persistence API of livekit-agents ([livekit/agents#7465](https://github.com/livekit/agents/pull/7465)), with sessions stored in local SQLite rather than agent-db. The API may still change, so everything that touches it is kept in one place; the upgrade checklist is at the end.

## Using it

```bash
curl -X POST http://<agent-host>:8952/chat \
     -H "Authorization: Bearer $TEXT_API_TOKEN" \
     -H 'content-type: application/json' \
     -d '{"text": "客厅的灯开着吗？", "user": "alice"}'
```

- The body is either JSON `{"text": "...", "new": false, "wait": false, "steps": true, "user": "..."}` or plain text. `?new=1`, `?wait=1`, `?steps=0` and `?user=` work too.
- By default it answers `202 accepted` as soon as the turn has started, and the turn runs on in the background: its progress and its answer go to the person's phone (below), and the card's Text tab shows it from the history. With `"wait": true` it answers with the agent's reply instead.
- `user` names the family member speaking; the name is matched without regard to case. Each person has their own conversation and Mem0 memories, and gets notifications and phone progress on the devices ticked for them under People in the card's Settings tab (progress goes to the first ticked phone). A name not listed there still gets its own conversation and memories, with the default devices for notifications and no phone progress. Without `user` the request is no one in particular: the conversation, memories and `TEXT_LIVE_ACTIVITY` phone from before there were people. Through the HA integration (the card's Text tab) the HA login stands in for the name, once a person is linked to it.
- With `wait`, the reply is plain text: one line per tool call the turn made (`→ HassTurnOn({"name": "客厅 灯"})`, plus `  … ` lines for a tool's progress reports), a blank line, then the agent's answer. `"steps": false` returns the answer alone, which suits *Speak Text*. A turn that started a new conversation begins with `(new conversation)`. The `X-Conversation-Id` header names the conversation.
- A new message replaces a turn still running: it stops that turn's speech and the calls that can be stopped, and nothing more of that turn reaches the phone. A tool the agent already moved to the background keeps running. `POST /chat/cancel` `{"user"}` stops the running turn the same way and answers at once, so a client calls it the moment the person starts speaking; `{"task_id"}` from the history stops only that turn, never a newer one.
- A person's requests go into one conversation, so the agent remembers earlier turns, across restarts too, until a new one is asked for: `"new": true`, or the new-conversation button on the card's Text tab (with no `text` it only starts the new one). Setting `TEXT_RENEW_AFTER` to a number of seconds also starts one on its own after that long without a request; it is off by default.
- A loaded conversation keeps only its latest `TEXT_MAX_ITEMS` (100) chat items in the agent's context, cut with `ChatContext.truncate()` each time it loads. The session's own history, which the model never reads and the Text tab shows, is not cut; it grows by a few items per turn until a new conversation starts.
- `wait` is capped at `TEXT_REPLY_TIMEOUT` (80 s), under the reverse proxy's 90 s read timeout. When it runs out, only the waiting stops: the request answers `[timeout] still working`, and the turn runs on and still sends its answer to the phone.
- A conversation not used for `TEXT_IDLE_TIMEOUT` is saved and unloaded, and loading it again takes a few seconds (mostly MCP connects). `POST /chat/warm` `{"user"}` loads the person's current conversation ahead of their message, with no turn, and `/chat/cancel` does the same, since a client calls it as the person starts to speak.
- `POST /chat/events` `{"source", "text"}` takes an update from an MCP server's webhook into the person's conversation, e.g. a Claude Code session that finished. It waits for a running turn rather than replacing it, and the agent answers it with no tools.
- Both endpoints need `TEXT_API_TOKEN` as a bearer token; with it unset, neither is mounted.

### History, and the card's Text tab

`GET /chat/history?limit=200` (same bearer token) returns `{"conversation_id", "busy", "items"}`: the current conversation's messages and tool calls, shaped like the card's conversation items (`{"kind": "message", "role": "user"|"agent", "text"}` and `{"kind": "action", "name", "args", "status"}`). It reads the loaded session while the conversation is live, and the store otherwise; `busy` is true while a turn runs. `?conversation_id=` reads one of the person's past conversations instead, with `current: false` and never `busy`.

`GET /chat/conversations` lists the person's conversations, latest first, as `{"current", "conversations": [{"id", "title", "created", "updated"}]}`; the title is the first message. `POST /chat/switch` `{"conversation_id"}` makes a past one current, so the next turn, from the card or the phone, continues it; it is refused while a turn runs. `POST /chat/delete` `{"conversation_id"}` removes a past conversation's SQLite files; the current one, and one still loaded, are never deleted. Who owns which conversation, and its title and last turn, are kept in `agent-data/conversations.json`; a conversation missing from it belongs to no one and is not listed.

The HA integration (`ha-livekit-agent-frontend`) proxies these and `POST /chat` as `/api/livekit_voice/chat[/history|/conversations|/switch|/delete|/cancel]`, configured with `chat_url` (here `http://127.0.0.1:8952`, since HA runs on the host network) and `chat_token` (`TEXT_API_TOKEN`). The card's **Text** tab renders that conversation (its history button lists the others, to read, continue or delete), polls it every second while a turn runs, and sends into it; `?lk_tab=text` on a dashboard URL opens the tab, which is what `TEXT_LIVE_URL` points the phone progress at.

The chat routes are registered before `@server.a2a_session`: the A2A binding mounts a catch-all that shadows GET routes added after it.

### Progress on the phone

With `TEXT_LIVE_ACTIVITY` set to a phone's notify service (`mobile_app_my_iphone`), each turn's progress also goes to that phone, with no unlock needed to read it. Every push carries the tag `ha-text`, so each replaces the last instead of piling up. Steps show as `✓` done and `…` running, with arguments as plain values (`HassTurnOff · 背景灯 电视 左键`), then the answer. Tapping opens `TEXT_LIVE_URL`, the card's Text tab. `TEXT_LIVE_MODE` picks how:

- `notification` (default): an ordinary notification titled with the question. The question and the answer pop with sound; each tool call is appended to the question's notification as a `passive` update without sound, so it never pops, and once that notification is gone it arrives quietly on its own. iOS alerts only once per tag, so the answer first sends `clear_notification` for the tag and then posts anew. Plain remote pushes, so it works with the app closed, but they can lag by a few seconds.
- `activity`: the agent only reports each turn's phases, `start` (the question), `progress` (each tool call) and `final` (the answer), to the HA integration's `POST /api/livekit_voice/progress` (in `ha-livekit-agent-frontend`, `progress.py`), with the HA token. That view runs inside Home Assistant, so it can read from mobile_app whether the phone's Live Activity for the tag is running — which the phone reports when an activity starts and withdraws when it is swiped away — and does the rest:
  - `start` starts the activity, or updates the running one, and always alerts.
  - `progress` updates a running activity quietly; while a start is still waiting for the phone's token, it holds the latest update and sends it once the token appears (up to 20 seconds), since mobile_app sends an update for a tag it has no token for as another push-to-start; with no activity at all it is skipped.
  - `final` shows the answer on the activity, and sends it as a regular notification (tag `ha-text-answer`, cleared first so it pops, carrying the reply buttons). On a running activity the answer alerts there and the notification is quiet; otherwise, or when HA cannot tell, the notification sounds. The activity clears `TEXT_LIVE_CLEAR_AFTER` (900) seconds after the last turn.

  The activity is titled "Home Assistant" (iOS fixes the title at start), one line per update with `critical_text` as a short status (`…`, `Step 2`, `Done`) and an icon per tool. Quiet updates pass a title-only `alert: {title: ""}`, which the push relay sends as given at full priority; `silent: true` drops to APNs priority 5, which iOS never showed. A start must carry a real alert: iOS starts no activity from a push whose alert is empty.

**What broke it, and how to tell.** Found on 2026-09-27/28: (1) iOS rations push-to-start: after a burst of starts (about seven in 25 minutes on 2026-09-28, with testing) HA-sent starts, the Companion app's own examples sent through HA included, stop starting on the phone; a phone restart brought them back once, and the allowance also refills over hours. The in-app examples keep working throughout because they start locally, not through HA; (2) a new activity per turn, cleared a minute later, spent that allowance fast; (3) `silent` updates go out at APNs priority 5 and the phone never applied them. Updates never use `silent`; a quiet step uses the title-only alert instead. To see what happens, `logger.set_level` `homeassistant.components.mobile_app: debug`: a remote send logs a `rate limits` line, and a started activity makes the phone send a `live_activity_token` webhook within seconds; no such webhook means it never started. Upstream, [home-assistant/iOS#5766](https://github.com/home-assistant/iOS/issues/5766) is open for push-to-start with the app closed.

### Reply buttons

The answer notification carries buttons: each quick reply the agent offered with `suggest_replies` in that turn (such as 确认 / 取消 when it asks to confirm), plus a **Reply** button that takes free text. Tapping one sends that text into the same conversation as the next turn, from the lock screen, without opening the Shortcut. iOS shows the buttons on a long press or when the notification is expanded; each one asks for Face ID first, since a tap can control the house. They are HA [actionable notifications](https://companion.home-assistant.io/docs/notifications/actionable-notifications/) (`activationMode` background): the tap fires `mobile_app_notification_action` in HA, which the agent subscribes to over HA's WebSocket API (`ha.subscribe`). Only the latest answer's buttons are live, and they do not survive an agent restart.

### iPhone Shortcut

`scripts/make_shortcut.py` builds a signed `.shortcut` file to import (it needs a Mac, for the `shortcuts sign` CLI):

```bash
uv run --no-project python scripts/make_shortcut.py -o "Ask Home.shortcut" --url http://<agent-host>:8952/chat --token "$TEXT_API_TOKEN" --user alice
```

AirDrop the file to the iPhone, or open it on a Mac signed in to the same iCloud account, and import it. The import asks for the URL, the token and the person's name, prefilled with `--url` (the agent's `/chat`, as the phone reaches it), `--token` and `--user`. The name is sent as `user` on every request; left empty, the shortcut talks as no one in particular. Without `--token` the file holds no secret and can be shared, so one file serves the whole family, each person typing their own name on import. Run on its own, it first calls `POST /chat/cancel`, so the agent stops at once, then dictates what you say and sends it without waiting; the answer arrives as the Home Assistant notification. Run from another shortcut with text as its input, it sends that text instead of dictating.

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

- after `TEXT_IDLE_TIMEOUT` seconds (1800) with no request, when the endpoint drops the context;
- when a new conversation replaces it (`"new": true`, or `TEXT_RENEW_AFTER` when set; the bridge sends the A2A goodbye first);
- on a graceful shutdown (`docker compose stop`/`restart`; `stop_grace_period: 30s` in the compose file).

A crash loses the turns since the last save. The next request after a drop or a restart rehydrates the session from SQLite (`rehydrated a persisted session` in the log).

## Where the draft API is used

| What | Where |
| --- | --- |
| `livekit-agents` and `livekit-protocol` pinned to commits (`[tool.uv.sources]`) | `agent/pyproject.toml`, `agent/uv.lock` |
| `git` in the image, for those pins | `agent/Dockerfile` |
| `AgentServer(store=...)` | `agent/src/main.py`, one line |
| `livekit.agents.store.LocalStore`, `create_database()`, `session(...).load()`/`.release()` and `SessionRecord.history` (the history view of an unloaded conversation), `StoreError`, the private `SessionStore._databases` and the `<id>.sqlite` file naming (deleting a conversation) | `agent/src/text_chat.py` |
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

## Deploying

On the Docker host, check out the branch, make sure `.env` has `TEXT_API_TOKEN`, then `docker compose up -d --build agent`. The build fetches the pinned commits from GitHub.
