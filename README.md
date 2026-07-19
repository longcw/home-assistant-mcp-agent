# Home Assistant Voice Agent (LiveKit)

A voice assistant for [Home Assistant](https://www.home-assistant.io/), built on
[LiveKit Agents](https://github.com/livekit/agents). The agent connects to Home
Assistant's native [MCP Server](https://www.home-assistant.io/integrations/mcp_server/)
using the framework's built-in `MCPToolset`, and a push-to-talk web UI (based on
[agent-starter-react](https://github.com/livekit-examples/agent-starter-react)) lets you
talk to it from the browser.

- **Native MCP** — Home Assistant tools (`HassTurnOn`, `HassLightSet`, `GetLiveContext`, …)
  are exposed to the LLM directly through LiveKit's `MCPToolset`. No custom MCP client code.
- **LiveKit Inference** — STT, LLM, and TTS are served by LiveKit's inference gateway; no
  per-provider API keys required.
- **Push-to-talk** — the agent uses manual turn detection; the UI drives it with
  `start_turn` / `end_turn` / `cancel_turn` RPCs.
- **Helper tools** — a few function tools (`get_areas`, `get_device_domains`, `get_devices`,
  `get_environment_info`) pre-process `GetLiveContext` into compact, area/domain-filtered
  views so the LLM isn't flooded with the full home state.

## Structure

```
.
├── agent/            # Python LiveKit agent (uv)
│   ├── agent.py
│   ├── pyproject.toml
│   └── Dockerfile
├── frontend/         # Next.js push-to-talk UI (pnpm)
├── docker-compose.yml
├── .env.example      # agent env
└── README.md
```

## Prerequisites

- Home Assistant with the **MCP Server** integration enabled, and a long-lived access token.
- A LiveKit server (self-hosted or LiveKit Cloud) for the realtime transport.
- LiveKit Inference credentials for STT/LLM/TTS (LiveKit Cloud).

## Agent

Uses [uv](https://docs.astral.sh/uv/) for the environment.

```bash
cd agent
cp ../.env.example ../.env    # then fill in the values (see below)
uv sync
uv run agent.py dev           # or `console` to test in the terminal
```

### Environment (`.env` in the repo root)

The agent calls `load_dotenv()`, which walks up to the repo-root `.env`.

| Variable | Purpose |
| --- | --- |
| `HA_URL` | Home Assistant base URL. `/api/mcp` is appended automatically. |
| `HA_TOKEN` | Home Assistant long-lived access token (sent as a bearer token). |
| `AGENT_NAME` | Explicit-dispatch worker name (default `ha-agent`). Must match the frontend. |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | Your LiveKit server. |
| `LIVEKIT_INFERENCE_URL` / `LIVEKIT_INFERENCE_API_KEY` / `LIVEKIT_INFERENCE_API_SECRET` | Inference gateway. Falls back to `LIVEKIT_*` if unset. |

Models default to Chinese-friendly choices and can be overridden via
`STT_MODEL`, `LLM_MODEL`, `TTS_MODEL`, `TTS_VOICE`, `TTS_LANGUAGE` (see `.env.example`):

- STT `deepgram/nova-3` (`language=multi`)
- LLM `google/gemini-2.5-flash`
- TTS `cartesia/sonic-3` (`language=zh`) — for a native Chinese voice, pick one from
  [play.cartesia.ai/voices](https://play.cartesia.ai/voices) and set `TTS_VOICE`.

## Frontend

Uses [pnpm](https://pnpm.io/).

```bash
cd frontend
cp .env.example .env.local     # fill in LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET
pnpm install
pnpm dev                       # http://localhost:3000
```

The agent is dispatched explicitly by name. `AGENT_NAME` in `.env.local` sets the default
(must match the agent's `AGENT_NAME`), and the **welcome screen** lets you override the
agent name and choose the turn mode per session:

- **Push to talk** — press and hold the button to talk; release to send, or drag off the
  button and release to cancel.
- **Automatic** — the agent detects turns on its own and the mic stays live.

> The `/api/token` route mints room tokens without authentication. That's fine for a
> personal, trusted-network deployment, but add an auth layer before exposing it publicly.

## Docker Compose

```bash
docker compose up --build
```

This builds and runs both services (`agent` and `frontend` on port 3000). The `agent`
service reads `.env`; the `frontend` service reads `frontend/.env.local`.

## Turn modes

The frontend tells the agent which turn mode to use via agent-dispatch metadata
(`{ "input_mode": "push_to_talk" | "auto" }`), and the agent configures itself accordingly:

- **`push_to_talk`** → `turn_detection="manual"`. The UI keeps the mic off; on press it
  enables the mic and calls `start_turn`, and on release calls `end_turn` (generate a reply)
  or `cancel_turn` (discard). The agent handles those RPCs to interrupt, listen, and
  commit/clear the user turn.
- **`auto`** → automatic turn detection (`inference.TurnDetector`); the mic stays live and
  no RPCs are used.
