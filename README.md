# Home Assistant Voice Agent (LiveKit)

A voice + text assistant for [Home Assistant](https://www.home-assistant.io/), built on
[LiveKit Agents](https://github.com/livekit/agents). This repo is the **worker** (the agent
brain): it connects to Home Assistant's native
[MCP Server](https://www.home-assistant.io/integrations/mcp_server/) through the framework's
built-in `MCPToolset`, and serves STT / LLM / TTS via LiveKit Inference.

The **dashboard UI** is a Home Assistant custom integration + Lovelace card that runs the
session inside HA:
**[ha-livekit-agent-frontend](https://github.com/longcw/ha-livekit-agent-frontend)**. Run this
worker, install that integration, point both at the same LiveKit project, and you're set.

- **Native MCP** — Home Assistant tools (`HassTurnOn`, `HassLightSet`, `GetLiveContext`, …)
  are exposed to the LLM directly through LiveKit's `MCPToolset`. No custom MCP client code.
- **LiveKit Inference** — STT, LLM, and TTS are served by LiveKit's inference gateway; no
  per-provider API keys required.
- **Cost-aware** — STT is billed continuously, so it follows the mic: enabled while audio
  input is live and torn down `STT_IDLE_TIMEOUT` seconds (default 120) after the mic is
  gated. The worker boots dormant (STT off, no TTS), so an idle card on a dashboard costs
  nothing beyond the worker connection, and text chat works without ever enabling STT.
- **Turn modes** — manual push-to-talk (`start_turn` / `end_turn` / `cancel_turn`) or auto
  turn detection; switchable at runtime via `set_turn_mode`.
- **Helper tools** — a few function tools (`get_areas`, `get_device_domains`, `get_devices`,
  `get_environment_info`) pre-process `GetLiveContext` into compact, area/domain-filtered
  views so the LLM isn't flooded with the full home state.

## Structure

```
.
├── agent/            # Python LiveKit worker (uv)
│   ├── src/          # main.py (entrypoint), agent.py, config.py, ha.py, scheduler_client.py, …
│   ├── prompt.yaml   # system prompt (bind-mounted, edit without a rebuild)
│   ├── pyproject.toml
│   └── Dockerfile
├── scheduler/        # task scheduler service (FastAPI + APScheduler)
├── docker-compose.yml
├── .env.example      # agent + scheduler env
└── README.md
```

## Prerequisites

- Home Assistant with the **MCP Server** integration enabled, and a long-lived access token.
- A LiveKit server (self-hosted or LiveKit Cloud) for the realtime transport.
- LiveKit Inference credentials for STT/LLM/TTS (LiveKit Cloud).
- The **[HA integration](https://github.com/longcw/ha-livekit-agent-frontend)** installed for
  the dashboard UI.

## Run the worker

Uses [uv](https://docs.astral.sh/uv/).

```bash
cd agent
cp ../.env.example ../.env    # then fill in the values (see below)
uv sync
uv run src/main.py dev        # or `console` to test in the terminal
```

Or with Docker Compose (builds and runs the worker):

```bash
docker compose up --build
```

### Environment (`.env` in the repo root)

The agent calls `load_dotenv()`, which walks up to the repo-root `.env`.

| Variable | Purpose |
| --- | --- |
| `HA_URL` | Home Assistant base URL. `/api/mcp` is appended automatically. |
| `HA_TOKEN` | Home Assistant long-lived access token (sent as a bearer token). |
| `AGENT_NAME` | Explicit-dispatch worker name (default `ha-agent`). Must match the integration's **Agent name**. |
| `STT_IDLE_TIMEOUT` | Seconds after the mic is gated before STT is torn down to save cost (default `120`). |
| `LIVEKIT_URL` / `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | Your LiveKit server. |
| `LIVEKIT_INFERENCE_URL` / `LIVEKIT_INFERENCE_API_KEY` / `LIVEKIT_INFERENCE_API_SECRET` | Inference gateway. Falls back to `LIVEKIT_*` if unset. |

Models default to Chinese-friendly choices and can be overridden via
`STT_MODEL`, `STT_LANGUAGE`, `LLM_MODEL`, `TTS_MODEL`, `TTS_VOICE`, `TTS_LANGUAGE`
(see `.env.example`).

## Turn modes & session control

The card configures the session at runtime over RPCs the worker registers:

- **`set_turn_mode`** (`manual` | `auto`) — `manual` is push-to-talk: the mic stays gated
  until `start_turn`, then `end_turn` commits (agent replies) or `cancel_turn` discards.
  `auto` keeps the mic live and lets the model detect turn ends.
- **`start_turn` / `end_turn` / `cancel_turn`** — manual turn lifecycle. `start_turn` opens
  the mic (bringing STT back); `end_turn`/`cancel_turn` gate it (STT tears down after the
  idle timeout). STT simply follows the mic.
- **`set_audio_output`** (`on` | `off`) — toggles the agent's spoken (TTS) replies; off means
  text-only. Off by default, so text chat always works without any STT/TTS cost.

The worker broadcasts its live STT + audio-output state on the `ha.speech_state` data channel
so the card can mirror it (e.g. a "sleeping" indicator when STT is torn down), and the
tool-execution lifecycle on `ha.tool_call`.

## Frontend / dashboard UI

Use the companion Home Assistant integration:
**[ha-livekit-agent-frontend](https://github.com/longcw/ha-livekit-agent-frontend)** — a HACS
custom integration + Lovelace card with native device tiles (tap to control), voice, text
chat, transcript, and tool-call status. It mints LiveKit tokens for logged-in HA users and
dispatches this worker by `AGENT_NAME`.
