# LiveKit Voice Assistant — Home Assistant integration + dashboard card

Brings the LiveKit voice agent *into* Home Assistant: a custom integration serves a
Lovelace card that runs the voice session inside the HA frontend and (phase 2) renders
device tiles from live `hass.states`. The existing standalone web app (`../frontend`) is
untouched — this is an additive, parallel surface.

## Architecture

```
LiveKit worker (../agent/agent.py) — UNCHANGED
  ├─ talks to Home Assistant over MCP  → voice control ("turn on the light")
  └─ streams tool-execution events on the "ha.tool_call" data channel

custom_components/livekit_voice/  (thin glue integration)
  ├─ token.py  → POST /api/livekit_voice/token: mints a LiveKit token for the
  │              logged-in HA user + explicitly dispatches the worker (agent_name)
  └─ __init__.py → serves the built card and registers it as a frontend module

card/  (vanilla web component, bundled with esbuild → custom_components/.../frontend/)
  ├─ connects to LiveKit from the HA frontend, does push-to-talk / auto voice
  ├─ plays agent audio, shows a live transcript + agent state
  └─ (phase 2) renders tiles from hass.states: tap = hass.callService,
     tap-hold = native hass-more-info
```

Two independent control paths that both end at HA and both reflect back into the card:
**voice → worker → MCP → HA**, and (phase 2) **tap → card → `hass.callService` → HA**.

## Phases

- **Phase 1 (implemented):** integration + card that connects and does voice
  (push-to-talk / auto, transcript, agent state). Proves the end-to-end plumbing inside
  your dashboard.
- **Phase 2 (next):** native device tiles from `hass.states`, scoped by area/domain from
  the worker's `ha.tool_call` events, with tap-to-toggle and tap-hold more-info.
- **Phase 3 (later):** voice-driven navigation of your existing Lovelace views
  (`/lovelace/living-room`, `/lovelace/bedroom`, …).

### Why phase-2 tiles render by `entity_id`/area, not by device name

Matching the worker's spoken device *names* back to entities is unreliable in this setup:
there are no registry aliases, one device name maps to many entities (multi-gang switches,
AC/浴霸 devices expose `climate.*` **and** `switch.*`/`light.*`), and real `friendly_name`s
carry channel suffixes — so a bare name often matches zero or the wrong entity. The card
therefore renders from `hass.states` keyed by `entity_id`, **scoped by area** via HA's own
area registry (~10 distinct areas → reliable), rather than fuzzy per-device name matching.
State stays correct for free: HA pushes state changes into the card's `hass` object, so
tiles update themselves even without a name→id match.

## Install

**1. Build the card** (needs Node; run on your dev machine, not the HA host):

```bash
cd ha-integration/card
pnpm install
pnpm build      # → ../custom_components/livekit_voice/frontend/livekit-voice-card.js
```

The built bundle is committed, so the HA host (no Node) serves it straight from the repo.

**2. Copy the integration** into your HA config's `custom_components/`:

```bash
# e.g. for the Docker HA at 192.168.100.198 (config → /home/longc/homeassistant/data)
rsync -a ha-integration/custom_components/livekit_voice/ \
  longc@192.168.100.198:/home/longc/homeassistant/data/custom_components/livekit_voice/
```

**3. Restart Home Assistant.**

**4. Add the integration:** Settings → Devices & Services → **Add Integration** →
“LiveKit Voice Assistant”, then enter:

| Field | Value |
|---|---|
| LiveKit server URL | `wss://your-project.livekit.cloud` |
| LiveKit API key / secret | your LiveKit project credentials |
| Agent name | `ha-agent` (must match `AGENT_NAME` in `../agent/agent.py`) |

**5. Make sure the worker is running** (`../agent`), dispatched by that agent name.

**6. Add the card** to a dashboard (Edit dashboard → Add card → “LiveKit Voice
Assistant”, or in YAML):

```yaml
type: custom:livekit-voice-card
title: Voice Assistant
input_mode: push_to_talk   # or: auto
```

**7. Open the dashboard over HTTPS** and tap **Start voice**.

## Caveats

- **Microphone needs a secure context.** Open the dashboard over HTTPS (or `localhost`).
  A LAN-IP `http://…:8123` origin cannot use the mic — the card will connect but capture
  nothing.
- **Reload requires a restart.** HA has no runtime unregister for HTTP views / static
  paths / frontend modules, so reconfiguring updates the token config live, but removing
  the integration fully needs a restart.
- After rebuilding the card, bump/refresh to bust the frontend cache (hard-refresh, or
  force-quit the companion app).

## Development

- Card source: `card/src/` (`index.js` = the custom element, `livekit-session.js` = the
  LiveKit room wrapper, `styles.js` = theme-aware CSS). `pnpm watch` rebuilds on change.
- Integration: `custom_components/livekit_voice/` (`token.py` = token endpoint,
  `__init__.py` = setup/registration, `config_flow.py` = setup UI).
