# LiveKit Voice Assistant for Home Assistant

A Home Assistant **custom integration + Lovelace card** that lets you talk to your LiveKit
voice agent straight from your dashboard, and (phase 2) control devices with native
tap-to-toggle tiles.

The voice **worker** — the agent brain (STT / LLM / TTS + Home Assistant MCP) — lives in
the companion repo **[home-assistant-mcp-agent](https://github.com/longcw/home-assistant-mcp-agent)**
and runs unchanged. This repo is the HA-side glue: it mints LiveKit tokens for logged-in
Home Assistant users and serves a card that runs the voice session inside the HA frontend.

## Architecture

```
LiveKit worker  (companion repo: home-assistant-mcp-agent) — UNCHANGED
  ├─ talks to Home Assistant over MCP  → voice control ("turn on the light")
  └─ streams tool-execution events on the "ha.tool_call" data channel

custom_components/livekit_voice/  (this repo — thin glue integration)
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

## Install (HACS)

1. In HACS, open the **⋮** menu (top right) → **Custom repositories**. Paste
   `https://github.com/longcw/ha-livekit-agent-frontend`, category **Integration**, → **Add**.
2. Find **LiveKit Voice Assistant** in HACS → **Download** → **Restart Home Assistant**.
3. Settings → Devices & Services → **Add Integration** → “LiveKit Voice Assistant”, then:

   | Field | Value |
   |---|---|
   | LiveKit server URL | `wss://your-project.livekit.cloud` |
   | LiveKit API key / secret | your LiveKit project credentials |
   | Agent name | `ha-agent` (must match `AGENT_NAME` in the worker's `agent/agent.py`) |

4. Make sure the **worker** is running (companion repo) and connected to the same LiveKit
   project.
5. Add the card to a dashboard (Edit dashboard → Add card → “LiveKit Voice Assistant”, or
   in YAML):

   ```yaml
   type: custom:livekit-voice-card
   title: Voice Assistant
   input_mode: push_to_talk   # or: auto
   ```

6. Open the dashboard **over HTTPS** and tap **Start voice**.

When a new version is released, HACS offers the update; restart HA and hard-refresh the
dashboard to pick up the new card.

## Phases

- **Phase 1 (implemented):** integration + card that connects and does voice
  (push-to-talk / auto, live transcript, agent state).
- **Phase 2 (next):** native device tiles from `hass.states`, scoped by area/domain from
  the worker's `ha.tool_call` events, with tap-to-toggle and tap-hold more-info.
- **Phase 3 (later):** voice-driven navigation of your existing Lovelace views.

### Why phase-2 tiles render by `entity_id`/area, not by device name

Matching the worker's spoken device *names* back to entities is unreliable in a typical
setup: registry aliases are often absent, one device name can map to many entities
(multi-gang switches, or climate devices that also expose a `switch`/`light`), and real
`friendly_name`s carry channel suffixes — so a bare name often matches zero or the wrong
entity. The card renders from `hass.states` keyed by `entity_id`, **scoped by area** via
HA's own area registry, rather than fuzzy per-device name matching. State stays correct for
free: HA pushes state changes into the card's `hass` object, so tiles update themselves.

## Caveats

- **Microphone needs a secure context.** Open the dashboard over HTTPS (or `localhost`). A
  LAN-IP `http://…:8123` origin cannot use the mic — the card connects but captures nothing.
- **Reload requires a restart.** HA has no runtime unregister for HTTP views / static
  paths / frontend modules, so reconfiguring updates the token config live, but fully
  removing the integration needs a restart.

## Manual install (without HACS)

The built card bundle is committed, so no Node toolchain is needed on the HA host — just
copy the integration into your HA config's `custom_components/` and restart:

```bash
rsync -a custom_components/livekit_voice/ \
  <user>@<your-ha-host>:/config/custom_components/livekit_voice/   # adjust to your config path
```

## Development

- Card source: `card/src/` (`index.js` = the custom element, `livekit-session.js` = the
  LiveKit room wrapper, `styles.js` = theme-aware CSS).
- Rebuild the bundle after changing the card (needs Node; run on your dev machine, not the
  HA host):

  ```bash
  cd card
  pnpm install
  pnpm build     # → ../custom_components/livekit_voice/frontend/livekit-voice-card.js
  pnpm watch     # rebuild on change
  ```

  Commit the rebuilt bundle so HACS / manual installs serve it directly.
- Integration: `custom_components/livekit_voice/` (`token.py` = token endpoint,
  `__init__.py` = setup/registration, `config_flow.py` = setup UI).
