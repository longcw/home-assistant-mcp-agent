// Thin wrapper around a LiveKit Room for the voice card.
//
// It owns the browser side of the same protocol the worker (agent/agent.py) speaks:
//   - publishes the mic and plays back the agent's audio
//   - reflects the agent's state via the `lk.agent.state` participant attribute
//   - drives push-to-talk through the agent's `start_turn` / `end_turn` RPC methods
//   - forwards `ha.tool_call` data-channel events (used by the status tiles in phase 2)
//   - surfaces transcriptions from the `lk.transcription` text stream
//
// All the DOM/UI lives in index.js; this module only emits callbacks.

import {
  ParticipantKind,
  Room,
  RoomEvent,
  Track,
} from 'livekit-client';

// Matches TOOL_CALL_TOPIC in agent/agent.py.
const TOOL_CALL_TOPIC = 'ha.tool_call';
const TRANSCRIPTION_TOPIC = 'lk.transcription';
const AGENT_STATE_ATTR = 'lk.agent.state';

export class LiveKitSession {
  /**
   * @param {object} handlers
   * @param {(state: string) => void} handlers.onState  connection/agent state
   * @param {(line: {role: string, text: string, id: string, final: boolean}) => void} handlers.onTranscript
   * @param {(event: object) => void} handlers.onToolEvent  parsed ha.tool_call payload
   * @param {(message: string) => void} handlers.onError
   */
  constructor(handlers = {}) {
    this._h = handlers;
    /** @type {Room|null} */
    this._room = null;
    this._agentIdentity = null;
    this._localIdentity = null;
    this._audioEls = new Set();
    this._decoder = new TextDecoder();
    this._segCounter = 0;
  }

  get connected() {
    return this._room?.state === 'connected';
  }

  async connect(details, { inputMode }) {
    this._emitState('connecting');
    const room = new Room({ adaptiveStream: false, dynacast: false });
    this._room = room;
    this._wire(room);

    await room.connect(details.serverUrl, details.participantToken);
    this._localIdentity = room.localParticipant.identity;
    // Publish the mic. In push-to-talk the worker gates its own input until start_turn,
    // so we keep the track published either way and let the worker decide when to listen.
    await room.localParticipant.setMicrophoneEnabled(true);
    // Unlock audio playback within the connect user-gesture (browser autoplay policy).
    try {
      await room.startAudio();
    } catch (_e) {
      /* will retry on the next user gesture if blocked */
    }

    this._captureAgent(room);
    this._emitState(inputMode === 'push_to_talk' ? 'ready' : 'listening');
  }

  async disconnect() {
    const room = this._room;
    this._room = null;
    this._agentIdentity = null;
    this._localIdentity = null;
    for (const el of this._audioEls) el.remove();
    this._audioEls.clear();
    if (room) await room.disconnect();
    this._emitState('disconnected');
  }

  // --- push-to-talk -------------------------------------------------------

  async startTurn() {
    await this._rpc('start_turn');
  }

  async endTurn() {
    await this._rpc('end_turn');
  }

  async cancelTurn() {
    await this._rpc('cancel_turn');
  }

  async _rpc(method) {
    const room = this._room;
    if (!room || !this._agentIdentity) return;
    try {
      await room.localParticipant.performRpc({
        destinationIdentity: this._agentIdentity,
        method,
        payload: '',
      });
    } catch (e) {
      this._emitError(`${method} failed: ${e?.message ?? e}`);
    }
  }

  // --- wiring -------------------------------------------------------------

  _wire(room) {
    room
      .on(RoomEvent.Disconnected, () => this._emitState('disconnected'))
      .on(RoomEvent.TrackSubscribed, (track) => {
        if (track.kind === Track.Kind.Audio) {
          const el = track.attach();
          el.style.display = 'none';
          document.body.appendChild(el);
          this._audioEls.add(el);
        }
      })
      .on(RoomEvent.ParticipantConnected, (p) => this._maybeAgent(p))
      .on(RoomEvent.ParticipantAttributesChanged, (_changed, p) => {
        if (this._isAgent(p)) this._reflectAgentState(p);
      })
      .on(RoomEvent.DataReceived, (payload, _p, _kind, topic) => {
        if (topic !== TOOL_CALL_TOPIC) return;
        try {
          this._h.onToolEvent?.(JSON.parse(this._decoder.decode(payload)));
        } catch (_e) {
          /* ignore malformed events */
        }
      });

    this._registerTranscription(room);
  }

  _captureAgent(room) {
    for (const p of room.remoteParticipants.values()) this._maybeAgent(p);
  }

  _isAgent(p) {
    return p?.kind === ParticipantKind.AGENT || p?.isAgent === true;
  }

  _maybeAgent(p) {
    if (!this._isAgent(p)) return;
    this._agentIdentity = p.identity;
    this._reflectAgentState(p);
  }

  _reflectAgentState(p) {
    const state = p.attributes?.[AGENT_STATE_ATTR];
    if (state) this._emitState(state);
  }

  _registerTranscription(room) {
    try {
      room.registerTextStreamHandler(TRANSCRIPTION_TOPIC, async (reader, info) => {
        const id = reader.info?.id ?? `seg-${this._segCounter++}`;
        // The local participant's own speech is the user; anything else is the agent.
        const role = info?.identity === this._localIdentity ? 'user' : 'agent';
        let text = '';
        try {
          for await (const chunk of reader) {
            text += chunk;
            this._h.onTranscript?.({ id, role, text, final: false });
          }
        } finally {
          this._h.onTranscript?.({ id, role, text, final: true });
        }
      });
    } catch (_e) {
      /* transcription is best-effort; a handler may already be registered */
    }
  }

  _emitState(state) {
    this._h.onState?.(state);
  }

  _emitError(message) {
    this._h.onError?.(message);
  }
}
