// LiveKit Voice Assistant — Home Assistant Lovelace custom card.
//
// Phase 1 (this file): connect to the LiveKit worker from inside the HA frontend, do
// push-to-talk / auto voice, and show a live transcript + agent state. Device tiles
// rendered from `hass.states` (tap-to-toggle, tap-hold more-info) land in phase 2 — the
// data-channel `ha.tool_call` events are already forwarded here (see onToolEvent).

import { LiveKitSession } from './livekit-session.js';
import { CARD_STYLES } from './styles.js';

const DEFAULT_INPUT_MODE = 'push_to_talk'; // matches frontend/lib/input-mode.ts
const MAX_TRANSCRIPT_LINES = 6;
const PULSE_STATES = new Set(['listening', 'thinking', 'speaking']);

class LiveKitVoiceCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._hass = null;
    this._config = {};
    this._inputMode = DEFAULT_INPUT_MODE;
    this._connState = 'disconnected'; // disconnected | connecting | connected
    this._agentState = ''; // listening | thinking | speaking | ...
    this._holding = false;
    this._error = '';
    /** @type {Map<string, {role: string, text: string}>} */
    this._lines = new Map();
    this._lineOrder = [];
    this._session = new LiveKitSession({
      onState: (s) => this._onState(s),
      onTranscript: (line) => this._onTranscript(line),
      onToolEvent: (ev) => this._onToolEvent(ev),
      onError: (msg) => this._onError(msg),
    });
  }

  // --- Lovelace API -------------------------------------------------------

  setConfig(config) {
    this._config = config || {};
    this._inputMode =
      this._config.input_mode === 'auto' ? 'auto' : DEFAULT_INPUT_MODE;
  }

  set hass(hass) {
    this._hass = hass;
  }

  getCardSize() {
    return 4;
  }

  connectedCallback() {
    this._ensureDom();
    this._render();
  }

  disconnectedCallback() {
    // Leave the room if the card is removed from the DOM.
    this._session.disconnect().catch(() => {});
  }

  // --- session callbacks --------------------------------------------------

  _onState(state) {
    if (state === 'connecting') {
      this._connState = 'connecting';
    } else if (state === 'disconnected') {
      this._connState = 'disconnected';
      this._agentState = '';
      this._holding = false;
    } else {
      this._connState = 'connected';
      this._agentState = state === 'ready' ? '' : state;
    }
    this._render();
  }

  _onTranscript({ id, role, text }) {
    if (!text) return;
    if (!this._lines.has(id)) this._lineOrder.push(id);
    this._lines.set(id, { role, text });
    while (this._lineOrder.length > MAX_TRANSCRIPT_LINES) {
      this._lines.delete(this._lineOrder.shift());
    }
    this._renderTranscript();
  }

  _onToolEvent(_event) {
    // Phase 2: map the touched device names/areas to hass.states tiles + highlight.
    // Intentionally a no-op for now beyond forwarding the stream into the card.
  }

  _onError(message) {
    this._error = message;
    this._render();
  }

  // --- connection ---------------------------------------------------------

  async _connect() {
    if (!this._hass) return;
    this._error = '';
    this._onState('connecting');
    try {
      const details = await this._hass.callApi('POST', 'livekit_voice/token', {
        input_mode: this._inputMode,
      });
      await this._session.connect(details, { inputMode: this._inputMode });
    } catch (e) {
      this._onError(`Could not connect: ${e?.message ?? e}`);
      this._onState('disconnected');
    }
  }

  async _disconnect() {
    await this._session.disconnect().catch(() => {});
  }

  // --- push-to-talk handlers ---------------------------------------------

  _onHoldStart(ev) {
    ev.preventDefault();
    if (this._connState !== 'connected' || this._holding) return;
    this._holding = true;
    ev.currentTarget.setPointerCapture?.(ev.pointerId);
    this._render();
    this._session.startTurn().catch((e) => this._onError(String(e)));
  }

  _onHoldEnd(ev) {
    if (!this._holding) return;
    ev.preventDefault();
    this._holding = false;
    this._render();
    this._session.endTurn().catch((e) => this._onError(String(e)));
  }

  // --- rendering ----------------------------------------------------------

  _ensureDom() {
    if (this._root) return;
    const style = document.createElement('style');
    style.textContent = CARD_STYLES;

    const card = document.createElement('ha-card');
    card.innerHTML = `
      <div class="lk-header">
        <span class="lk-title"></span>
        <span class="lk-state"><span class="lk-dot"></span><span class="lk-state-label"></span></span>
      </div>
      <div class="lk-transcript"></div>
      <div class="lk-error" hidden></div>
      <div class="lk-controls"></div>
    `;
    this.shadowRoot.append(style, card);
    this._root = card;
    this._els = {
      title: card.querySelector('.lk-title'),
      dot: card.querySelector('.lk-dot'),
      stateLabel: card.querySelector('.lk-state-label'),
      transcript: card.querySelector('.lk-transcript'),
      error: card.querySelector('.lk-error'),
      controls: card.querySelector('.lk-controls'),
    };
  }

  _render() {
    if (!this._root) return;
    const connected = this._connState === 'connected';
    this._els.title.textContent = this._config.title || 'Voice Assistant';

    this._els.dot.dataset.active = connected ? '1' : '0';
    this._els.dot.dataset.pulse = PULSE_STATES.has(this._agentState) ? '1' : '0';
    this._els.stateLabel.textContent = this._stateLabel();

    this._els.error.hidden = !this._error;
    this._els.error.textContent = this._error;

    this._renderControls();
    this._renderTranscript();
  }

  _stateLabel() {
    if (this._connState === 'connecting') return 'connecting…';
    if (this._connState === 'disconnected') return 'offline';
    if (this._agentState) return this._agentState;
    return this._holding ? 'listening' : 'ready';
  }

  _renderControls() {
    const c = this._els.controls;
    c.replaceChildren();

    if (this._connState === 'disconnected') {
      c.append(this._button('Start voice', 'lk-btn', () => this._connect()));
      return;
    }
    if (this._connState === 'connecting') {
      const btn = this._button('Connecting…', 'lk-btn', () => {});
      btn.disabled = true;
      c.append(btn);
      return;
    }

    // connected
    if (this._inputMode === 'push_to_talk') {
      const talk = this._button('Hold to talk', 'lk-btn lk-talk', () => {});
      talk.dataset.holding = this._holding ? '1' : '0';
      talk.addEventListener('pointerdown', (e) => this._onHoldStart(e));
      talk.addEventListener('pointerup', (e) => this._onHoldEnd(e));
      talk.addEventListener('pointercancel', (e) => this._onHoldEnd(e));
      c.append(talk);
    } else {
      c.append(this._hint('Listening — just speak'));
    }
    c.append(this._button('End', 'lk-btn lk-secondary', () => this._disconnect()));
  }

  _renderTranscript() {
    const t = this._els?.transcript;
    if (!t) return;
    if (this._lineOrder.length === 0) {
      t.replaceChildren(this._hint('Ask about your home — e.g. “turn on the study light”'));
      return;
    }
    const nodes = this._lineOrder.map((id) => {
      const { role, text } = this._lines.get(id);
      const line = document.createElement('div');
      line.className = 'lk-line';
      line.dataset.role = role;
      const who = document.createElement('span');
      who.className = 'lk-who';
      who.textContent = role === 'agent' ? 'AI' : 'You';
      line.append(who, document.createTextNode(text));
      return line;
    });
    t.replaceChildren(...nodes);
    t.scrollTop = t.scrollHeight;
  }

  _button(label, className, onClick) {
    const btn = document.createElement('button');
    btn.className = className;
    btn.textContent = label;
    btn.addEventListener('click', onClick);
    return btn;
  }

  _hint(text) {
    const el = document.createElement('div');
    el.className = 'lk-hint';
    el.textContent = text;
    return el;
  }
}

customElements.define('livekit-voice-card', LiveKitVoiceCard);

window.customCards = window.customCards || [];
window.customCards.push({
  type: 'livekit-voice-card',
  name: 'LiveKit Voice Assistant',
  description: 'Talk to your Home Assistant voice agent from the dashboard.',
  preview: false,
});
