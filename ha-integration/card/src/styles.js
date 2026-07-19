// Card styles. Uses Home Assistant theme variables so the card matches the active
// dashboard theme (light/dark) with no extra configuration.
export const CARD_STYLES = `
  :host { display: block; }

  ha-card {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .lk-header {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .lk-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--primary-text-color);
  }
  .lk-state {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    color: var(--secondary-text-color);
    text-transform: capitalize;
  }
  .lk-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--disabled-text-color, #9e9e9e);
    transition: background-color 0.2s ease;
  }
  .lk-dot[data-active="1"] { background: var(--primary-color); }
  .lk-dot[data-pulse="1"] { animation: lk-pulse 1.4s ease-in-out infinite; }
  @keyframes lk-pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

  .lk-transcript {
    min-height: 44px;
    max-height: 180px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 0.9rem;
    line-height: 1.35;
  }
  .lk-line { color: var(--primary-text-color); }
  .lk-line[data-role="user"] { color: var(--secondary-text-color); }
  .lk-line .lk-who {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--secondary-text-color);
    margin-right: 6px;
  }
  .lk-hint {
    color: var(--secondary-text-color);
    font-size: 0.85rem;
    text-align: center;
  }

  .lk-controls {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
  }
  button.lk-btn {
    font: inherit;
    cursor: pointer;
    border: none;
    border-radius: 999px;
    padding: 12px 20px;
    font-weight: 600;
    color: var(--text-primary-color, #fff);
    background: var(--primary-color);
    transition: transform 0.06s ease, opacity 0.2s ease, background-color 0.2s ease;
    -webkit-user-select: none;
    user-select: none;
    touch-action: none;
  }
  button.lk-btn:disabled { opacity: 0.5; cursor: default; }
  button.lk-talk { min-width: 180px; }
  button.lk-talk[data-holding="1"] {
    background: var(--error-color, #db4437);
    transform: scale(0.98);
  }
  button.lk-secondary {
    background: transparent;
    color: var(--secondary-text-color);
    border: 1px solid var(--divider-color);
  }

  .lk-error {
    color: var(--error-color, #db4437);
    font-size: 0.85rem;
    text-align: center;
  }
`;
