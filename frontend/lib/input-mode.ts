export type InputMode = 'push_to_talk' | 'auto';

/**
 * How the user talks to the agent.
 *
 * - `push_to_talk`: manual turns driven by the push-to-talk button; the agent runs with
 *   manual turn detection.
 * - `auto`: the agent uses automatic turn detection and the microphone stays live.
 *
 * The selected mode is sent to the agent as dispatch metadata (`{ input_mode }`) so it can
 * configure turn detection to match, and it gates the push-to-talk UI.
 */
export const DEFAULT_INPUT_MODE: InputMode = 'push_to_talk';
