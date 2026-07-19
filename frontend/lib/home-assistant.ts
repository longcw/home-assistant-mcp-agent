/**
 * Shared types and pure helpers for the Home Assistant agent UI.
 *
 * The agent (agent/agent.py) publishes the tool-execution lifecycle on a single
 * data-channel topic. Most tools render as a generic tool card; the two state tools
 * (`get_environment_info` / `get_devices`) return YAML, so the same event also powers
 * the rich device/sensor status cards — no second channel required.
 */
import { load as loadYaml } from 'js-yaml';

/** Must match `TOOL_CALL_TOPIC` in agent/agent.py. */
export const TOOL_CALL_TOPIC = 'ha.tool_call';

export type ToolStatus = 'running' | 'done' | 'error' | 'cancelled';

export interface HomeEntity {
  name: string;
  state: string;
  area?: string;
  domain?: string;
  unit?: string;
  deviceClass?: string;
}

export interface HomeStateSnapshot {
  /** `environment` groups sensors; `devices` groups the entities in an area. */
  kind: 'environment' | 'devices';
  title: string;
  entities: HomeEntity[];
}

export interface ToolCall {
  callId: string;
  name: string;
  /** Parsed tool arguments, or the raw string if it was not valid JSON. */
  args: Record<string, unknown> | string | null;
  status: ToolStatus;
  /** Result or error text voiced by the tool; may be trimmed by the agent. */
  output?: string | null;
  startedAt: number;
  endedAt?: number;
  /** Present for `get_devices` / `get_environment_info` — powers the status cards. */
  homeState?: HomeStateSnapshot;
}

function asString(value: unknown): string {
  if (value === null || value === undefined) return '';
  // `.nan` in the agent's YAML (a missing column) decodes to NaN — treat it as absent.
  if (typeof value === 'number' && Number.isNaN(value)) return '';
  return String(value);
}

/** Turn a raw Home Assistant row into a normalized entity for the UI. */
export function normalizeEntity(raw: Record<string, unknown>): HomeEntity {
  const attrs =
    raw.attributes && typeof raw.attributes === 'object'
      ? (raw.attributes as Record<string, unknown>)
      : {};
  return {
    name: asString(raw.names) || asString(raw.name) || asString(attrs.friendly_name) || 'Unknown',
    state: asString(raw.state),
    area: asString(raw.areas) || asString(raw.area) || undefined,
    domain: asString(raw.domain) || undefined,
    unit: asString(attrs.unit_of_measurement) || undefined,
    deviceClass: asString(attrs.device_class) || undefined,
  };
}

function isNumeric(value: string): boolean {
  return value.trim() !== '' && !Number.isNaN(Number(value));
}

function joinUnit(value: string, unit: string): string {
  const tight = unit === '%' || unit.startsWith('°');
  return `${value}${tight ? '' : ' '}${unit}`;
}

/** Human-facing rendering of an entity's state (e.g. `23.4°C`, `On`, `Idle`). */
export function formatState(entity: HomeEntity): string {
  const raw = (entity.state ?? '').toString();
  const low = raw.toLowerCase();
  if (low === '' || low === 'unknown') return 'Unknown';
  if (low === 'unavailable') return 'Unavailable';
  if (low === 'on') return 'On';
  if (low === 'off') return 'Off';
  if (isNumeric(raw) && entity.unit) return joinUnit(raw, entity.unit);
  if (isNumeric(raw)) return raw;
  return raw.charAt(0).toUpperCase() + raw.slice(1);
}

/** Whether an entity reads as "active" (drives the accent/glow treatment). */
export function isActiveState(entity: HomeEntity): boolean {
  const low = (entity.state ?? '').toString().toLowerCase();
  if (['off', 'unavailable', 'unknown', 'idle', 'closed', 'locked', '', 'standby'].includes(low)) {
    return false;
  }
  if (low === 'on' || low === 'open' || low === 'unlocked' || low === 'playing') return true;
  // numeric sensors are not "on/off"; treat as inactive for glow purposes
  return !isNumeric(low);
}

// Device-list ordering: things you control first, ambient sensors last.
const DOMAIN_RANK: Record<string, number> = {
  light: 0,
  climate: 1,
  media_player: 2,
  fan: 3,
  cover: 4,
  lock: 5,
  switch: 6,
  vacuum: 7,
  camera: 8,
  scene: 9,
  script: 10,
  automation: 10,
  person: 11,
  binary_sensor: 20,
  sensor: 21,
};

// Environment ordering: the readings people usually ask about first.
const DEVICE_CLASS_RANK: Record<string, number> = {
  temperature: 0,
  humidity: 1,
  carbon_dioxide: 2,
  carbon_monoxide: 2,
  aqi: 3,
  pm25: 3,
  pm10: 3,
  illuminance: 4,
  pressure: 5,
  power: 6,
  energy: 7,
  battery: 8,
};

function relevanceRank(entity: HomeEntity, kind: HomeStateSnapshot['kind']): number {
  if (kind === 'environment') {
    return entity.deviceClass ? (DEVICE_CLASS_RANK[entity.deviceClass] ?? 50) : 50;
  }
  return entity.domain ? (DOMAIN_RANK[entity.domain] ?? 15) : 15;
}

/**
 * Order entities so the most relevant surface first: controllable devices before
 * ambient sensors (device lists), or the common readings first (environment). Active
 * devices win ties so what's currently on floats up. Stable within a rank.
 */
export function sortEntitiesByRelevance(
  entities: HomeEntity[],
  kind: HomeStateSnapshot['kind']
): HomeEntity[] {
  return entities
    .map((entity, index) => ({ entity, index }))
    .sort((a, b) => {
      const byRank = relevanceRank(a.entity, kind) - relevanceRank(b.entity, kind);
      if (byRank !== 0) return byRank;
      const byActive = Number(isActiveState(b.entity)) - Number(isActiveState(a.entity));
      if (byActive !== 0) return byActive;
      return a.index - b.index;
    })
    .map((x) => x.entity);
}

/**
 * Does this tool change the home (vs. just reading it)? Action tools get the
 * accent ripple animation when they complete.
 */
export function isActionTool(name: string): boolean {
  if (/^(get|list)/i.test(name)) return false;
  if (/livecontext|status|context|areas|domains|devices|info/i.test(name)) return false;
  return /(turn|set|toggle|open|close|lock|unlock|start|stop|play|pause|activate|press|select|increase|decrease|cancel|dim|brighten|boost)/i.test(
    name
  );
}

/** `HassTurnOn` → `Turn on`, `get_environment_info` → `Get environment info`. */
export function humanizeToolName(name: string): string {
  const spaced = name
    .replace(/^Hass/, '')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .trim();
  if (!spaced) return name;
  return spaced.charAt(0).toUpperCase() + spaced.slice(1).toLowerCase();
}

/** Flatten tool args into short `key: value` chips for the card header. */
export function argChips(args: ToolCall['args']): { key: string; value: string }[] {
  if (!args || typeof args === 'string') {
    return args ? [{ key: '', value: args }] : [];
  }
  return Object.entries(args).map(([key, value]) => ({
    key,
    value: typeof value === 'object' ? JSON.stringify(value) : String(value),
  }));
}

/** Tools whose JSON output is rendered as a rich status card instead of a tool card. */
const STATUS_TOOLS: Record<string, HomeStateSnapshot['kind']> = {
  get_environment_info: 'environment',
  get_devices: 'devices',
};

export function isStatusTool(name: string): boolean {
  return name in STATUS_TOOLS;
}

function titleFromArgs(args: ToolCall['args']): string | undefined {
  if (!args || typeof args === 'string') return undefined;
  const area = (args as Record<string, unknown>).area;
  if (Array.isArray(area)) return area.map(String).join(' · ');
  if (typeof area === 'string' && area.trim()) return area;
  return undefined;
}

/**
 * Build a status-card snapshot from a state tool's YAML output. Returns null while the
 * tool is still running, or when the output is not the expected YAML list (e.g. the
 * "no devices found" fallback string) — the caller then shows the plain tool card.
 */
export function parseHomeState(tool: ToolCall): HomeStateSnapshot | null {
  const kind = STATUS_TOOLS[tool.name];
  if (!kind || !tool.output) return null;

  let rows: unknown;
  try {
    rows = loadYaml(tool.output);
  } catch {
    return null;
  }
  if (!Array.isArray(rows) || rows.length === 0) return null;

  const entities = rows.map((row) => normalizeEntity(row as Record<string, unknown>));
  const title = kind === 'environment' ? 'Environment' : (titleFromArgs(tool.args) ?? 'Devices');
  return { kind, title, entities };
}
