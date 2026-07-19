import {
  Activity,
  Blinds,
  Bot,
  CircleDot,
  Clapperboard,
  Clock,
  CloudSun,
  DoorOpen,
  Droplets,
  Fan,
  Gauge,
  Lightbulb,
  Lock,
  type LucideIcon,
  Play,
  Plug,
  Power,
  Radar,
  Search,
  SlidersHorizontal,
  Speaker,
  Sun,
  Thermometer,
  ToggleRight,
  User,
  Video,
  Wifi,
  Wind,
  Zap,
} from 'lucide-react';
import { type HomeEntity, isActionTool } from '@/lib/home-assistant';

const DOMAIN_ICONS: Record<string, LucideIcon> = {
  light: Lightbulb,
  switch: ToggleRight,
  outlet: Plug,
  fan: Fan,
  climate: Thermometer,
  cover: Blinds,
  lock: Lock,
  media_player: Speaker,
  sensor: Gauge,
  binary_sensor: Activity,
  camera: Video,
  person: User,
  weather: CloudSun,
  vacuum: Bot,
  scene: Clapperboard,
  automation: Play,
  script: Play,
};

const DEVICE_CLASS_ICONS: Record<string, LucideIcon> = {
  temperature: Thermometer,
  humidity: Droplets,
  battery: Zap,
  power: Zap,
  energy: Zap,
  current: Zap,
  voltage: Zap,
  illuminance: Sun,
  pressure: Gauge,
  carbon_dioxide: Wind,
  carbon_monoxide: Wind,
  gas: Wind,
  aqi: Wind,
  pm25: Wind,
  motion: Radar,
  occupancy: Radar,
  presence: Radar,
  door: DoorOpen,
  window: DoorOpen,
  opening: DoorOpen,
  garage_door: DoorOpen,
  timestamp: Clock,
  signal_strength: Wifi,
};

/** Best icon for an entity, preferring its device class, then its domain. */
export function entityIcon(entity: HomeEntity): LucideIcon {
  if (entity.deviceClass && DEVICE_CLASS_ICONS[entity.deviceClass]) {
    return DEVICE_CLASS_ICONS[entity.deviceClass];
  }
  if (entity.domain && DOMAIN_ICONS[entity.domain]) {
    return DOMAIN_ICONS[entity.domain];
  }
  return CircleDot;
}

/** Icon for a tool card, reflecting what the tool does. */
export function toolIcon(name: string): LucideIcon {
  if (/turn(on|off)?/i.test(name)) return Power;
  if (/toggle/i.test(name)) return ToggleRight;
  if (/light|set|dim|brighten|volume|temperature/i.test(name)) return SlidersHorizontal;
  if (isActionTool(name)) return Zap;
  if (/environment|sensor/i.test(name)) return Gauge;
  return Search;
}
