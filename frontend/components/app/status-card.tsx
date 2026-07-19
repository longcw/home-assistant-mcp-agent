'use client';

import { useMemo, useState } from 'react';
import { ChevronDown, LayoutGrid } from 'lucide-react';
import { motion } from 'motion/react';
import {
  type HomeEntity,
  type HomeStateSnapshot,
  formatState,
  isActiveState,
  sortEntitiesByRelevance,
} from '@/lib/home-assistant';
import { cn } from '@/lib/shadcn/utils';
import { entityIcon } from './ha-icons';

const HA_ACCENT = '#03a9f4';

// How many tiles to show before collapsing behind "Show more" (2–3 grid rows).
const COLLAPSED_COUNT = 6;

const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.05 } },
};

const item = {
  hidden: { opacity: 0, y: 8, scale: 0.98 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: { type: 'spring', stiffness: 400, damping: 30 },
  },
} as const;

type Tone = 'metric' | 'warm' | 'active' | 'idle';

function toneFor(entity: HomeEntity): Tone {
  const raw = (entity.state ?? '').toString().trim();
  const isNumeric = raw !== '' && !Number.isNaN(Number(raw));
  if (isNumeric) return 'metric';
  if (!isActiveState(entity)) return 'idle';
  return entity.domain === 'light' ? 'warm' : 'active';
}

/** Split a state into a large value + a smaller trailing unit for the readout. */
function readout(entity: HomeEntity): { value: string; unit?: string } {
  const raw = (entity.state ?? '').toString().trim();
  const num = Number(raw);
  if (raw !== '' && !Number.isNaN(num)) {
    const rounded = Number.isInteger(num) ? num : Math.round(num * 10) / 10;
    return { value: String(rounded), unit: entity.unit };
  }
  return { value: formatState(entity) };
}

function EntityTile({ entity }: { entity: HomeEntity }) {
  const Icon = entityIcon(entity);
  const tone = toneFor(entity);
  const { value, unit } = readout(entity);

  const warm = tone === 'warm';
  const active = tone === 'active';
  const glow = warm || active;
  const accent = warm ? '#f59e0b' : HA_ACCENT;

  return (
    <motion.div
      variants={item}
      className={cn(
        'relative flex flex-col gap-3 overflow-hidden rounded-xl border p-3 transition-colors',
        'bg-muted/40 dark:bg-muted/20',
        glow ? 'border-transparent' : 'border-border/60'
      )}
      style={
        glow ? { boxShadow: `inset 0 0 0 1px ${accent}55, 0 0 22px -12px ${accent}` } : undefined
      }
    >
      {/* soft top glow when a device is on */}
      {glow && (
        <div
          aria-hidden
          className="pointer-events-none absolute -top-8 -right-6 size-20 rounded-full opacity-40 blur-2xl"
          style={{ backgroundColor: accent }}
        />
      )}
      <div className="flex items-start justify-between">
        <span
          className={cn(
            'flex size-8 items-center justify-center rounded-lg',
            glow ? 'text-white' : 'text-muted-foreground bg-background/80'
          )}
          style={glow ? { backgroundColor: accent } : undefined}
        >
          <Icon className="size-4" />
        </span>
        {(warm || active) && (
          <motion.span
            layout
            className="size-2 rounded-full"
            style={{ backgroundColor: accent }}
            animate={{ opacity: [1, 0.4, 1] }}
            transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          />
        )}
      </div>

      <div className="min-w-0">
        <div className="flex items-baseline gap-1 font-mono">
          <span className="text-foreground truncate text-2xl leading-none font-semibold tabular-nums">
            {value}
          </span>
          {unit && <span className="text-muted-foreground text-sm">{unit}</span>}
        </div>
        <p className="text-muted-foreground mt-1.5 truncate text-xs" title={entity.name}>
          {entity.name}
        </p>
      </div>
    </motion.div>
  );
}

export function StatusCard({ snapshot }: { snapshot: HomeStateSnapshot }) {
  const [expanded, setExpanded] = useState(false);
  const label = snapshot.kind === 'environment' ? 'Environment' : snapshot.title;

  const sorted = useMemo(
    () => sortEntitiesByRelevance(snapshot.entities, snapshot.kind),
    [snapshot.entities, snapshot.kind]
  );
  const collapsible = sorted.length > COLLAPSED_COUNT;
  const visible = expanded || !collapsible ? sorted : sorted.slice(0, COLLAPSED_COUNT);
  const hiddenCount = sorted.length - COLLAPSED_COUNT;

  return (
    <div className="bg-card w-full rounded-2xl border p-3 shadow-sm">
      <div className="mb-2.5 flex items-center gap-2 px-1">
        <LayoutGrid className="size-3.5" style={{ color: HA_ACCENT }} />
        <span className="text-foreground text-sm font-semibold">{label}</span>
        <span className="text-muted-foreground ml-auto text-xs tabular-nums">
          {sorted.length} {sorted.length === 1 ? 'entity' : 'entities'}
        </span>
      </div>
      <motion.div
        layout
        variants={container}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-2 gap-2 sm:grid-cols-3"
      >
        {visible.map((entity, i) => (
          <EntityTile key={`${entity.name}-${i}`} entity={entity} />
        ))}
      </motion.div>

      {collapsible && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="text-muted-foreground hover:text-foreground hover:bg-muted/50 mt-2 flex w-full items-center justify-center gap-1 rounded-lg py-1.5 text-xs font-medium transition-colors"
        >
          {expanded ? 'Show less' : `Show ${hiddenCount} more`}
          <ChevronDown className={cn('size-3.5 transition-transform', expanded && 'rotate-180')} />
        </button>
      )}
    </div>
  );
}
