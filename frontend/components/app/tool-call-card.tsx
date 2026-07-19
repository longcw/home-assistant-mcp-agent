'use client';

import { useState } from 'react';
import { Ban, Check, ChevronRight, Loader, TriangleAlert } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { type ToolCall, argChips, humanizeToolName, isActionTool } from '@/lib/home-assistant';
import { cn } from '@/lib/shadcn/utils';
import { toolIcon } from './ha-icons';

const HA_ACCENT = '#03a9f4';

function elapsedLabel(tool: ToolCall): string | null {
  if (!tool.endedAt) return null;
  const secs = Math.max(0, tool.endedAt - tool.startedAt) / 1000;
  return `${secs < 0.1 ? '<0.1' : secs.toFixed(1)}s`;
}

function StatusGlyph({ tool, isAction }: { tool: ToolCall; isAction: boolean }) {
  if (tool.status === 'running') {
    return <Loader className="size-4 animate-spin" style={{ color: HA_ACCENT }} />;
  }
  if (tool.status === 'error') {
    return <TriangleAlert className="text-destructive size-4" />;
  }
  if (tool.status === 'cancelled') {
    return <Ban className="text-muted-foreground size-4" />;
  }
  return (
    <Check className="size-4" style={{ color: isAction ? HA_ACCENT : 'var(--muted-foreground)' }} />
  );
}

export function ToolCallCard({ tool }: { tool: ToolCall }) {
  const [open, setOpen] = useState(false);
  const Icon = toolIcon(tool.name);
  const isAction = isActionTool(tool.name);
  const running = tool.status === 'running';
  const done = tool.status === 'done';
  const chips = argChips(tool.args);
  const elapsed = elapsedLabel(tool);
  const hasDetails = chips.length > 0 || Boolean(tool.output);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: 'spring', stiffness: 400, damping: 32 }}
      className={cn(
        'bg-card w-full max-w-md rounded-2xl border p-2.5 shadow-sm',
        running && 'border-transparent'
      )}
      style={
        running
          ? { boxShadow: `inset 0 0 0 1px ${HA_ACCENT}44, 0 0 18px -12px ${HA_ACCENT}` }
          : undefined
      }
    >
      <button
        type="button"
        disabled={!hasDetails}
        onClick={() => setOpen((v) => !v)}
        className={cn(
          'flex w-full items-center gap-3 rounded-lg text-left',
          hasDetails && 'cursor-pointer'
        )}
      >
        {/* icon badge + one-shot ripple when an action completes */}
        <span
          className={cn(
            'relative flex size-8 shrink-0 items-center justify-center rounded-lg',
            isAction && done ? 'text-white' : 'text-muted-foreground bg-muted'
          )}
          style={isAction && done ? { backgroundColor: HA_ACCENT } : undefined}
        >
          <Icon className="size-4" />
          {isAction && done && (
            <motion.span
              aria-hidden
              className="absolute inset-0 rounded-lg"
              style={{ border: `2px solid ${HA_ACCENT}` }}
              initial={{ opacity: 0.7, scale: 1 }}
              animate={{ opacity: 0, scale: 1.9 }}
              transition={{ duration: 0.7, ease: 'easeOut' }}
            />
          )}
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-foreground truncate text-sm font-medium">
              {humanizeToolName(tool.name)}
            </span>
            {isAction && (
              <span
                className="rounded-full px-1.5 py-0.5 text-[10px] font-semibold tracking-wide uppercase"
                style={{ color: HA_ACCENT, backgroundColor: `${HA_ACCENT}1a` }}
              >
                Action
              </span>
            )}
          </div>
          {/* arg chips */}
          {chips.length > 0 && (
            <div className="mt-1 flex flex-wrap gap-1">
              {chips.map((chip, i) => (
                <span
                  key={i}
                  className="bg-muted text-muted-foreground max-w-[16rem] truncate rounded-md px-1.5 py-0.5 font-mono text-[11px]"
                >
                  {chip.key ? `${chip.key}: ${chip.value}` : chip.value}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          {elapsed && (
            <span className="text-muted-foreground font-mono text-[11px] tabular-nums">
              {elapsed}
            </span>
          )}
          <StatusGlyph tool={tool} isAction={isAction} />
          {hasDetails && (
            <ChevronRight
              className={cn(
                'text-muted-foreground size-4 transition-transform',
                open && 'rotate-90'
              )}
            />
          )}
        </div>
      </button>

      <AnimatePresence initial={false}>
        {open && tool.output && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            className="overflow-hidden"
          >
            <pre className="text-muted-foreground bg-muted/60 mt-2.5 max-h-56 overflow-auto rounded-lg p-2.5 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
              {tool.output}
            </pre>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
