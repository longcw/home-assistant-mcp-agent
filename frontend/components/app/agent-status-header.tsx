'use client';

import { Track } from 'livekit-client';
import { AnimatePresence, motion } from 'motion/react';
import { VideoTrack, useTracks, useVoiceAssistant } from '@livekit/components-react';
import { AgentAudioVisualizerBar } from '@/components/agents-ui/agent-audio-visualizer-bar';
import { useLocalTrackRef } from '@/components/agents-ui/blocks/agent-session-view-01/components/tile-view';
import { cn } from '@/lib/shadcn/utils';

const HA_ACCENT = '#03a9f4';

const STATE_LABELS: Record<string, string> = {
  connecting: 'Connecting…',
  initializing: 'Waking up…',
  listening: 'Listening',
  thinking: 'Thinking…',
  speaking: 'Speaking',
};

export function AgentStatusHeader({ color = HA_ACCENT }: { color?: `#${string}` | string }) {
  const { state, audioTrack } = useVoiceAssistant();
  const cameraTrack = useLocalTrackRef(Track.Source.Camera);
  const [screenShareTrack] = useTracks([Track.Source.ScreenShare]);

  const label = STATE_LABELS[state ?? ''] ?? 'Ready';
  const active = state === 'listening' || state === 'thinking' || state === 'speaking';
  const cameraOn = Boolean(cameraTrack) && !cameraTrack?.publication.isMuted;
  const screenOn = Boolean(screenShareTrack?.publication) && !screenShareTrack.publication.isMuted;

  return (
    <header className="border-border/60 bg-background/70 relative z-20 flex items-center gap-3 border-b px-4 py-2.5 backdrop-blur-md">
      <div
        className="flex size-9 shrink-0 items-center justify-center rounded-full"
        style={{ backgroundColor: `${color}14` }}
      >
        <AgentAudioVisualizerBar
          size="icon"
          barCount={5}
          state={state}
          audioTrack={audioTrack}
          color={color as `#${string}`}
        />
      </div>

      <div className="min-w-0">
        <p className="text-foreground text-sm leading-tight font-semibold">Home Assistant</p>
        <div className="flex items-center gap-1.5">
          <motion.span
            className="size-1.5 rounded-full"
            style={{ backgroundColor: active ? color : 'var(--muted-foreground)' }}
            animate={active ? { opacity: [1, 0.35, 1] } : { opacity: 0.5 }}
            transition={active ? { duration: 1.6, repeat: Infinity, ease: 'easeInOut' } : {}}
          />
          <p className="text-muted-foreground text-xs leading-tight">{label}</p>
        </div>
      </div>

      <AnimatePresence>
        {(cameraOn || screenOn) && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="ml-auto flex items-center gap-2"
          >
            {cameraOn && cameraTrack && (
              <VideoTrack
                trackRef={cameraTrack}
                className={cn('bg-muted aspect-video h-9 rounded-md object-cover')}
              />
            )}
            {screenOn && (
              <VideoTrack
                trackRef={screenShareTrack}
                className={cn('bg-muted aspect-video h-9 rounded-md object-cover')}
              />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
