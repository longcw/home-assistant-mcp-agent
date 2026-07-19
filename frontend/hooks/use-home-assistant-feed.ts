'use client';

import { useEffect, useMemo, useState } from 'react';
import { RoomEvent } from 'livekit-client';
import { useSessionContext } from '@livekit/components-react';
import {
  TOOL_CALL_TOPIC,
  type ToolCall,
  type ToolStatus,
  parseHomeState,
} from '@/lib/home-assistant';

interface FeedState {
  tools: Record<string, ToolCall>;
}

const EMPTY: FeedState = { tools: {} };

/**
 * Subscribes to the agent's tool-execution data channel and exposes an ordered list of
 * tool calls. Each call carries its arguments, live status, and output; the two state
 * tools additionally expose a parsed `homeState` snapshot for the status cards.
 */
export function useHomeAssistantFeed(): { toolCalls: ToolCall[] } {
  const session = useSessionContext();
  const room = session.room;
  const [state, setState] = useState<FeedState>(EMPTY);

  useEffect(() => {
    if (!room) return;

    const decoder = new TextDecoder();

    const onData = (payload: Uint8Array, _p: unknown, _k: unknown, topic?: string) => {
      if (topic !== TOOL_CALL_TOPIC) return;

      let data: Record<string, unknown>;
      try {
        data = JSON.parse(decoder.decode(payload));
      } catch {
        return;
      }

      const update = data.update as Record<string, unknown> | undefined;
      if (!update) return;
      const at = typeof data.created_at === 'number' ? data.created_at * 1000 : Date.now();

      if (update.type === 'tool_call_started') {
        const fc = update.function_call as Record<string, unknown>;
        const callId = String(fc.call_id);
        let args: ToolCall['args'] = null;
        if (typeof fc.arguments === 'string' && fc.arguments) {
          try {
            args = JSON.parse(fc.arguments);
          } catch {
            args = fc.arguments;
          }
        }
        setState((s) => ({
          tools: {
            ...s.tools,
            [callId]: { callId, name: String(fc.name), args, status: 'running', startedAt: at },
          },
        }));
      } else if (update.type === 'tool_call_ended') {
        const callId = String(update.call_id);
        setState((s) => {
          const prev = s.tools[callId] ?? {
            callId,
            name: 'tool',
            args: null,
            status: 'running' as ToolStatus,
            startedAt: at,
          };
          return {
            tools: {
              ...s.tools,
              [callId]: {
                ...prev,
                status: update.status as ToolStatus,
                output: (update.message as string | null) ?? prev.output ?? null,
                endedAt: at,
              },
            },
          };
        });
      } else if (update.type === 'tool_call_updated') {
        const callId = String(update.call_id);
        setState((s) => {
          const prev = s.tools[callId];
          if (!prev) return s;
          return {
            tools: {
              ...s.tools,
              [callId]: { ...prev, output: (update.message as string) ?? prev.output },
            },
          };
        });
      }
    };

    room.on(RoomEvent.DataReceived, onData);
    return () => {
      room.off(RoomEvent.DataReceived, onData);
    };
  }, [room]);

  // clear the feed when a session ends so the next call starts fresh
  useEffect(() => {
    if (!session.isConnected) setState(EMPTY);
  }, [session.isConnected]);

  const toolCalls = useMemo(
    () =>
      Object.values(state.tools)
        .map((tool) => ({ ...tool, homeState: parseHomeState(tool) ?? undefined }))
        .sort((a, b) => a.startedAt - b.startedAt),
    [state]
  );

  return { toolCalls };
}
