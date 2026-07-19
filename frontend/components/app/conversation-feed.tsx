'use client';

import { useMemo } from 'react';
import { AnimatePresence } from 'motion/react';
import { type AgentState, type ReceivedMessage } from '@livekit/components-react';
import { AgentChatIndicator } from '@/components/agents-ui/agent-chat-indicator';
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import { Message, MessageContent, MessageResponse } from '@/components/ai-elements/message';
import { type ToolCall } from '@/lib/home-assistant';
import { cn } from '@/lib/shadcn/utils';
import { StatusCard } from './status-card';
import { ToolCallCard } from './tool-call-card';

type FeedItem =
  | { kind: 'message'; id: string; ts: number; role: 'user' | 'assistant'; text: string }
  | { kind: 'tool'; id: string; ts: number; tool: ToolCall };

interface ConversationFeedProps {
  messages: ReceivedMessage[];
  toolCalls: ToolCall[];
  agentState?: AgentState;
  emptyHint?: string;
  className?: string;
}

export function ConversationFeed({
  messages,
  toolCalls,
  agentState,
  emptyHint = 'Connected. Ask about your home — try “what’s the temperature?”',
  className,
}: ConversationFeedProps) {
  const items = useMemo<FeedItem[]>(() => {
    const msgItems: FeedItem[] = messages.map((m) => ({
      kind: 'message',
      id: m.id,
      ts: typeof m.timestamp === 'number' ? m.timestamp : new Date(m.timestamp).getTime(),
      role: m.from?.isLocal ? 'user' : 'assistant',
      text: m.message,
    }));
    const toolItems: FeedItem[] = toolCalls.map((t) => ({
      kind: 'tool',
      id: t.callId,
      ts: t.startedAt,
      tool: t,
    }));
    return [...msgItems, ...toolItems].sort((a, b) => a.ts - b.ts);
  }, [messages, toolCalls]);

  const isEmpty = items.length === 0;

  return (
    <Conversation className={cn('h-full', className)}>
      <ConversationContent className="mx-auto w-full max-w-2xl gap-4 px-4 pt-6 pb-4">
        {isEmpty && (
          <div className="text-muted-foreground flex h-full min-h-40 items-center justify-center text-center text-sm text-balance">
            {emptyHint}
          </div>
        )}

        {items.map((item) =>
          item.kind === 'message' ? (
            <Message key={item.id} from={item.role}>
              <MessageContent>
                <MessageResponse>{item.text}</MessageResponse>
              </MessageContent>
            </Message>
          ) : (
            <div key={item.id} className="is-assistant flex w-full justify-start">
              {item.tool.homeState ? (
                <StatusCard snapshot={item.tool.homeState} />
              ) : (
                <ToolCallCard tool={item.tool} />
              )}
            </div>
          )
        )}

        <AnimatePresence>
          {agentState === 'thinking' && <AgentChatIndicator size="sm" />}
        </AnimatePresence>
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
}
