'use client';

import React, { useEffect, useState } from 'react';
import { useAgent, useSessionContext, useSessionMessages } from '@livekit/components-react';
import {
  AgentControlBar,
  type AgentControlBarControls,
} from '@/components/agents-ui/agent-control-bar';
import { AgentStatusHeader } from '@/components/app/agent-status-header';
import { ConversationFeed } from '@/components/app/conversation-feed';
import { PushToTalkButton } from '@/components/app/push-to-talk-button';
import { useSessionSettings } from '@/components/app/session-settings';
import { useHomeAssistantFeed } from '@/hooks/use-home-assistant-feed';
import { cn } from '@/lib/shadcn/utils';

export interface AgentSessionView_01Props {
  /** Enables the chat toggle / text input in the control bar. @default true */
  supportsChatInput?: boolean;
  /** Enables the camera control in the control bar. @default true */
  supportsVideoInput?: boolean;
  /** Enables the screen-share control in the control bar. @default true */
  supportsScreenShare?: boolean;
  /** Accent color for the agent header visualizer. */
  audioVisualizerColor?: `#${string}`;
  /** Optional class name merged onto the outer `<section>`. */
  className?: string;
}

/**
 * The connected session view: a fixed agent header, a scrollable conversation feed
 * (chat + tool/status cards), and a docked control bar. The three regions are flex
 * children, so the controls can never overlap the transcript regardless of whether the
 * push-to-talk button is shown.
 */
export function AgentSessionView_01({
  supportsChatInput = true,
  supportsVideoInput = true,
  supportsScreenShare = true,
  audioVisualizerColor,
  ref,
  className,
  ...props
}: React.ComponentProps<'section'> & AgentSessionView_01Props) {
  const session = useSessionContext();
  const { messages } = useSessionMessages(session);
  const { toolCalls } = useHomeAssistantFeed();
  const { state: agentState } = useAgent();
  const { inputMode } = useSessionSettings();
  const [chatInputOpen, setChatInputOpen] = useState(false);

  // Set the initial microphone state for the chosen input mode on connect:
  //  - auto: mic stays live so the agent's turn detection can hear continuously
  //  - push-to-talk: mic starts muted; the PTT button enables it per press
  const localParticipant = session.room?.localParticipant;
  useEffect(() => {
    if (!session.isConnected || !localParticipant) return;
    localParticipant.setMicrophoneEnabled(inputMode === 'auto');
  }, [inputMode, session.isConnected, localParticipant]);

  const controls: AgentControlBarControls = {
    leave: true,
    // in push-to-talk mode the PTT button owns the mic; in auto mode show the mic toggle
    microphone: inputMode !== 'push_to_talk',
    chat: supportsChatInput,
    camera: supportsVideoInput,
    screenShare: supportsScreenShare,
  };

  return (
    <section
      ref={ref}
      className={cn(
        'bg-background relative z-10 flex h-full w-full flex-col overflow-hidden',
        className
      )}
      {...props}
    >
      <AgentStatusHeader color={audioVisualizerColor} />

      {/* Conversation feed — a flex child that scrolls internally and can never be
          occluded by the controls below. */}
      <div className="relative min-h-0 flex-1">
        <ConversationFeed messages={messages} toolCalls={toolCalls} agentState={agentState} />
      </div>

      {/* Controls */}
      <div className="mx-auto w-full max-w-2xl shrink-0 px-3 pb-3 md:px-0 md:pb-6">
        <PushToTalkButton />
        <AgentControlBar
          variant="livekit"
          controls={controls}
          isChatOpen={chatInputOpen}
          isConnected={session.isConnected}
          onDisconnect={session.end}
          onIsChatOpenChange={setChatInputOpen}
        />
      </div>
    </section>
  );
}
