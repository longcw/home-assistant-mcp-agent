'use client';

import {
  type MouseEvent as ReactMouseEvent,
  type TouchEvent as ReactTouchEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import { MicIcon } from 'lucide-react';
import { motion } from 'motion/react';
import { useAgent, useSessionContext } from '@livekit/components-react';
import { useSessionSettings } from '@/components/app/session-settings';
import { cn } from '@/lib/shadcn/utils';

/**
 * A press-and-hold "push to talk" button.
 *
 * The agent runs with manual turn detection and advertises `push-to-talk: "1"` on its
 * participant attributes. While the button is held, the microphone is enabled and the
 * agent listens; on release the turn is committed (or cancelled if released off-button).
 *
 * It drives the agent through three RPC methods registered in agent.py:
 *  - `start_turn`  — interrupt, clear, start listening
 *  - `end_turn`    — stop listening and generate a reply
 *  - `cancel_turn` — stop listening and discard the turn
 */
export function PushToTalkButton({ className }: { className?: string }) {
  const session = useSessionContext();
  const agent = useAgent();
  const { inputMode } = useSessionSettings();

  const room = session.room;
  const localParticipant = room?.localParticipant;
  const agentIdentity = agent.identity;

  const [isPressed, setIsPressed] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const lastActionTime = useRef(0);
  const rpcTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  // keep the mic off until the user presses the button
  useEffect(() => {
    if (session.isConnected && localParticipant) {
      localParticipant.setMicrophoneEnabled(false);
    }
  }, [session.isConnected, localParticipant]);

  const performRpcWithTimeout = useCallback(
    async (method: string, timeoutMs = 3000): Promise<boolean> => {
      if (!localParticipant || !agentIdentity) return false;

      let succeeded = false;
      setIsLoading(true);
      try {
        const timeoutPromise = new Promise<void>((_, reject) => {
          rpcTimeoutRef.current = setTimeout(
            () => reject(new Error(`RPC timeout after ${timeoutMs}ms`)),
            timeoutMs
          );
        });
        await Promise.race([
          localParticipant
            .performRpc({ destinationIdentity: agentIdentity, method, payload: '' })
            .then(() => {
              succeeded = true;
            }),
          timeoutPromise,
        ]);
      } catch (error) {
        console.error(`RPC ${method} failed:`, error);
      } finally {
        if (rpcTimeoutRef.current) {
          clearTimeout(rpcTimeoutRef.current);
          rpcTimeoutRef.current = null;
        }
        setIsLoading(false);
      }
      return succeeded;
    },
    [localParticipant, agentIdentity]
  );

  const startTurn = useCallback(async () => {
    if (!localParticipant || !agentIdentity || isLoading) return;
    try {
      await localParticipant.setMicrophoneEnabled(true);
      const ok = await performRpcWithTimeout('start_turn');
      if (ok) {
        setIsPressed(true);
        setIsCancelling(false);
      } else {
        await localParticipant.setMicrophoneEnabled(false);
      }
    } catch (error) {
      console.error('Failed to start turn:', error);
    }
  }, [localParticipant, agentIdentity, isLoading, performRpcWithTimeout]);

  const endTurn = useCallback(
    async (method: 'end_turn' | 'cancel_turn') => {
      if (!localParticipant || !isPressed) return;
      const now = Date.now();
      if (now - lastActionTime.current < 250) return;
      lastActionTime.current = now;
      try {
        await localParticipant.setMicrophoneEnabled(false);
        await performRpcWithTimeout(method);
      } finally {
        setIsPressed(false);
        setIsCancelling(false);
      }
    },
    [localParticipant, isPressed, performRpcWithTimeout]
  );

  // --- mouse ---
  const handleMouseDown = useCallback(
    (e: ReactMouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      startTurn();
    },
    [startTurn]
  );
  const handleMouseLeave = useCallback(() => {
    if (isPressed) setIsCancelling(true);
  }, [isPressed]);
  const handleMouseEnter = useCallback(() => {
    if (isPressed) setIsCancelling(false);
  }, [isPressed]);

  // --- touch ---
  const isTouchOutside = useCallback((x: number, y: number): boolean => {
    const rect = buttonRef.current?.getBoundingClientRect();
    if (!rect) return false;
    return x < rect.left || x > rect.right || y < rect.top || y > rect.bottom;
  }, []);
  const handleTouchStart = useCallback(
    (e: ReactTouchEvent<HTMLButtonElement>) => {
      e.preventDefault();
      startTurn();
    },
    [startTurn]
  );
  const handleTouchMove = useCallback(
    (e: ReactTouchEvent) => {
      if (!isPressed) return;
      const touch = e.touches[0];
      setIsCancelling(isTouchOutside(touch.clientX, touch.clientY));
    },
    [isPressed, isTouchOutside]
  );

  // release (mouse up / touch end) anywhere in the window ends the turn
  useEffect(() => {
    if (!isPressed) return;
    const end = () => endTurn(isCancelling ? 'cancel_turn' : 'end_turn');
    window.addEventListener('mouseup', end, { once: true });
    window.addEventListener('touchend', end, { once: true });
    return () => {
      window.removeEventListener('mouseup', end);
      window.removeEventListener('touchend', end);
    };
  }, [isPressed, isCancelling, endTurn]);

  // cleanup on unmount
  useEffect(() => {
    return () => {
      if (rpcTimeoutRef.current) clearTimeout(rpcTimeoutRef.current);
    };
  }, []);

  // prevent the long-press context menu on touch devices
  useEffect(() => {
    const el = buttonRef.current;
    if (!el) return;
    const prevent = (e: Event) => e.preventDefault();
    el.addEventListener('contextmenu', prevent);
    return () => el.removeEventListener('contextmenu', prevent);
  }, []);

  if (inputMode !== 'push_to_talk' || !session.isConnected || !agentIdentity) return null;

  const label = isLoading
    ? 'Processing…'
    : isPressed
      ? isCancelling
        ? 'Release to cancel'
        : 'Listening…'
      : 'Push to talk';

  return (
    <motion.button
      ref={buttonRef}
      type="button"
      aria-label="Push to talk"
      disabled={isLoading}
      onMouseDown={handleMouseDown}
      onMouseLeave={handleMouseLeave}
      onMouseEnter={handleMouseEnter}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      whileTap={{ scale: 0.97 }}
      className={cn(
        'mx-auto mb-3 flex h-12 min-w-52 touch-none items-center justify-center gap-2 rounded-full px-8',
        'font-medium text-white shadow-md transition-colors select-none',
        'focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:outline-none',
        isLoading
          ? 'bg-muted-foreground/60 cursor-not-allowed'
          : isPressed
            ? isCancelling
              ? 'bg-destructive'
              : 'bg-blue-600'
            : 'bg-primary hover:bg-primary/90',
        className
      )}
    >
      <MicIcon className="size-5" />
      {label}
    </motion.button>
  );
}
