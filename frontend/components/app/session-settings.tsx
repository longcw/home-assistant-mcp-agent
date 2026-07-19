'use client';

import { createContext, useContext } from 'react';
import type { InputMode } from '@/lib/input-mode';

export interface SessionSettings {
  /** Explicit-dispatch agent name. */
  agentName: string;
  setAgentName: (name: string) => void;
  /** Turn mode requested from the agent. */
  inputMode: InputMode;
  setInputMode: (mode: InputMode) => void;
}

const SessionSettingsContext = createContext<SessionSettings | null>(null);

export const SessionSettingsProvider = SessionSettingsContext.Provider;

export function useSessionSettings(): SessionSettings {
  const ctx = useContext(SessionSettingsContext);
  if (!ctx) {
    throw new Error('useSessionSettings must be used within a SessionSettingsProvider');
  }
  return ctx;
}
