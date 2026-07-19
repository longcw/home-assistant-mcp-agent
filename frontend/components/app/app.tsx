'use client';

import { useMemo, useState } from 'react';
import { useSession } from '@livekit/components-react';
import { WarningIcon } from '@phosphor-icons/react/dist/ssr';
import type { AppConfig } from '@/app-config';
import { AgentSessionProvider } from '@/components/agents-ui/agent-session-provider';
import { StartAudioButton } from '@/components/agents-ui/start-audio-button';
import { SessionSettingsProvider } from '@/components/app/session-settings';
import { ViewController } from '@/components/app/view-controller';
import { Toaster } from '@/components/ui/sonner';
import { useAgentErrors } from '@/hooks/useAgentErrors';
import { useDebugMode } from '@/hooks/useDebug';
import { EndpointTokenSource } from '@/lib/endpoint-token-source';
import { DEFAULT_INPUT_MODE, type InputMode } from '@/lib/input-mode';
import { getSandboxTokenSource } from '@/lib/utils';

const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

function AppSetup() {
  useDebugMode({ enabled: IN_DEVELOPMENT });
  useAgentErrors();

  return null;
}

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  const [agentName, setAgentName] = useState(appConfig.agentName ?? 'ha-agent');
  const [inputMode, setInputMode] = useState<InputMode>(DEFAULT_INPUT_MODE);

  const tokenSource = useMemo(() => {
    // EndpointTokenSource fetches fresh each connect; see its doc comment for the
    // livekit-client caching bug that otherwise pins the agent to the mount-time turn mode.
    return typeof process.env.NEXT_PUBLIC_CONN_DETAILS_ENDPOINT === 'string'
      ? getSandboxTokenSource(appConfig)
      : new EndpointTokenSource('/api/token');
  }, [appConfig]);

  // read at connect time by useSession; updating before start() takes effect
  const sessionOptions = useMemo(
    () => ({
      // explicit dispatch of the configured agent worker
      agentName: agentName.trim() || undefined,
      // tell the agent which turn mode this client uses
      agentMetadata: JSON.stringify({ input_mode: inputMode }),
    }),
    [agentName, inputMode]
  );

  const session = useSession(tokenSource, sessionOptions);

  const settings = useMemo(
    () => ({ agentName, setAgentName, inputMode, setInputMode }),
    [agentName, inputMode]
  );

  return (
    <SessionSettingsProvider value={settings}>
      <AgentSessionProvider session={session}>
        <AppSetup />
        <main className="grid h-svh grid-cols-1 place-content-center">
          <ViewController appConfig={appConfig} />
        </main>
        <StartAudioButton label="Start Audio" />
        <Toaster
          icons={{
            warning: <WarningIcon weight="bold" />,
          }}
          position="top-center"
          className="toaster group"
          style={
            {
              '--normal-bg': 'var(--popover)',
              '--normal-text': 'var(--popover-foreground)',
              '--normal-border': 'var(--border)',
            } as React.CSSProperties
          }
        />
      </AgentSessionProvider>
    </SessionSettingsProvider>
  );
}
