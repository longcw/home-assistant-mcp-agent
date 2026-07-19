import { TokenSourceConfigurable, type TokenSourceFetchOptions } from 'livekit-client';
import { RoomAgentDispatch, RoomConfiguration } from '@livekit/protocol';

/**
 * A cache-free replacement for `TokenSource.endpoint(...)`.
 *
 * livekit-client 2.17.2's built-in `TokenSourceCached` has an inverted cache check
 * (`shouldReturnCachedValueFromFetch` returns the cached token when the fetch options
 * *differ* from the cached ones instead of when they match). Because `useSession`
 * pre-warms a token on mount via `prepareConnection` — using the default options —
 * switching the turn mode afterwards never reaches the agent: `start()` is served the
 * stale mount-time token, so the agent is always dispatched in the default mode.
 *
 * Fetching a fresh token on every call sidesteps the bug. The extra request only happens
 * on connect, so the cost is negligible. Posts the same `{ room_config }` body shape the
 * `/api/token` route already expects.
 */
export class EndpointTokenSource extends TokenSourceConfigurable {
  constructor(private readonly url: string) {
    super();
  }

  async fetch(options: TokenSourceFetchOptions) {
    const roomConfig = new RoomConfiguration({
      agents: options.agentName
        ? [
            new RoomAgentDispatch({
              agentName: options.agentName,
              metadata: options.agentMetadata ?? '',
            }),
          ]
        : [],
    });

    const res = await fetch(this.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ room_config: roomConfig.toJson({ useProtoFieldName: true }) }),
    });
    if (!res.ok) {
      throw new Error(`Token endpoint ${this.url} responded ${res.status}`);
    }
    return res.json();
  }
}
