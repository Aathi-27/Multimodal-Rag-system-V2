import { useCallback, useEffect, useRef } from 'react';
import { API_BASE_URL, AUTH_ENABLED } from '../api/axios';
import type { SSEEvent } from '../types/api.types';

interface UseSSEOptions {
  /** Called for every parsed SSE event */
  onEvent: (event: SSEEvent) => void;
  /** Called when the stream ends normally */
  onDone?: () => void;
  /** Called on network / parse error */
  onError?: (error: string) => void;
}

/**
 * Hook that opens a fetch-based SSE connection to POST /query.
 * Returns `start(body)` and `cancel()`.
 *
 * We use `fetch` (not EventSource) because:
 *   1. EventSource only supports GET
 *   2. We need to POST a JSON body
 */
export function useSSE({ onEvent, onDone, onError }: UseSSEOptions) {
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(
    async (body: Record<string, unknown>) => {
      // Abort any previous stream
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      };

      if (AUTH_ENABLED) {
        const token = localStorage.getItem('rag_token');
        if (token) headers['Authorization'] = `Bearer ${token}`;
      }

      try {
        const res = await fetch(`${API_BASE_URL}/query`, {
          method: 'POST',
          headers,
          body: JSON.stringify(body),
          signal: controller.signal,
        });

        if (!res.ok) {
          const text = await res.text();
          onError?.(`HTTP ${res.status}: ${text}`);
          return;
        }

        const reader = res.body?.getReader();
        if (!reader) {
          onError?.('No response body');
          return;
        }

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // SSE lines are separated by double newlines
          const parts = buffer.split('\n\n');
          // Keep the last (possibly incomplete) chunk
          buffer = parts.pop() || '';

          for (const part of parts) {
            const line = part.trim();
            if (!line.startsWith('data:')) continue;
            const jsonStr = line.slice(5).trim();
            if (!jsonStr) continue;

            try {
              const event: SSEEvent = JSON.parse(jsonStr);
              onEvent(event);

              if (event.type === 'done') {
                onDone?.();
                return;
              }
              if (event.type === 'error') {
                onError?.(event.content || 'Stream error');
                return;
              }
            } catch {
              // Non-JSON line — skip
            }
          }
        }

        // Stream ended without explicit "done"
        onDone?.();
      } catch (err: unknown) {
        if ((err as DOMException)?.name === 'AbortError') return;
        onError?.(err instanceof Error ? err.message : 'Stream failed');
      }
    },
    [onEvent, onDone, onError],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
  }, []);

  /* Abort on unmount — prevents leaked connections */
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  return { start, cancel };
}
