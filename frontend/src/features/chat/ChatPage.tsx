import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChatMessage, Citation, DebugInfo, SSEEvent, ImageQueryResponse, AudioQueryResponse } from '@/shared/types/api.types';
import { useSSE } from '@/shared/hooks/useSSE';
import { Settings2, Bug } from 'lucide-react';
import ChatInput from './ChatInput';
import MessageList from './MessageList';
import RetrievalDebugPanel from './RetrievalDebugPanel';
import AdvancedSettingsPanel from './AdvancedSettingsPanel';

export default function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const assistantIdRef = useRef<string>('');
  const citationsRef = useRef<Citation[]>([]);

  /* ─── Debug mode ───────────────────────────────────────────────── */
  const [debugEnabled, setDebugEnabled] = useState(false);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [debugPanelOpen, setDebugPanelOpen] = useState(false);

  /* ─── Advanced settings modal ──────────────────────────────────── */
  const [settingsOpen, setSettingsOpen] = useState(false);

  /* ─── Token buffering (prevents per-token re-renders) ──────────── */
  const tokenBufferRef = useRef('');
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flushTokenBuffer = useCallback(() => {
    flushTimerRef.current = null;
    const buf = tokenBufferRef.current;
    if (!buf) return;
    tokenBufferRef.current = '';
    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantIdRef.current
          ? { ...m, content: m.content + buf }
          : m,
      ),
    );
  }, []);

  /** Buffer incoming tokens — flushed to state every 60ms */
  const bufferToken = useCallback(
    (token: string) => {
      tokenBufferRef.current += token;
      if (flushTimerRef.current === null) {
        flushTimerRef.current = setTimeout(flushTokenBuffer, 60);
      }
    },
    [flushTokenBuffer],
  );

  /** Flush remaining buffer then mark stream as done */
  const finishStream = useCallback(() => {
    if (flushTimerRef.current !== null) {
      clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    const leftover = tokenBufferRef.current;
    tokenBufferRef.current = '';
    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantIdRef.current
          ? {
              ...m,
              content: m.content + leftover,
              isStreaming: false,
              citations: [...citationsRef.current],
            }
          : m,
      ),
    );
  }, []);

  /** Handle error on assistant message */
  const handleError = useCallback((error: string) => {
    if (flushTimerRef.current !== null) {
      clearTimeout(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    tokenBufferRef.current = '';
    setMessages((prev) =>
      prev.map((m) =>
        m.id === assistantIdRef.current
          ? { ...m, isStreaming: false, error }
          : m,
      ),
    );
  }, []);

  /* ─── SSE handler ──────────────────────────────────────────────────── */

  const onEvent = useCallback(
    (event: SSEEvent) => {
      switch (event.type) {
        case 'debug':
          setDebugInfo({
            vector_results: event.vector_results || [],
            bm25_results: event.bm25_results || [],
            rrf_fused: event.rrf_fused || [],
            reranked: event.reranked || [],
            dropped: event.dropped || [],
            effective_params: event.effective_params || null,
            entity_info: event.entity_info || null,
            retrieval_mode: event.retrieval_mode || null,
            image_branch_info: event.image_branch_info || null,
          });
          setDebugPanelOpen(true);
          break;
        case 'token':
          if (event.content) bufferToken(event.content);
          break;
        case 'citation':
          citationsRef.current.push({
            source: event.source || 'unknown',
            page: event.page,
            speaker: event.speaker,
            modality: event.modality || 'document',
            image_path: event.image_path,
            timestamp_start: event.timestamp_start,
            timestamp_end: event.timestamp_end,
            file_id: event.file_id,
          });
          break;
        case 'status':
          // "generating" — no UI action needed
          break;
        case 'meta':
          // Confidence + cost + hallucination metadata from backend
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantIdRef.current
                ? {
                    ...m,
                    confidence: event.confidence,
                    cost: event.cost,
                    hallucination: event.hallucination,
                    cached: event.cached,
                  }
                : m,
            ),
          );
          break;
        case 'done':
          finishStream();
          break;
        case 'error':
          handleError(event.content || 'Unknown error');
          break;
      }
    },
    [bufferToken, finishStream, handleError],
  );

  const { start, cancel } = useSSE({
    onEvent,
    onDone: finishStream,
    onError: handleError,
  });

  /* ─── Cleanup on unmount ───────────────────────────────────────────── */
  useEffect(() => {
    return () => {
      cancel();
      if (flushTimerRef.current !== null) clearTimeout(flushTimerRef.current);
    };
  }, [cancel]);

  /* ─── Send a message ───────────────────────────────────────────────── */

  const isStreaming = messages.some((m) => m.isStreaming);

  const handleSend = useCallback(
    (text: string) => {
      if (isStreaming) return;

      // User message
      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: text,
        timestamp: Date.now(),
      };

      // Placeholder assistant message (streaming)
      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
        isStreaming: true,
        timestamp: Date.now(),
      };

      assistantIdRef.current = assistantMsg.id;
      citationsRef.current = [];

      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      // Fire the SSE stream
      start({ query: text, max_tokens: 768, debug: debugEnabled });
    },
    [isStreaming, start],
  );

  /* ─── Cancel current stream ────────────────────────────────────────── */

  const handleCancel = useCallback(() => {
    cancel();
    finishStream();
  }, [cancel, finishStream]);

  /* ─── Image query handler ──────────────────────────────────────────── */

  const handleImageQuery = useCallback(
    async (file: File, prompt?: string) => {
      if (isStreaming) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: prompt ? `🖼️ Image query: ${prompt}` : '🖼️ Image query',
        timestamp: Date.now(),
      };

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
        isStreaming: true,
        timestamp: Date.now(),
      };
      assistantIdRef.current = assistantMsg.id;
      citationsRef.current = [];
      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      try {
        const formData = new FormData();
        formData.append('file', file);
        if (prompt) formData.append('text_prompt', prompt);
        formData.append('top_k', '5');
        formData.append('debug', debugEnabled ? 'true' : 'false');

        const resp = await fetch('/query/image', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error(`Image query failed: ${resp.status}`);
        const data = (await resp.json()) as ImageQueryResponse;

        let resultText = '';

        // If LLM answer is available (OCR fallback path), show it prominently
        if (data.answer) {
          resultText = data.answer;
          if (data.ocr_text) {
            resultText += `\n\n---\n_📝 Text extracted from image (OCR):_\n> ${data.ocr_text.slice(0, 500)}${data.ocr_text.length > 500 ? '…' : ''}`;
          }
          if (data.results.length > 0) {
            resultText += `\n\n**Related sources** (${data.result_count}):\n` +
              data.results.slice(0, 5).map((r, i) =>
                `${i + 1}. **${r.source}**${r.page ? ` (p${r.page})` : ''}`
              ).join('\n');
          }
          resultText += `\n\n_Latency: ${data.latency_split.total_ms?.toFixed(0) ?? '?'}ms_`;
        } else if (data.results.length > 0) {
          resultText = `Found **${data.result_count}** results via image search.\n\n` +
            data.results.map((r, i) =>
              `${i + 1}. **${r.source}**${r.page ? ` (p${r.page})` : ''} — score ${r.score?.toFixed(4) ?? '?'} [${r.origin}]`
            ).join('\n') +
            `\n\n_Latency: ${data.latency_split.total_ms?.toFixed(0) ?? '?'}ms_`;
        } else if (data.ocr_text) {
          resultText = `No matching documents found, but here's the text extracted from the image:\n\n> ${data.ocr_text.slice(0, 1000)}`;
        } else {
          resultText = 'No matching results found for this image.';
        }

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantIdRef.current
              ? { ...m, content: resultText, isStreaming: false }
              : m,
          ),
        );
      } catch (err: unknown) {
        const errMsg = err instanceof Error ? err.message : 'Image query failed';
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantIdRef.current
              ? { ...m, isStreaming: false, error: errMsg }
              : m,
          ),
        );
      }
    },
    [isStreaming, debugEnabled],
  );

  /* ─── Audio query handler ──────────────────────────────────────────── */

  const handleAudioQuery = useCallback(
    async (file: File) => {
      if (isStreaming) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: `🎙️ Audio query: ${file.name}`,
        timestamp: Date.now(),
      };

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: '',
        isStreaming: true,
        timestamp: Date.now(),
      };
      assistantIdRef.current = assistantMsg.id;
      citationsRef.current = [];
      setMessages((prev) => [...prev, userMsg, assistantMsg]);

      try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('max_tokens', '768');

        const resp = await fetch('/query/audio', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error(`Audio query failed: ${resp.status}`);
        const data = (await resp.json()) as AudioQueryResponse;

        const resultText =
          `**Transcript:** ${data.transcript}\n\n` +
          `**Answer:** ${data.answer}\n\n` +
          `_Transcription: ${data.transcription_latency_ms.toFixed(0)}ms | Total: ${data.total_latency_ms.toFixed(0)}ms_`;

        // Add citations from audio response
        citationsRef.current = data.citations || [];

        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantIdRef.current
              ? { ...m, content: resultText, isStreaming: false, citations: [...citationsRef.current] }
              : m,
          ),
        );
      } catch (err: unknown) {
        const errMsg = err instanceof Error ? err.message : 'Audio query failed';
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantIdRef.current
              ? { ...m, isStreaming: false, error: errMsg }
              : m,
          ),
        );
      }
    },
    [isStreaming],
  );

  return (
    <div className="flex flex-col h-full">
      {/* Debug toggle bar */}
      <div className="flex items-center justify-end gap-4 px-4 py-2 border-b border-slate-800/50 bg-slate-900/30">
        <button
          onClick={() => setSettingsOpen(true)}
          className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-blue-400 transition-colors"
        >
          <Settings2 className="w-3.5 h-3.5" />
          <span className="font-medium">Tuning</span>
        </button>
        <label className="flex items-center gap-2 cursor-pointer select-none">
          <Bug className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-xs text-slate-500 font-medium">Debug</span>
          <button
            role="switch"
            aria-checked={debugEnabled}
            onClick={() => setDebugEnabled((v) => !v)}
            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors
                       ${debugEnabled ? 'bg-amber-600' : 'bg-slate-700'}`}
          >
            <span
              className={`inline-block h-3.5 w-3.5 rounded-full bg-white transition-transform
                         ${debugEnabled ? 'translate-x-[18px]' : 'translate-x-[3px]'}`}
            />
          </button>
        </label>
      </div>

      {/* 2-column layout: chat thread (left) + debug panel (right) */}
      <div className="flex flex-1 min-h-0">
        {/* Left: Chat thread */}
        <div className="flex flex-col flex-1 min-w-0">
          <MessageList messages={messages} onSuggestionClick={handleSend} />

          {isStreaming && (
            <div className="flex justify-center py-2">
              <button
                onClick={handleCancel}
                className="text-xs text-slate-500 hover:text-slate-300 border border-slate-700
                           rounded-full px-4 py-1.5 transition-all duration-150
                           hover:bg-slate-800/50 active:scale-95"
              >
                ■ Stop generating
              </button>
            </div>
          )}

          <ChatInput
            onSend={handleSend}
            onImageQuery={handleImageQuery}
            onAudioQuery={handleAudioQuery}
            disabled={isStreaming}
          />
        </div>

        {/* Right: Debug panel (when active) */}
        {debugPanelOpen && debugEnabled && (
          <div className="w-[400px] flex-shrink-0 border-l border-slate-800/80 bg-slate-900/60 backdrop-blur-sm overflow-y-auto animate-slide-in">
            <RetrievalDebugPanel
              debugInfo={debugInfo}
              isOpen={true}
              onClose={() => setDebugPanelOpen(false)}
            />
          </div>
        )}
      </div>

      {/* Advanced Settings Modal */}
      <AdvancedSettingsPanel
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
