import { useEffect, useRef, useState, lazy, Suspense } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';
import remarkGfm from 'remark-gfm';
import type { ChatMessage, Citation, ConfidenceInfo, QueryCostInfo, HallucinationInfo } from '@/shared/types/api.types';
import { Search, Timer, BookOpen, MessageSquare, FileText, Headphones, Image, ShieldCheck, Zap, DollarSign, AlertTriangle } from 'lucide-react';
import Loader from '@/shared/components/Loader';

// Lazy-load viewers to keep bundle light until needed
const PDFViewer = lazy(() => import('./viewers/PDFViewer'));
const AudioPlayer = lazy(() => import('./viewers/AudioPlayer'));
const ImageModal = lazy(() => import('./viewers/ImageModal'));

/* ─── Viewer state ───────────────────────────────────────────────────────── */

interface ViewerState {
  type: 'pdf' | 'audio' | 'image' | 'download';
  fileId: string;
  fileName: string;
  page?: number;
  timestamp?: number;
  speaker?: string | null;
}

interface MessageListProps {
  messages: ChatMessage[];
  onSuggestionClick?: (text: string) => void;
}

export default function MessageList({ messages, onSuggestionClick }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [viewer, setViewer] = useState<ViewerState | null>(null);

  // Auto-scroll: instant during streaming (avoids jank), smooth for new messages
  const lastMsg = messages[messages.length - 1];
  const isActivelyStreaming = lastMsg?.isStreaming ?? false;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: isActivelyStreaming ? 'auto' : 'smooth',
    });
  }, [messages, lastMsg?.content, isActivelyStreaming]);

  const handleCitationClick = (citation: Citation) => {
    if (!citation.file_id) return; // No file_id — can't navigate

    const modality = citation.modality || 'document';

    if (modality === 'document') {
      const ext = (citation.source || '').split('.').pop()?.toLowerCase() || '';
      if (ext === 'pdf') {
        setViewer({
          type: 'pdf',
          fileId: citation.file_id,
          fileName: citation.source,
          page: citation.page ?? 1,
        });
      } else {
        // Non-PDF documents (docx, xlsx, pptx, txt, etc.) — download directly
        const API_BASE = import.meta.env.VITE_API_URL || '';
        const link = document.createElement('a');
        link.href = `${API_BASE}/files/${citation.file_id}`;
        link.download = citation.source || 'document';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } else if (modality === 'audio') {
      setViewer({
        type: 'audio',
        fileId: citation.file_id,
        fileName: citation.source,
        timestamp: citation.timestamp_start ?? 0,
        speaker: citation.speaker,
      });
    } else if (modality === 'image') {
      setViewer({
        type: 'image',
        fileId: citation.file_id,
        fileName: citation.source,
      });
    }
  };

  if (messages.length === 0) {
    return (
      <>
        <EmptyState onSuggestionClick={onSuggestionClick} />
        {viewer && (
          <Suspense fallback={null}>
            <ViewerOverlay viewer={viewer} onClose={() => setViewer(null)} />
          </Suspense>
        )}
      </>
    );
  }

  return (
    <>
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
        <div className="mx-auto max-w-3xl space-y-5">
          {messages.map((msg) => (
            <MessageBubble
              key={msg.id}
              message={msg}
              onCitationClick={handleCitationClick}
            />
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Viewer overlay */}
      {viewer && (
        <Suspense fallback={null}>
          <ViewerOverlay viewer={viewer} onClose={() => setViewer(null)} />
        </Suspense>
      )}
    </>
  );
}

/* ─── Viewer overlay (renders the correct viewer) ────────────────────────── */

function ViewerOverlay({ viewer, onClose }: { viewer: ViewerState; onClose: () => void }) {
  switch (viewer.type) {
    case 'pdf':
      return (
        <PDFViewer
          fileId={viewer.fileId}
          fileName={viewer.fileName}
          initialPage={viewer.page}
          onClose={onClose}
        />
      );
    case 'audio':
      return (
        <AudioPlayer
          fileId={viewer.fileId}
          fileName={viewer.fileName}
          initialTimestamp={viewer.timestamp}
          speaker={viewer.speaker}
          onClose={onClose}
        />
      );
    case 'image':
      return (
        <ImageModal
          fileId={viewer.fileId}
          fileName={viewer.fileName}
          onClose={onClose}
        />
      );
    case 'download':
      return null; // Downloads are handled directly, no overlay needed
  }
}

/* ─── Single message bubble ──────────────────────────────────────────────── */

function MessageBubble({
  message,
  onCitationClick,
}: {
  message: ChatMessage;
  onCitationClick: (c: Citation) => void;
}) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {/* Avatar */}
      {!isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white text-xs font-bold mt-1">
          AI
        </div>
      )}

      <div className={`max-w-[85%] min-w-0 ${isUser ? 'order-first' : ''}`}>
        {/* Bubble */}
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-blue-600 text-white rounded-br-md'
              : 'bg-slate-800 text-slate-200 rounded-bl-md border border-slate-700/50'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <>
              {message.content ? (
                <div className="prose-chat">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeSanitize]}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
              ) : message.isStreaming ? (
                <Loader size="sm" label="Thinking…" />
              ) : null}

              {message.isStreaming && message.content && (
                <span className="inline-block w-1.5 h-4 bg-blue-400 rounded-sm ml-0.5 animate-pulse" />
              )}
            </>
          )}

          {message.error && (
            <p className="mt-2 text-xs text-red-400">⚠ {message.error}</p>
          )}
        </div>

        {/* Citations */}
        {!isUser && message.citations && message.citations.length > 0 && !message.isStreaming && (
          <CitationList citations={message.citations} onCitationClick={onCitationClick} />
        )}

        {/* Confidence + Cost meta row */}
        {!isUser && !message.isStreaming && message.confidence && (
          <MetaInfoRow confidence={message.confidence} cost={message.cost} hallucination={message.hallucination} cached={message.cached} />
        )}

        {/* Timestamp */}
        <p className={`text-[10px] mt-1.5 text-slate-600 ${isUser ? 'text-right' : ''}`}>
          {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>

      {/* User avatar */}
      {isUser && (
        <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-slate-700 flex items-center justify-center text-slate-300 text-xs font-bold mt-1">
          U
        </div>
      )}
    </div>
  );
}

/* ─── Confidence + Cost meta row ─────────────────────────────────────────── */

function MetaInfoRow({
  confidence,
  cost,
  hallucination,
  cached,
}: {
  confidence: ConfidenceInfo;
  cost?: QueryCostInfo;
  hallucination?: HallucinationInfo;
  cached?: boolean;
}) {
  const badgeColors: Record<string, string> = {
    high: 'bg-emerald-900/50 border-emerald-600/40 text-emerald-300',
    medium: 'bg-amber-900/50 border-amber-600/40 text-amber-300',
    low: 'bg-red-900/50 border-red-600/40 text-red-300',
  };
  const dotColors: Record<string, string> = {
    high: 'bg-emerald-400',
    medium: 'bg-amber-400',
    low: 'bg-red-400',
  };

  // Hallucination risk colors (inverted — low risk is good)
  const hallucinationColors: Record<string, string> = {
    low: 'text-emerald-400',
    medium: 'text-amber-400',
    high: 'text-red-400',
    unknown: 'text-slate-500',
  };

  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
      {/* Confidence badge */}
      <span
        className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md border ${badgeColors[confidence.level] || badgeColors.low}`}
        title={`Score: ${(confidence.score * 100).toFixed(0)}% — ${confidence.grounding}`}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${dotColors[confidence.level] || dotColors.low}`} />
        <ShieldCheck className="w-3 h-3" />
        {confidence.level.charAt(0).toUpperCase() + confidence.level.slice(1)} confidence
      </span>

      {/* Hallucination grounding */}
      {hallucination && (
        <span
          className={`inline-flex items-center gap-1 ${hallucinationColors[hallucination.risk_level] || 'text-slate-500'}`}
          title={`${hallucination.grounded_sentences}/${hallucination.total_sentences} sentences grounded${hallucination.ungrounded_claims.length > 0 ? '\nUngrounded: ' + hallucination.ungrounded_claims[0].slice(0, 80) + '...' : ''}`}
        >
          <AlertTriangle className="w-3 h-3" />
          {(hallucination.grounded_ratio * 100).toFixed(0)}% grounded
        </span>
      )}

      {/* Source count */}
      <span className="text-slate-500">
        {confidence.source_count} source{confidence.source_count !== 1 ? 's' : ''}
      </span>

      {/* Latency */}
      {cost && (
        <span className="inline-flex items-center gap-1 text-slate-500" title={`Retrieval: ${cost.retrieval_time_ms}ms | Generation: ${cost.generation_time_ms}ms`}>
          <Zap className="w-3 h-3" />
          {cost.total_time_ms < 1000 ? `${cost.total_time_ms.toFixed(0)}ms` : `${(cost.total_time_ms / 1000).toFixed(1)}s`}
        </span>
      )}

      {/* Cost */}
      {cost && (
        <span className="inline-flex items-center gap-1 text-slate-500" title={`Tokens: ${cost.total_tokens} (${cost.prompt_tokens} prompt + ${cost.completion_tokens} completion)`}>
          <DollarSign className="w-3 h-3" />
          ${cost.estimated_cost_usd.toFixed(6)}
        </span>
      )}

      {/* Cached indicator */}
      {cached && (
        <span className="px-1.5 py-0.5 rounded bg-blue-900/40 border border-blue-600/30 text-blue-300 text-[10px]">
          cached
        </span>
      )}
    </div>
  );
}

/* ─── Citations panel ────────────────────────────────────────────────────── */

function CitationList({
  citations,
  onCitationClick,
}: {
  citations: Citation[];
  onCitationClick: (c: Citation) => void;
}) {
  return (
    <div className="mt-2 flex flex-wrap gap-2">
      {citations.map((c, i) => (
        <CitationBadge
          key={i}
          citation={c}
          index={i + 1}
          onClick={() => onCitationClick(c)}
        />
      ))}
    </div>
  );
}

function CitationBadge({
  citation,
  index,
  onClick,
}: {
  citation: Citation;
  index: number;
  onClick: () => void;
}) {
  const { source, page, speaker, modality, timestamp_start, file_id } = citation;
  const isClickable = !!file_id;

  // Build label
  let label = `[${index}] ${source}`;
  if (modality === 'document' && page != null) {
    label += `, p.${page}`;
  }
  if (modality === 'audio' && timestamp_start != null) {
    const mins = Math.floor(timestamp_start / 60);
    const secs = Math.floor(timestamp_start % 60);
    label += ` @ ${mins}:${secs.toString().padStart(2, '0')}`;
    if (speaker) label += ` (${speaker})`;
  }
  if (modality === 'image') {
    label = `[${index}] 🖼 ${source}`;
  }

  // Modality icon
  const Icon =
    modality === 'audio'
      ? Headphones
      : modality === 'image'
        ? Image
        : FileText;

  // Color scheme
  const badgeColor =
    modality === 'audio'
      ? 'bg-purple-900/40 border-purple-700/50 text-purple-300'
      : modality === 'image'
        ? 'bg-emerald-900/40 border-emerald-700/50 text-emerald-300'
        : 'bg-slate-800 border-slate-700 text-slate-400';

  const hoverClass = isClickable
    ? 'cursor-pointer hover:brightness-125 hover:scale-[1.03] active:scale-[0.98] transition-all duration-150'
    : '';

  return (
    <button
      type="button"
      onClick={isClickable ? onClick : undefined}
      disabled={!isClickable}
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs
                  ${badgeColor} ${hoverClass}
                  disabled:cursor-default disabled:opacity-70`}
      title={isClickable ? `Open ${source}` : 'File not available for preview'}
    >
      <Icon className="w-3 h-3 flex-shrink-0" />
      {label}
    </button>
  );
}

/* ─── Empty state ────────────────────────────────────────────────────────── */

const SUGGESTIONS = [
  { text: 'Inspect retrieval quality', desc: 'Test how well chunks match your queries', icon: Search },
  { text: 'Analyze system latency', desc: 'Check response time across pipeline stages', icon: Timer },
  { text: 'Summarize indexed knowledge', desc: 'Get an overview of your document corpus', icon: BookOpen },
];

function EmptyState({ onSuggestionClick }: { onSuggestionClick?: (text: string) => void }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-center px-4">
      <div className="w-14 h-14 rounded-2xl bg-slate-800 flex items-center justify-center mb-5">
        <MessageSquare className="w-7 h-7 text-slate-500" />
      </div>
      <h2 className="text-xl font-semibold text-slate-200 mb-1">Ask anything</h2>
      <p className="text-sm text-slate-500 max-w-md mb-8">
        Query your uploaded documents, images, and audio files.
        Responses are generated using retrieval-augmented generation.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full max-w-2xl">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.text}
            onClick={() => onSuggestionClick?.(s.text)}
            className="text-left p-4 rounded-xl bg-slate-900 hover:bg-slate-800/80
                       border border-slate-800/60 hover:border-slate-600/50
                       transition-all duration-200 group
                       hover:scale-[1.02] hover:shadow-lg hover:shadow-slate-950/50
                       animate-fade-in-up"
          >
            <s.icon className="w-4 h-4 text-slate-500 group-hover:text-blue-400 mb-2 transition-colors" />
            <p className="text-sm font-medium text-slate-300 group-hover:text-slate-100">{s.text}</p>
            <p className="text-xs text-slate-500 mt-1">{s.desc}</p>
          </button>
        ))}
      </div>
    </div>
  );
}
