import { useEffect, useMemo, useRef, useState } from 'react';
import Loader from '@/shared/components/Loader';
import type { ChunkDetail } from './KnowledgeBasePage';

interface Props {
  source: string;
  chunks: ChunkDetail[];
  loading: boolean;
  onClose: () => void;
}

type ViewMode = 'chunks' | 'pages';

export default function ChunkViewer({ source, chunks, loading, onClose }: Props) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('chunks');

  /* Close on Escape key */
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  /* Group chunks by page */
  const pages = useMemo(() => {
    const map = new Map<string, { page: number | null; chunks: ChunkDetail[]; totalTokens: number }>();
    for (const chunk of chunks) {
      const key = chunk.page_start != null ? String(chunk.page_start) : 'unknown';
      if (!map.has(key)) {
        map.set(key, { page: chunk.page_start, chunks: [], totalTokens: 0 });
      }
      const group = map.get(key)!;
      group.chunks.push(chunk);
      group.totalTokens += chunk.token_count;
    }
    // Sort by page number
    return Array.from(map.values()).sort((a, b) => (a.page ?? 9999) - (b.page ?? 9999));
  }, [chunks]);

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Slide-over panel */}
      <div
        ref={panelRef}
        className="fixed inset-y-0 right-0 z-50 w-full max-w-2xl bg-slate-900 border-l
                   border-slate-800 shadow-2xl shadow-black/40 flex flex-col animate-slide-in"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <div className="min-w-0 flex-1 mr-4">
            <h2 className="text-sm font-semibold text-slate-100 truncate" title={source}>
              {source}
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              {loading ? 'Loading…' : `${chunks.length} chunk${chunks.length !== 1 ? 's' : ''}`}
              {!loading && chunks.length > 0 && (
                <>
                  {' · '}{chunks.reduce((s, c) => s + c.token_count, 0).toLocaleString()} tokens
                  {' · '}{pages.length} page{pages.length !== 1 ? 's' : ''}
                </>
              )}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* View mode toggle */}
            {!loading && chunks.length > 0 && (
              <div className="flex items-center rounded-lg bg-slate-800 p-0.5">
                <button
                  onClick={() => setViewMode('chunks')}
                  className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors
                    ${viewMode === 'chunks' ? 'bg-slate-700 text-slate-200' : 'text-slate-500 hover:text-slate-300'}`}
                >
                  Chunks
                </button>
                <button
                  onClick={() => setViewMode('pages')}
                  className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors
                    ${viewMode === 'pages' ? 'bg-slate-700 text-slate-200' : 'text-slate-500 hover:text-slate-300'}`}
                >
                  Pages
                </button>
              </div>
            )}
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-slate-200 transition-colors"
            >
              <CloseIcon className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {loading ? (
            <div className="flex items-center justify-center h-48">
              <Loader size="md" label="Loading chunks…" />
            </div>
          ) : chunks.length === 0 ? (
            <p className="text-center text-slate-500 py-12">No chunks found.</p>
          ) : viewMode === 'chunks' ? (
            chunks.map((chunk, idx) => (
              <ChunkCard key={chunk.chunk_id} chunk={chunk} index={idx} />
            ))
          ) : (
            pages.map((pageGroup, idx) => (
              <PageGroup key={idx} page={pageGroup.page} chunks={pageGroup.chunks} totalTokens={pageGroup.totalTokens} />
            ))
          )}
        </div>
      </div>
    </>
  );
}

/* ── Page Group ───────────────────────────────────────────── */

function PageGroup({ page, chunks, totalTokens }: { page: number | null; chunks: ChunkDetail[]; totalTokens: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-800/20 overflow-hidden">
      {/* Page header — clickable */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-800/40
                   hover:bg-slate-800/60 transition-colors text-left"
      >
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-slate-200">
            {page != null ? `Page ${page}` : 'Unknown Page'}
          </span>
          <span className="text-xs text-slate-500">
            {chunks.length} chunk{chunks.length !== 1 ? 's' : ''}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 tabular-nums">
            {totalTokens.toLocaleString()} tokens
          </span>
          <ChevronIcon className={`w-4 h-4 text-slate-500 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`} />
        </div>
      </button>

      {/* Expanded chunks */}
      {expanded && (
        <div className="p-3 space-y-3 border-t border-slate-800">
          {chunks.map((chunk, idx) => (
            <ChunkCard key={chunk.chunk_id} chunk={chunk} index={idx} />
          ))}
        </div>
      )}
    </div>
  );
}

/* ── Chunk Card ───────────────────────────────────────────── */

function ChunkCard({ chunk, index }: { chunk: ChunkDetail; index: number }) {
  return (
    <div className="group rounded-lg border border-slate-800 bg-slate-800/30 overflow-hidden">
      {/* Chunk header */}
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800/50 text-xs text-slate-400">
        <div className="flex items-center gap-3">
          <span className="font-mono font-medium text-slate-300">
            Chunk #{index + 1}
          </span>
          {chunk.page_start != null && (
            <span>Page {chunk.page_start}</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span>{chunk.token_count} tokens</span>
          <span className="font-mono text-slate-600 text-[10px] truncate max-w-[120px]" title={chunk.chunk_id}>
            {chunk.chunk_id.slice(0, 12)}…
          </span>
        </div>
      </div>

      {/* Chunk text */}
      <div className="px-4 py-3">
        <p className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto
                      scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
          {chunk.text}
        </p>
      </div>
    </div>
  );
}

/* ── Icons ────────────────────────────────────────────────── */

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
    </svg>
  );
}
