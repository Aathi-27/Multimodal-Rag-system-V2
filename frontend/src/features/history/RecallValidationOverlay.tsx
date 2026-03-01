import { useCallback, useEffect, useState } from 'react';
import axios from '@/shared/api/axios';
import type {
  ChunkRelevanceAnnotation,
  QueryDetailResponse,
  RecallMetrics,
} from '@/shared/types/api.types';

interface Props {
  queryId: string;
  isOpen: boolean;
  onClose: () => void;
}

export default function RecallValidationOverlay({ queryId, isOpen, onClose }: Props) {
  const [detail, setDetail] = useState<QueryDetailResponse | null>(null);
  const [annotations, setAnnotations] = useState<Record<string, boolean>>({});
  const [metrics, setMetrics] = useState<RecallMetrics | null>(null);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(false);

  /* ── Load query detail ───────────────────────────────────── */
  const loadDetail = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await axios.get<QueryDetailResponse>(`/queries/${queryId}`);
      setDetail(data);

      // Pre-populate annotations if they already exist
      const existing = data.recall_validation?.annotations;
      if (existing) {
        const map: Record<string, boolean> = {};
        for (const a of existing) {
          map[a.chunk_id] = a.relevant;
        }
        setAnnotations(map);
      }

      // Load existing recall metrics if available
      try {
        const { data: rm } = await axios.get<RecallMetrics>(`/queries/${queryId}/recall`);
        setMetrics(rm);
      } catch {
        // No recall data yet — that's fine
      }
    } catch {
      // Failed to load
    } finally {
      setLoading(false);
    }
  }, [queryId]);

  useEffect(() => {
    if (isOpen && queryId) loadDetail();
  }, [isOpen, queryId, loadDetail]);

  if (!isOpen) return null;

  /* ── Toggle annotation ───────────────────────────────────── */
  const toggle = (chunkId: string) => {
    setAnnotations((prev) => {
      const current = prev[chunkId];
      if (current === undefined) return { ...prev, [chunkId]: true };
      if (current === true) return { ...prev, [chunkId]: false };
      // If false, remove it
      const next = { ...prev };
      delete next[chunkId];
      return next;
    });
  };

  /* ── Submit annotations ──────────────────────────────────── */
  const handleSubmit = async () => {
    setSaving(true);
    try {
      const body: { annotations: ChunkRelevanceAnnotation[] } = {
        annotations: Object.entries(annotations).map(([chunk_id, relevant]) => ({
          chunk_id,
          relevant,
        })),
      };
      const { data } = await axios.post<RecallMetrics>(
        `/queries/${queryId}/validate`,
        body,
      );
      setMetrics(data);
    } catch {
      // Handle error
    } finally {
      setSaving(false);
    }
  };

  const chunks = detail?.retrieved_chunks || [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-700/80 rounded-xl shadow-2xl shadow-black/40 w-full max-w-2xl mx-4 max-h-[85vh] flex flex-col animate-fade-in-up">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <div>
            <h2 className="text-sm font-semibold text-slate-200">Recall Validation</h2>
            <p className="text-[10px] text-slate-500 mt-0.5">
              Mark which retrieved chunks are relevant to the query
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors"
          >
            <CloseIcon className="w-4 h-4" />
          </button>
        </div>

        {/* Query preview */}
        {detail && (
          <div className="px-5 py-2 border-b border-slate-800/50">
            <p className="text-xs text-slate-400 italic">"{detail.query}"</p>
          </div>
        )}

        {/* Metrics banner */}
        {metrics && (
          <div className="px-5 py-2 border-b border-slate-800/50 flex gap-6 text-xs">
            <MetricPill label="Recall@5" value={metrics.recall_at_5} />
            <MetricPill label="Recall@10" value={metrics.recall_at_10} />
            <MetricPill label="MRR" value={metrics.mrr} />
            <span className="text-slate-600">
              {metrics.relevant_count}/{metrics.annotated_count} relevant
            </span>
          </div>
        )}

        {/* Chunk list */}
        <div className="flex-1 overflow-y-auto px-5 py-3 space-y-2 scrollbar-thin scrollbar-thumb-slate-700">
          {loading ? (
            <p className="text-xs text-slate-600 text-center py-8">Loading…</p>
          ) : chunks.length === 0 ? (
            <p className="text-xs text-slate-600 text-center py-8">No retrieved chunks.</p>
          ) : (
            chunks.map((chunk, idx) => {
              const state = annotations[chunk.chunk_id]; // undefined | true | false
              return (
                <div
                  key={chunk.chunk_id}
                  className={`flex items-center gap-3 p-2.5 rounded-lg border transition-colors cursor-pointer
                    ${state === true
                      ? 'border-emerald-600/40 bg-emerald-500/5'
                      : state === false
                        ? 'border-red-600/40 bg-red-500/5'
                        : 'border-slate-800 bg-slate-800/30 hover:border-slate-700'
                    }`}
                  onClick={() => toggle(chunk.chunk_id)}
                >
                  {/* Rank */}
                  <span className="text-xs font-mono text-slate-500 w-5 text-right tabular-nums">
                    {idx + 1}
                  </span>

                  {/* Relevance indicator */}
                  <span className="flex-shrink-0 w-6 text-center">
                    {state === true && <span className="text-emerald-400 text-sm">✓</span>}
                    {state === false && <span className="text-red-400 text-sm">✗</span>}
                    {state === undefined && <span className="text-slate-700 text-sm">○</span>}
                  </span>

                  {/* Chunk info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-slate-400">{chunk.chunk_id}</span>
                      <span className="text-xs text-slate-500 truncate">{chunk.source}</span>
                    </div>
                    <span className="text-[10px] text-slate-600">
                      Score: {chunk.score.toFixed(4)}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-slate-800">
          <span className="text-[10px] text-slate-600">
            Click: relevant (✓) → irrelevant (✗) → unannotated (○)
          </span>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 border border-slate-700
                         rounded-lg transition-all duration-150 active:scale-95"
            >
              Cancel
            </button>
            <button
              onClick={handleSubmit}
              disabled={saving || Object.keys(annotations).length === 0}
              className="px-4 py-1.5 text-xs font-medium text-white bg-emerald-600 hover:bg-emerald-500
                         rounded-lg transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed
                         active:scale-95"
            >
              {saving ? 'Saving…' : `Save (${Object.keys(annotations).length} annotated)`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Helpers ───────────────────────────────────────────────── */

function MetricPill({ label, value }: { label: string; value: number | null }) {
  const display = value != null ? (value * 100).toFixed(1) + '%' : '—';
  const color = value != null && value >= 0.8
    ? 'text-emerald-400'
    : value != null && value >= 0.5
      ? 'text-amber-400'
      : 'text-red-400';

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-slate-500">{label}</span>
      <span className={`font-mono font-medium tabular-nums ${color}`}>{display}</span>
    </div>
  );
}

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
    </svg>
  );
}
