import { useCallback, useEffect, useState } from 'react';
import api from '@/shared/api/axios';
import type {
  QueryHistoryItem,
  QueryHistoryResponse,
  QueryDetailResponse,
  QuerySummaryResponse,
} from '@/shared/types/api.types';
import { RefreshCw, ChevronRight, ChevronDown, AlertCircle, Trash2 } from 'lucide-react';
import Loader from '@/shared/components/Loader';
import ErrorMessage from '@/shared/components/ErrorMessage';
import Skeleton, { SkeletonCard, SkeletonRow } from '@/shared/components/Skeleton';
import RecallValidationOverlay from './RecallValidationOverlay';

export default function QueryHistoryPage() {
  const [queries, setQueries] = useState<QueryHistoryItem[]>([]);
  const [summary, setSummary] = useState<QuerySummaryResponse | null>(null);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);
  const [detail, setDetail] = useState<QueryDetailResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [recallQueryId, setRecallQueryId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null); // query_id being deleted, or 'all'

  const fetchHistory = useCallback(async () => {
    try {
      const [histRes, sumRes] = await Promise.allSettled([
        api.get<QueryHistoryResponse>('/queries?limit=100'),
        api.get<QuerySummaryResponse>('/queries/summary'),
      ]);
      if (histRes.status === 'fulfilled') {
        setQueries(histRes.value.data.queries);
        setTotal(histRes.value.data.total);
      }
      if (sumRes.status === 'fulfilled') {
        setSummary(sumRes.value.data);
      }
      setError('');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch query history');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const deleteQuery = useCallback(async (queryId: string) => {
    if (!confirm('Delete this query record?')) return;
    setDeleting(queryId);
    try {
      await api.delete(`/queries/${queryId}`);
      setQueries((prev) => prev.filter((q) => q.query_id !== queryId));
      setTotal((prev) => prev - 1);
      if (expanded === queryId) {
        setExpanded(null);
        setDetail(null);
      }
    } catch {
      setError('Failed to delete query');
    } finally {
      setDeleting(null);
    }
  }, [expanded]);

  const deleteAll = useCallback(async () => {
    if (!confirm(`Delete all ${total} query records? This cannot be undone.`)) return;
    setDeleting('all');
    try {
      await api.delete('/queries');
      setQueries([]);
      setTotal(0);
      setExpanded(null);
      setDetail(null);
      setSummary(null);
    } catch {
      setError('Failed to clear query history');
    } finally {
      setDeleting(null);
    }
  }, [total]);

  const toggleDetail = async (queryId: string) => {
    if (expanded === queryId) {
      setExpanded(null);
      setDetail(null);
      return;
    }
    setExpanded(queryId);
    setDetailLoading(true);
    try {
      const res = await api.get<QueryDetailResponse>(`/queries/${queryId}`);
      setDetail(res.data);
    } catch {
      setDetail(null);
    } finally {
      setDetailLoading(false);
    }
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1400px] px-6 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div>
            <h1 className="text-xl font-semibold text-slate-100">Query History</h1>
            <p className="text-sm text-slate-500 mt-0.5">
              {loading ? <Skeleton className="h-4 w-40 inline-block" /> : <>{total} total queries · Click a row to expand detail</>}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {queries.length > 0 && (
              <button
                onClick={deleteAll}
                disabled={deleting === 'all'}
                className="flex items-center gap-2 text-xs px-3 py-2 rounded-lg bg-red-900/30 text-red-400
                           hover:bg-red-900/50 transition-colors border border-red-800/50 disabled:opacity-50"
              >
                <Trash2 className="w-3.5 h-3.5" />
                {deleting === 'all' ? 'Clearing…' : 'Clear All'}
              </button>
            )}
            <button
              onClick={fetchHistory}
              className="flex items-center gap-2 text-xs px-3 py-2 rounded-lg bg-slate-800 text-slate-300
                         hover:bg-slate-700 transition-colors border border-slate-700/50"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Refresh
            </button>
          </div>
        </div>

        {error && (
          <div className="mb-6">
            <ErrorMessage message={error} onRetry={fetchHistory} />
          </div>
        )}

        {/* Summary Cards */}
        {summary && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3 mb-5">
            <SummaryCard label="Total Queries" value={summary.total_queries} />
            <SummaryCard label="Avg Latency" value={`${summary.avg_latency.toFixed(2)}s`} />
            <SummaryCard label="Avg Retrieval" value={`${summary.avg_retrieval_latency.toFixed(2)}s`} />
            <SummaryCard label="Avg Generation" value={`${summary.avg_generation_latency.toFixed(2)}s`} />
            <SummaryCard label="Avg Chunks" value={summary.avg_chunks_per_query.toFixed(1)} />
            <SummaryCard
              label="Errors"
              value={summary.error_count}
              alert={summary.error_count > 0}
            />
          </div>
        )}

        {/* Query Table */}
        {loading ? (
          <div className="space-y-5">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {Array.from({ length: 6 }).map((_, i) => <SkeletonCard key={i} />)}
            </div>
            <div className="rounded-xl border border-slate-800/80 overflow-hidden">
              <table className="w-full">
                <tbody>
                  {Array.from({ length: 6 }).map((_, i) => <SkeletonRow key={i} cols={6} />)}
                </tbody>
              </table>
            </div>
          </div>
        ) : queries.length === 0 ? (
          <div className="text-center py-16 text-slate-500 text-sm">
            No queries yet. Ask a question in Chat to start building history.
          </div>
        ) : (
          <div className="rounded-xl border border-slate-800/80 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-900/80 text-slate-400 text-xs uppercase tracking-wider">
                  <th className="text-left px-4 py-3 font-medium">Query</th>
                  <th className="text-right px-3 py-3 font-medium w-20">Chunks</th>
                  <th className="text-right px-3 py-3 font-medium w-28">Retrieval</th>
                  <th className="text-right px-3 py-3 font-medium w-28">Generation</th>
                  <th className="text-right px-3 py-3 font-medium w-24">Total</th>
                  <th className="text-right px-4 py-3 font-medium w-36">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {queries.map((q) => (
                  <QueryRow
                    key={q.query_id}
                    item={q}
                    isExpanded={expanded === q.query_id}
                    onToggle={() => toggleDetail(q.query_id)}
                    onDelete={() => deleteQuery(q.query_id)}
                    isDeleting={deleting === q.query_id}
                    detail={expanded === q.query_id ? detail : null}
                    detailLoading={expanded === q.query_id && detailLoading}
                    onValidateRecall={() => setRecallQueryId(q.query_id)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Recall Validation Overlay */}
        {recallQueryId && (
          <RecallValidationOverlay
            queryId={recallQueryId}
            isOpen={!!recallQueryId}
            onClose={() => setRecallQueryId(null)}
          />
        )}
      </div>
    </div>
  );
}

/* ─── Summary Card ────────────────────────────────────────────────────────── */

function SummaryCard({
  label,
  value,
  alert = false,
}: {
  label: string;
  value: string | number;
  alert?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 px-3 py-3">
      <p className="text-[11px] text-slate-500 font-medium">{label}</p>
      <p className={`text-lg font-semibold mt-0.5 ${alert ? 'text-red-400' : 'text-slate-200'}`}>
        {String(value)}
      </p>
    </div>
  );
}

/* ─── Query Row ───────────────────────────────────────────────────────────── */

function QueryRow({
  item,
  isExpanded,
  onToggle,
  onDelete,
  isDeleting,
  detail,
  detailLoading,
  onValidateRecall,
}: {
  item: QueryHistoryItem;
  isExpanded: boolean;
  onToggle: () => void;
  onDelete: () => void;
  isDeleting: boolean;
  detail: QueryDetailResponse | null;
  detailLoading: boolean;
  onValidateRecall: () => void;
}) {
  const time = new Date(item.timestamp * 1000);
  const latencyColor = (v: number) =>
    v < 0.5 ? 'text-emerald-400' : v < 1.5 ? 'text-amber-400' : 'text-red-400';
  const totalMs = item.total_latency * 1000;
  const totalColor = totalMs < 500 ? 'text-emerald-400' : totalMs < 1500 ? 'text-amber-400' : 'text-red-400';

  return (
    <>
      <tr
        onClick={onToggle}
        className={`cursor-pointer transition-all duration-200 ${
          isExpanded ? 'bg-slate-800/50' : 'hover:bg-slate-900/50'
        }`}
      >
        <td className="px-4 py-3">
          <div className="flex items-center gap-2">
            {isExpanded
              ? <ChevronDown className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
              : <ChevronRight className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
            }
            <span className="text-slate-200 truncate max-w-md">{item.query}</span>
            {item.error && (
              <span className="flex items-center gap-1 text-xs px-1.5 py-0.5 rounded bg-red-900/40 text-red-400 border border-red-800/50">
                <AlertCircle className="w-3 h-3" /> error
              </span>
            )}
          </div>
        </td>
        <td className="text-right px-3 py-3 text-slate-400 tabular-nums">{item.chunk_count}</td>
        <td className={`text-right px-3 py-3 tabular-nums font-medium ${latencyColor(item.retrieval_latency)}`}>
          {item.retrieval_latency.toFixed(2)}s
        </td>
        <td className={`text-right px-3 py-3 tabular-nums font-medium ${latencyColor(item.generation_latency)}`}>
          {item.generation_latency.toFixed(2)}s
        </td>
        <td className={`text-right px-3 py-3 font-semibold tabular-nums ${totalColor}`}>
          {item.total_latency.toFixed(2)}s
        </td>
        <td className="text-right px-4 py-3 text-xs text-slate-500 tabular-nums">
          <div className="flex items-center justify-end gap-2">
            <span>{time.toLocaleDateString()} {time.toLocaleTimeString()}</span>
            <button
              onClick={(e) => { e.stopPropagation(); onDelete(); }}
              disabled={isDeleting}
              className="p-1 rounded hover:bg-red-900/40 text-slate-600 hover:text-red-400
                         transition-colors disabled:opacity-50"
              title="Delete this query"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </td>
      </tr>

      {/* Expanded detail row */}
      {isExpanded && (
        <tr>
          <td colSpan={6} className="bg-slate-900/30 px-6 py-4">
            {detailLoading ? (
              <div className="flex justify-center py-4">
                <Loader size="sm" label="Loading detail…" />
              </div>
            ) : detail ? (
              <div className="space-y-4">
                {/* Answer preview */}
                <div>
                  <h4 className="text-xs font-medium text-slate-400 mb-1.5">Answer</h4>
                  <p className="text-sm text-slate-300 whitespace-pre-wrap leading-relaxed max-h-40 overflow-y-auto">
                    {detail.answer || '(no answer)'}
                  </p>
                </div>

                {/* Latency breakdown bar */}
                <div>
                  <h4 className="text-xs font-medium text-slate-400 mb-2">Latency Breakdown</h4>
                  <LatencyBar
                    retrieval={detail.retrieval_latency}
                    rerank={detail.rerank_latency}
                    generation={detail.generation_latency}
                    total={detail.total_latency}
                  />
                </div>

                {/* Retrieved chunks */}
                {detail.retrieved_chunks.length > 0 && (
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <h4 className="text-xs font-medium text-slate-400">
                        Retrieved Chunks ({detail.retrieved_chunks.length})
                      </h4>
                      <button
                        onClick={(e) => { e.stopPropagation(); onValidateRecall(); }}
                        className="text-[10px] px-2.5 py-1 rounded bg-emerald-600/20 text-emerald-400
                                   border border-emerald-700/40 hover:bg-emerald-600/30 transition-colors"
                      >
                        ✓ Validate Recall
                      </button>
                    </div>

                    {/* Show recall metrics if available */}
                    {detail.recall_validation && (
                      <div className="flex gap-4 text-xs mb-2 px-2 py-1.5 rounded-lg bg-emerald-500/5 border border-emerald-700/20">
                        <span className="text-slate-500">Recall@5: <b className="text-emerald-400">{detail.recall_validation.recall_at_5 != null ? (detail.recall_validation.recall_at_5 * 100).toFixed(0) + '%' : '—'}</b></span>
                        <span className="text-slate-500">Recall@10: <b className="text-emerald-400">{detail.recall_validation.recall_at_10 != null ? (detail.recall_validation.recall_at_10 * 100).toFixed(0) + '%' : '—'}</b></span>
                        <span className="text-slate-500">MRR: <b className="text-emerald-400">{detail.recall_validation.mrr != null ? detail.recall_validation.mrr.toFixed(3) : '—'}</b></span>
                        <span className="text-slate-500">{detail.recall_validation.relevant_count}/{detail.recall_validation.annotated_count} relevant</span>
                      </div>
                    )}

                    <div className="flex flex-wrap gap-2">
                      {detail.retrieved_chunks.map((c, i) => (
                        <span
                          key={i}
                          className="text-xs px-2 py-1 rounded-lg bg-slate-800 text-slate-300 border border-slate-700/50"
                        >
                          {c.source} <span className="text-slate-500">({c.score.toFixed(3)})</span>
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-sm text-slate-500">Failed to load detail.</p>
            )}
          </td>
        </tr>
      )}
    </>
  );
}

/* ─── Latency Bar ─────────────────────────────────────────────────────────── */

function LatencyBar({
  retrieval,
  rerank,
  generation,
  total,
}: {
  retrieval: number;
  rerank: number;
  generation: number;
  total: number;
}) {
  const other = Math.max(0, total - retrieval - rerank - generation);
  const pct = (v: number) => (total > 0 ? Math.max(2, (v / total) * 100) : 0);

  return (
    <div className="space-y-1.5">
      <div className="flex h-3 rounded-full overflow-hidden bg-slate-800">
        <div
          className="bg-blue-500"
          style={{ width: `${pct(retrieval)}%` }}
          title={`Retrieval: ${retrieval.toFixed(3)}s`}
        />
        <div
          className="bg-purple-500"
          style={{ width: `${pct(rerank)}%` }}
          title={`Rerank: ${rerank.toFixed(3)}s`}
        />
        <div
          className="bg-emerald-500"
          style={{ width: `${pct(generation)}%` }}
          title={`Generation: ${generation.toFixed(3)}s`}
        />
        {other > 0.001 && (
          <div
            className="bg-slate-600"
            style={{ width: `${pct(other)}%` }}
            title={`Other: ${other.toFixed(3)}s`}
          />
        )}
      </div>
      <div className="flex gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" />
          Retrieval {retrieval.toFixed(2)}s
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-purple-500 inline-block" />
          Rerank {rerank.toFixed(2)}s
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
          Generation {generation.toFixed(2)}s
        </span>
      </div>
    </div>
  );
}
