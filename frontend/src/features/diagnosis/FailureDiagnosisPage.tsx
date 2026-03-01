import { useEffect, useState } from 'react';
import api from '@/shared/api/axios';
import Loader from '@/shared/components/Loader';
import Skeleton from '@/shared/components/Skeleton';
import {
  AlertTriangle,
  BarChart3,
  ChevronDown,
  ChevronRight,
  Eye,
  Filter,
  Layers,
  Search,
  ShieldAlert,
  TrendingDown,
  Zap,
} from 'lucide-react';
import type {
  BatchDiagnosisResponse,
  CorpusCoverageResponse,
  DiagnosisResponse,
  EmbeddingQualityResponse,
  QueryHistoryItem,
  QueryHistoryResponse,
  SurvivalResponse,
} from '@/shared/types/api.types';

/* ── Root-cause badge colors ─────────────────────────────────────────────── */
const CAUSE_COLORS: Record<string, string> = {
  corpus_gap: 'bg-red-500/20 text-red-400 border-red-500/30',
  embedding_mismatch: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  rerank_threshold: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  rrf_dilution: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  entity_miss: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
  parameter_issue: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  no_failure: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
};

const CAUSE_LABELS: Record<string, string> = {
  corpus_gap: 'Corpus Gap',
  embedding_mismatch: 'Embedding Mismatch',
  rerank_threshold: 'Rerank Threshold',
  rrf_dilution: 'RRF Dilution',
  entity_miss: 'Entity Miss',
  parameter_issue: 'Parameter Issue',
  no_failure: 'No Failure',
};

/* ── Stage labels ────────────────────────────────────────────────────────── */
const STAGE_ORDER = ['vector', 'bm25', 'rrf', 'entity', 'reranker', 'final'] as const;
const STAGE_COLORS: Record<string, string> = {
  vector: 'bg-blue-500',
  bm25: 'bg-emerald-500',
  rrf: 'bg-purple-500',
  entity: 'bg-cyan-500',
  reranker: 'bg-amber-500',
  final: 'bg-green-500',
};

export default function FailureDiagnosisPage() {
  const [queries, setQueries] = useState<QueryHistoryItem[]>([]);
  const [batchDiag, setBatchDiag] = useState<BatchDiagnosisResponse | null>(null);
  const [coverage, setCoverage] = useState<CorpusCoverageResponse | null>(null);
  const [embeddingQuality, setEmbeddingQuality] = useState<EmbeddingQualityResponse | null>(null);
  const [selectedQueryId, setSelectedQueryId] = useState<string | null>(null);
  const [queryDiag, setQueryDiag] = useState<DiagnosisResponse | null>(null);
  const [survival, setSurvival] = useState<SurvivalResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'queries' | 'corpus' | 'embeddings'>('overview');
  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set());

  /* ── Load overview data on mount ─────────────────────────────────── */
  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const [qRes, dRes] = await Promise.allSettled([
          api.get<QueryHistoryResponse>('/queries', { params: { limit: 50 } }),
          api.post<BatchDiagnosisResponse>('/diagnosis/batch', null, { params: { limit: 50 } }),
        ]);
        if (qRes.status === 'fulfilled') setQueries(qRes.value.data.queries);
        if (dRes.status === 'fulfilled') setBatchDiag(dRes.value.data);
      } catch {
        /* ignore */
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  /* ── Load tab-specific data ──────────────────────────────────────── */
  useEffect(() => {
    if (activeTab === 'corpus' && !coverage) {
      api.get<CorpusCoverageResponse>('/corpus/coverage').then((r: { data: CorpusCoverageResponse }) => setCoverage(r.data)).catch(() => {});
    }
    if (activeTab === 'embeddings' && !embeddingQuality) {
      api.get<EmbeddingQualityResponse>('/embeddings/quality').then((r: { data: EmbeddingQualityResponse }) => setEmbeddingQuality(r.data)).catch(() => {});
    }
  }, [activeTab, coverage, embeddingQuality]);

  /* ── Load single-query diagnosis ─────────────────────────────────── */
  const selectQuery = async (qid: string) => {
    setSelectedQueryId(qid);
    setDetailLoading(true);
    setQueryDiag(null);
    setSurvival(null);
    try {
      const [diagRes, survRes] = await Promise.allSettled([
        api.get<DiagnosisResponse>(`/queries/${qid}/diagnosis`),
        api.get<SurvivalResponse>(`/queries/${qid}/survival`),
      ]);
      if (diagRes.status === 'fulfilled') setQueryDiag(diagRes.value.data);
      if (survRes.status === 'fulfilled') setSurvival(survRes.value.data);
    } finally {
      setDetailLoading(false);
    }
  };

  const toggleChunk = (cid: string) => {
    setExpandedChunks(prev => {
      const next = new Set(prev);
      if (next.has(cid)) next.delete(cid);
      else next.add(cid);
      return next;
    });
  };

  if (loading) {
    return (
      <div className="p-6 space-y-4">
        <Skeleton className="h-8 w-64" />
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => <Skeleton key={i} className="h-28 rounded-xl" />)}
        </div>
        <Skeleton className="h-96 rounded-xl" />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-[1400px] mx-auto p-6 space-y-6">
        {/* ── Header ──────────────────────────────────────────── */}
        <div>
          <h1 className="text-xl font-semibold text-slate-100 tracking-tight flex items-center gap-2">
            <ShieldAlert className="w-5 h-5 text-red-400" />
            Failure Diagnosis
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Every query is data. Every failure is signal. Diagnose, track, optimize.
          </p>
        </div>

        {/* ── Tab bar ─────────────────────────────────────────── */}
        <div className="flex gap-1 bg-slate-900/60 rounded-lg p-1 w-fit border border-slate-800/60">
          {(['overview', 'queries', 'corpus', 'embeddings'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-1.5 rounded-md text-xs font-medium transition-all duration-200 capitalize
                ${activeTab === tab
                  ? 'bg-slate-800 text-blue-400 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* ═══ Overview Tab ═══ */}
        {activeTab === 'overview' && batchDiag && (
          <div className="space-y-6">
            {/* Stat cards */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard
                label="Queries Analyzed"
                value={batchDiag.total_analyzed}
                icon={<BarChart3 className="w-4 h-4" />}
              />
              <StatCard
                label="Avg Confidence"
                value={`${(batchDiag.avg_confidence * 100).toFixed(0)}%`}
                icon={<Zap className="w-4 h-4" />}
                color={batchDiag.avg_confidence > 0.7 ? 'text-emerald-400' : 'text-amber-400'}
              />
              <StatCard
                label="Root Causes Found"
                value={Object.keys(batchDiag.cause_distribution).length}
                icon={<AlertTriangle className="w-4 h-4" />}
              />
              <StatCard
                label="Top Issue"
                value={
                  CAUSE_LABELS[
                    Object.entries(batchDiag.cause_distribution).sort(
                      (a, b) => b[1] - a[1],
                    )[0]?.[0] ?? ''
                  ] ?? '—'
                }
                icon={<TrendingDown className="w-4 h-4" />}
                color="text-red-400"
              />
            </div>

            {/* Cause distribution */}
            <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5">
              <h2 className="text-sm font-semibold text-slate-200 mb-4">Cause Distribution</h2>
              <div className="space-y-3">
                {Object.entries(batchDiag.cause_distribution)
                  .sort((a, b) => b[1] - a[1])
                  .map(([cause, count]) => {
                    const pct = batchDiag.total_analyzed > 0 ? (count / batchDiag.total_analyzed) * 100 : 0;
                    return (
                      <div key={cause} className="flex items-center gap-3">
                        <span
                          className={`text-xs font-medium px-2 py-0.5 rounded border ${CAUSE_COLORS[cause] ?? 'bg-slate-700 text-slate-300 border-slate-600'}`}
                        >
                          {CAUSE_LABELS[cause] ?? cause}
                        </span>
                        <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 rounded-full transition-all duration-500"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="text-xs text-slate-400 w-16 text-right">
                          {count} ({pct.toFixed(0)}%)
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Common recommendations */}
            {batchDiag.common_recommendations.length > 0 && (
              <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5">
                <h2 className="text-sm font-semibold text-slate-200 mb-3">Top Recommendations</h2>
                <div className="space-y-2">
                  {batchDiag.common_recommendations.map((rec, i) => (
                    <div key={i} className="flex items-start gap-2 text-sm">
                      <Zap className="w-3.5 h-3.5 text-amber-400 mt-0.5 flex-shrink-0" />
                      <span className="text-slate-300">{rec.recommendation}</span>
                      <span className="text-xs text-slate-500 ml-auto">×{rec.frequency}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══ Queries Tab ═══ */}
        {activeTab === 'queries' && (
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            {/* Query list (left) */}
            <div className="lg:col-span-2 bg-slate-900/60 rounded-xl border border-slate-800/60 overflow-hidden">
              <div className="px-4 py-3 border-b border-slate-800/60">
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Search className="w-4 h-4" />
                  Recent Queries
                </h2>
              </div>
              <div className="max-h-[600px] overflow-y-auto divide-y divide-slate-800/40">
                {queries.map(q => (
                  <button
                    key={q.query_id}
                    onClick={() => selectQuery(q.query_id)}
                    className={`w-full text-left px-4 py-3 transition-all duration-200 hover:bg-slate-800/40 ${
                      selectedQueryId === q.query_id ? 'bg-slate-800/60 border-l-2 border-blue-500' : ''
                    }`}
                  >
                    <p className="text-sm text-slate-200 truncate">{q.query}</p>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="text-[11px] text-slate-500">
                        {q.chunk_count} chunks · {(q.total_latency * 1000).toFixed(0)}ms
                      </span>
                      {q.error && (
                        <span className="text-[10px] text-red-400 bg-red-500/10 px-1.5 py-0.5 rounded">
                          error
                        </span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Detail panel (right) */}
            <div className="lg:col-span-3 space-y-4">
              {detailLoading ? (
                <div className="flex items-center justify-center h-64">
                  <Loader size="md" label="Analyzing…" />
                </div>
              ) : selectedQueryId && queryDiag ? (
                <>
                  {/* Diagnosis card */}
                  <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <h2 className="text-sm font-semibold text-slate-200">Root Cause Analysis</h2>
                      <span
                        className={`text-xs font-medium px-2.5 py-1 rounded border ${
                          CAUSE_COLORS[queryDiag.root_cause] ?? 'bg-slate-700 text-slate-300 border-slate-600'
                        }`}
                      >
                        {CAUSE_LABELS[queryDiag.root_cause] ?? queryDiag.root_cause}
                      </span>
                    </div>

                    {/* Confidence bar */}
                    <div>
                      <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
                        <span>Confidence</span>
                        <span>{(queryDiag.confidence * 100).toFixed(0)}%</span>
                      </div>
                      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            queryDiag.confidence > 0.7 ? 'bg-emerald-500' :
                            queryDiag.confidence > 0.4 ? 'bg-amber-500' : 'bg-red-500'
                          }`}
                          style={{ width: `${queryDiag.confidence * 100}%` }}
                        />
                      </div>
                    </div>

                    {/* Evidence */}
                    <div>
                      <p className="text-xs font-medium text-slate-400 mb-2">Evidence</p>
                      <ul className="space-y-1">
                        {queryDiag.evidence.map((e, i) => (
                          <li key={i} className="text-xs text-slate-300 flex items-start gap-2">
                            <Eye className="w-3 h-3 mt-0.5 text-slate-500 flex-shrink-0" />
                            {e}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {/* Recommendations */}
                    <div>
                      <p className="text-xs font-medium text-slate-400 mb-2">Recommendations</p>
                      <ul className="space-y-1">
                        {queryDiag.recommendations.map((r, i) => (
                          <li key={i} className="text-xs text-emerald-300 flex items-start gap-2">
                            <Zap className="w-3 h-3 mt-0.5 text-emerald-500 flex-shrink-0" />
                            {r}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {queryDiag.secondary_causes.length > 0 && (
                      <div className="flex items-center gap-2 pt-2 border-t border-slate-800/40">
                        <span className="text-[11px] text-slate-500">Secondary:</span>
                        {queryDiag.secondary_causes.map(c => (
                          <span
                            key={c}
                            className={`text-[10px] px-1.5 py-0.5 rounded border ${
                              CAUSE_COLORS[c] ?? 'bg-slate-700 text-slate-300 border-slate-600'
                            }`}
                          >
                            {CAUSE_LABELS[c] ?? c}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Survival funnel */}
                  {survival && survival.summary && (
                    <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5">
                      <h2 className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
                        <Filter className="w-4 h-4 text-purple-400" />
                        Survival Funnel
                      </h2>

                      {/* Funnel visualization */}
                      <div className="space-y-2">
                        {STAGE_ORDER.map(stage => {
                          const count = survival.summary!.stage_counts[stage] ?? 0;
                          const total = survival.summary!.total_chunks_seen || 1;
                          const pct = (count / total) * 100;
                          return (
                            <div key={stage} className="flex items-center gap-3">
                              <span className="text-xs text-slate-400 w-16 text-right capitalize">{stage}</span>
                              <div className="flex-1 h-4 bg-slate-800 rounded overflow-hidden">
                                <div
                                  className={`h-full ${STAGE_COLORS[stage]} rounded transition-all duration-700`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                              <span className="text-xs text-slate-300 w-20 text-right">
                                {count} ({pct.toFixed(0)}%)
                              </span>
                            </div>
                          );
                        })}
                      </div>

                      {/* Drop reasons */}
                      {Object.keys(survival.summary.drop_reasons).length > 0 && (
                        <div className="mt-4 pt-3 border-t border-slate-800/40">
                          <p className="text-[11px] font-medium text-slate-500 mb-2">Drop Reasons</p>
                          <div className="flex flex-wrap gap-2">
                            {Object.entries(survival.summary.drop_reasons).map(([reason, count]) => (
                              <span
                                key={reason}
                                className="text-[10px] px-2 py-0.5 bg-slate-800/80 text-slate-400 rounded border border-slate-700/40"
                              >
                                {reason}: {count}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Per-chunk survival detail */}
                  {survival && survival.survival_log.length > 0 && (
                    <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 overflow-hidden">
                      <div className="px-5 py-3 border-b border-slate-800/60">
                        <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                          <Layers className="w-4 h-4 text-blue-400" />
                          Per-Chunk Survival ({survival.survival_log.length})
                        </h2>
                      </div>
                      <div className="max-h-[400px] overflow-y-auto divide-y divide-slate-800/30">
                        {survival.survival_log.slice(0, 30).map(entry => (
                          <div key={entry.chunk_id}>
                            <button
                              onClick={() => toggleChunk(entry.chunk_id)}
                              className="w-full text-left px-5 py-2.5 flex items-center gap-3 hover:bg-slate-800/30 transition-colors"
                            >
                              {expandedChunks.has(entry.chunk_id)
                                ? <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
                                : <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
                              }
                              <span className="text-xs font-mono text-slate-400">{entry.chunk_id.slice(0, 10)}</span>
                              <span className="text-xs text-slate-500 truncate flex-1">{entry.source}</span>
                              {/* Stage dots */}
                              <div className="flex gap-1">
                                {STAGE_ORDER.map(s => (
                                  <div
                                    key={s}
                                    className={`w-2 h-2 rounded-full ${
                                      entry.stages[s]?.present ? STAGE_COLORS[s] : 'bg-slate-700'
                                    }`}
                                    title={`${s}: ${entry.stages[s]?.present ? '✓' : '✗'}`}
                                  />
                                ))}
                              </div>
                              <span
                                className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${
                                  entry.survived
                                    ? 'bg-emerald-500/20 text-emerald-400'
                                    : 'bg-red-500/20 text-red-400'
                                }`}
                              >
                                {entry.survived ? 'survived' : 'dropped'}
                              </span>
                            </button>
                            {expandedChunks.has(entry.chunk_id) && (
                              <div className="px-5 pb-3 pl-12 text-xs space-y-1 text-slate-400">
                                {entry.dropped_at && (
                                  <p>
                                    <span className="text-slate-500">Dropped at:</span>{' '}
                                    <span className="text-red-400">{entry.dropped_at}</span>
                                    {entry.dropped_reason && (
                                      <span className="text-slate-500"> — {entry.dropped_reason}</span>
                                    )}
                                  </p>
                                )}
                                <div className="flex flex-wrap gap-3">
                                  {STAGE_ORDER.map(s => {
                                    const info = entry.stages[s];
                                    if (!info?.present) return null;
                                    return (
                                      <span key={s} className="text-[11px]">
                                        <span className="text-slate-500">{s}:</span>{' '}
                                        {info.rank != null && `#${info.rank}`}
                                        {info.score != null && ` (${info.score.toFixed(4)})`}
                                      </span>
                                    );
                                  })}
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="flex flex-col items-center justify-center h-64 text-slate-500">
                  <Search className="w-8 h-8 mb-2" />
                  <p className="text-sm">Select a query to diagnose</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ═══ Corpus Tab ═══ */}
        {activeTab === 'corpus' && (
          <div className="space-y-6">
            {!coverage ? (
              <div className="flex items-center justify-center h-48">
                <Loader size="md" label="Analyzing corpus…" />
              </div>
            ) : (
              <>
                {/* Coverage stats */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <StatCard label="Total Chunks" value={coverage.total_corpus_chunks} icon={<Layers className="w-4 h-4" />} />
                  <StatCard
                    label="Coverage Rate"
                    value={`${(coverage.coverage_rate * 100).toFixed(1)}%`}
                    icon={<BarChart3 className="w-4 h-4" />}
                    color={coverage.coverage_rate > 0.5 ? 'text-emerald-400' : 'text-amber-400'}
                  />
                  <StatCard label="Never Retrieved" value={coverage.never_retrieved_count} icon={<AlertTriangle className="w-4 h-4" />} color="text-red-400" />
                  <StatCard label="Queries Analyzed" value={coverage.total_queries_analyzed} icon={<Search className="w-4 h-4" />} />
                </div>

                {/* Per-source coverage */}
                <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 overflow-hidden">
                  <div className="px-5 py-3 border-b border-slate-800/60">
                    <h2 className="text-sm font-semibold text-slate-200">Per-Source Coverage</h2>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-slate-800/40 text-slate-500">
                          <th className="px-5 py-2 text-left font-medium">Source</th>
                          <th className="px-3 py-2 text-right font-medium">Chunks</th>
                          <th className="px-3 py-2 text-right font-medium">Retrieved</th>
                          <th className="px-3 py-2 font-medium">Coverage</th>
                          <th className="px-3 py-2 text-right font-medium">Never Retrieved</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/30">
                        {coverage.per_source_coverage.map(src => (
                          <tr key={src.source} className="hover:bg-slate-800/20">
                            <td className="px-5 py-2 text-slate-300 truncate max-w-[200px]">{src.source}</td>
                            <td className="px-3 py-2 text-right text-slate-400">{src.total_chunks}</td>
                            <td className="px-3 py-2 text-right text-slate-400">{src.retrieved_chunks}</td>
                            <td className="px-3 py-2">
                              <div className="flex items-center gap-2">
                                <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                                  <div
                                    className={`h-full rounded-full ${
                                      src.coverage_rate > 0.5 ? 'bg-emerald-500' :
                                      src.coverage_rate > 0.2 ? 'bg-amber-500' : 'bg-red-500'
                                    }`}
                                    style={{ width: `${src.coverage_rate * 100}%` }}
                                  />
                                </div>
                                <span className="text-slate-400 w-10 text-right">{(src.coverage_rate * 100).toFixed(0)}%</span>
                              </div>
                            </td>
                            <td className="px-3 py-2 text-right text-red-400">{src.never_retrieved}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Hotspots / Coldspots */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5">
                    <h2 className="text-sm font-semibold text-slate-200 mb-3">🔥 Hotspot Chunks</h2>
                    <div className="space-y-2">
                      {coverage.hotspot_chunks.map((c, i) => (
                        <div key={i} className="flex items-center gap-3 text-xs">
                          <span className="text-slate-500 w-4">{i + 1}</span>
                          <span className="text-slate-400 font-mono truncate flex-1">{c.chunk_id.slice(0, 12)}</span>
                          <span className="text-emerald-400">{c.retrieval_count}×</span>
                          <span className="text-slate-500">avg rank #{c.avg_rank.toFixed(1)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5">
                    <h2 className="text-sm font-semibold text-slate-200 mb-3">❄️ Coldspot Chunks</h2>
                    <div className="space-y-2">
                      {coverage.coldspot_chunks.map((c, i) => (
                        <div key={i} className="flex items-center gap-3 text-xs">
                          <span className="text-slate-500 w-4">{i + 1}</span>
                          <span className="text-slate-400 font-mono truncate flex-1">{c.chunk_id.slice(0, 12)}</span>
                          <span className="text-amber-400">{c.retrieval_count}×</span>
                          <span className="text-slate-500">avg rank #{c.avg_rank.toFixed(1)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ═══ Embeddings Tab ═══ */}
        {activeTab === 'embeddings' && (
          <div className="space-y-6">
            {!embeddingQuality ? (
              <div className="flex items-center justify-center h-48">
                <Loader size="md" label="Analyzing embeddings…" />
              </div>
            ) : (
              <>
                {/* Health status */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <StatCard
                    label="Health"
                    value={embeddingQuality.health_status}
                    icon={<ShieldAlert className="w-4 h-4" />}
                    color={
                      embeddingQuality.health_status === 'good' ? 'text-emerald-400' :
                      embeddingQuality.health_status === 'warning' ? 'text-amber-400' : 'text-red-400'
                    }
                  />
                  <StatCard label="Total Vectors" value={embeddingQuality.total_vectors} icon={<Layers className="w-4 h-4" />} />
                  <StatCard label="Dimension" value={embeddingQuality.embedding_dim ?? '—'} icon={<BarChart3 className="w-4 h-4" />} />
                  <StatCard label="Sample Size" value={embeddingQuality.sample_analyzed} icon={<Search className="w-4 h-4" />} />
                </div>

                {/* Issues */}
                {embeddingQuality.issues.length > 0 && (
                  <div className="bg-amber-500/5 rounded-xl border border-amber-500/20 p-5">
                    <h2 className="text-sm font-semibold text-amber-400 mb-2 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4" />
                      Issues Detected
                    </h2>
                    <ul className="space-y-1.5">
                      {embeddingQuality.issues.map((issue, i) => (
                        <li key={i} className="text-xs text-amber-300 flex items-start gap-2">
                          <span className="text-amber-500 mt-0.5">•</span>
                          {issue}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Norm + Cosine stats */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {embeddingQuality.norm_stats && (
                    <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5">
                      <h2 className="text-sm font-semibold text-slate-200 mb-3">L2 Norm Distribution</h2>
                      <div className="grid grid-cols-2 gap-4 text-xs">
                        <MetricRow label="Average" value={embeddingQuality.norm_stats.avg.toFixed(4)} />
                        <MetricRow label="Std Dev" value={embeddingQuality.norm_stats.std.toFixed(4)} />
                        <MetricRow label="Min" value={embeddingQuality.norm_stats.min.toFixed(4)} />
                        <MetricRow label="Max" value={embeddingQuality.norm_stats.max.toFixed(4)} />
                        <MetricRow
                          label="Outliers"
                          value={String(embeddingQuality.norm_stats.outlier_count)}
                          color={embeddingQuality.norm_stats.outlier_count > 0 ? 'text-amber-400' : 'text-slate-300'}
                        />
                      </div>
                    </div>
                  )}

                  {embeddingQuality.cosine_stats && (
                    <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5">
                      <h2 className="text-sm font-semibold text-slate-200 mb-3">Cosine Similarity</h2>
                      <div className="grid grid-cols-2 gap-4 text-xs">
                        <MetricRow label="Mean" value={embeddingQuality.cosine_stats.mean.toFixed(4)} />
                        <MetricRow label="Std Dev" value={embeddingQuality.cosine_stats.std.toFixed(4)} />
                        <MetricRow label="P25" value={embeddingQuality.cosine_stats.p25.toFixed(4)} />
                        <MetricRow label="P75" value={embeddingQuality.cosine_stats.p75.toFixed(4)} />
                        <MetricRow label="Min" value={embeddingQuality.cosine_stats.min.toFixed(4)} />
                        <MetricRow label="Max" value={embeddingQuality.cosine_stats.max.toFixed(4)} />
                      </div>
                    </div>
                  )}
                </div>

                {/* Per-source */}
                {embeddingQuality.per_source.length > 0 && (
                  <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 overflow-hidden">
                    <div className="px-5 py-3 border-b border-slate-800/60">
                      <h2 className="text-sm font-semibold text-slate-200">Per-Source Norm Stats</h2>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="border-b border-slate-800/40 text-slate-500">
                            <th className="px-5 py-2 text-left font-medium">Source</th>
                            <th className="px-3 py-2 text-right font-medium">Vectors</th>
                            <th className="px-3 py-2 text-right font-medium">Avg Norm</th>
                            <th className="px-3 py-2 text-right font-medium">Norm Std</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/30">
                          {embeddingQuality.per_source.map(src => (
                            <tr key={src.source} className="hover:bg-slate-800/20">
                              <td className="px-5 py-2 text-slate-300 truncate max-w-[200px]">{src.source}</td>
                              <td className="px-3 py-2 text-right text-slate-400">{src.count}</td>
                              <td className="px-3 py-2 text-right text-slate-300">{src.avg_norm.toFixed(4)}</td>
                              <td className="px-3 py-2 text-right text-slate-400">{src.norm_std.toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── Shared Sub-Components ─────────────────────────────────────────────────── */

function StatCard({
  label,
  value,
  icon,
  color = 'text-slate-100',
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color?: string;
}) {
  return (
    <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-4 hover:border-slate-700/60 transition-colors">
      <div className="flex items-center gap-2 mb-2 text-slate-400">
        {icon}
        <span className="text-[11px] font-medium">{label}</span>
      </div>
      <p className={`text-lg font-semibold ${color} capitalize`}>{value}</p>
    </div>
  );
}

function MetricRow({
  label,
  value,
  color = 'text-slate-300',
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div>
      <p className="text-slate-500 text-[11px]">{label}</p>
      <p className={`font-mono ${color}`}>{value}</p>
    </div>
  );
}
