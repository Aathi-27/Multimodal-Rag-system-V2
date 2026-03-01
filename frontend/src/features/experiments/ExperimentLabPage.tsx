import { useState } from 'react';
import api from '@/shared/api/axios';
import Loader from '@/shared/components/Loader';
import {
  ArrowLeftRight,
  BarChart3,
  FlaskConical,
  Layers,
  Play,
  Plus,
  Trash2,
  Zap,
} from 'lucide-react';
import type {
  BatchEvalResponse,
  CompareResponse,
  ExperimentParams,
} from '@/shared/types/api.types';

/* ── Parameter field definition ──────────────────────────────────────────── */
const PARAM_FIELDS: Array<{ key: keyof ExperimentParams; label: string; type: 'int' | 'float'; default: number }> = [
  { key: 'vector_top_k', label: 'Vector Top-K', type: 'int', default: 50 },
  { key: 'bm25_top_k', label: 'BM25 Top-K', type: 'int', default: 50 },
  { key: 'rrf_k', label: 'RRF K', type: 'int', default: 60 },
  { key: 'rerank_count', label: 'Rerank Count', type: 'int', default: 15 },
  { key: 'rerank_threshold', label: 'Rerank Threshold', type: 'float', default: 0.15 },
  { key: 'rerank_min_results', label: 'Min Results', type: 'int', default: 5 },
];

/* ── Default params ──────────────────────────────────────────────────────── */
const DEFAULT_PARAMS: ExperimentParams = {
  vector_top_k: 50,
  bm25_top_k: 50,
  rrf_k: 60,
  rerank_count: 15,
  rerank_threshold: 0.15,
  rerank_min_results: 5,
};

export default function ExperimentLabPage() {
  const [activeTab, setActiveTab] = useState<'compare' | 'batch'>('compare');

  return (
    <div className="h-full overflow-y-auto">
      <div className="max-w-[1400px] mx-auto p-6 space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-xl font-semibold text-slate-100 tracking-tight flex items-center gap-2">
            <FlaskConical className="w-5 h-5 text-purple-400" />
            Experiment Lab
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Every version is an experiment. Compare, evaluate, optimize.
          </p>
        </div>

        {/* Tab bar */}
        <div className="flex gap-1 bg-slate-900/60 rounded-lg p-1 w-fit border border-slate-800/60">
          {(['compare', 'batch'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-1.5 rounded-md text-xs font-medium transition-all duration-200
                ${activeTab === tab
                  ? 'bg-slate-800 text-purple-400 shadow-sm'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                }`}
            >
              {tab === 'compare' ? 'Side-by-Side' : 'Batch Evaluation'}
            </button>
          ))}
        </div>

        {activeTab === 'compare' && <ComparePanel />}
        {activeTab === 'batch' && <BatchPanel />}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ── Compare Panel ────────────────────────────────────────────────────────── */
/* ═══════════════════════════════════════════════════════════════════════════ */

function ComparePanel() {
  const [query, setQuery] = useState('');
  const [paramsA, setParamsA] = useState<ExperimentParams>({ ...DEFAULT_PARAMS });
  const [paramsB, setParamsB] = useState<ExperimentParams>({ ...DEFAULT_PARAMS, rerank_threshold: 0.1 });
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const runComparison = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const res = await api.post<CompareResponse>('/experiments/compare', {
        query: query.trim(),
        params_a: paramsA,
        params_b: paramsB,
        label_a: 'A',
        label_b: 'B',
      });
      setResult(res.data);
    } catch (err) {
      console.error('Comparison failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Query + params */}
      <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5 space-y-4">
        <div>
          <label className="text-xs font-medium text-slate-400 block mb-1.5">Query</label>
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && runComparison()}
            placeholder="Enter a query to compare retrieval with different parameters…"
            className="w-full bg-slate-800/60 border border-slate-700/60 rounded-lg px-3.5 py-2 text-sm
                       text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/40
                       focus:border-blue-500/50 transition-all"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ParamEditor label="Experiment A" params={paramsA} onChange={setParamsA} accent="blue" />
          <ParamEditor label="Experiment B" params={paramsB} onChange={setParamsB} accent="purple" />
        </div>

        <button
          onClick={runComparison}
          disabled={loading || !query.trim()}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium
                     rounded-lg transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed
                     hover:scale-[1.02] active:scale-95"
        >
          {loading ? <Loader size="sm" /> : <Play className="w-4 h-4" />}
          {loading ? 'Running…' : 'Run Comparison'}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6 animate-fade-in-up">
          {/* Analysis summary cards */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <MiniCard label="Overlap" value={result.analysis.overlap_count} color="text-emerald-400" />
            <MiniCard label="Unique to A" value={result.analysis.unique_to_a} color="text-blue-400" />
            <MiniCard label="Unique to B" value={result.analysis.unique_to_b} color="text-purple-400" />
            <MiniCard
              label="Jaccard"
              value={result.analysis.jaccard_similarity.toFixed(3)}
              color={result.analysis.jaccard_similarity > 0.5 ? 'text-emerald-400' : 'text-amber-400'}
            />
            <MiniCard
              label="Δ Latency"
              value={`${result.analysis.latency_diff > 0 ? '+' : ''}${(result.analysis.latency_diff * 1000).toFixed(0)}ms`}
              color={Math.abs(result.analysis.latency_diff) < 0.05 ? 'text-slate-300' : 'text-amber-400'}
            />
          </div>

          {/* Side-by-side chunks */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChunkList
              label={`A: ${result.experiment_a.label}`}
              chunks={result.experiment_a.result.chunks}
              latency={result.experiment_a.result.retrieval_latency}
              otherIds={new Set(result.experiment_b.result.chunks.map(c => c.chunk_id))}
              accent="blue"
            />
            <ChunkList
              label={`B: ${result.experiment_b.label}`}
              chunks={result.experiment_b.result.chunks}
              latency={result.experiment_b.result.retrieval_latency}
              otherIds={new Set(result.experiment_a.result.chunks.map(c => c.chunk_id))}
              accent="purple"
            />
          </div>

          {/* Rank differences */}
          {result.analysis.rank_differences.length > 0 && (
            <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5">
              <h3 className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
                <ArrowLeftRight className="w-4 h-4 text-amber-400" />
                Rank Differences (overlapping chunks)
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-800/40 text-slate-500">
                      <th className="px-3 py-2 text-left font-medium">Chunk ID</th>
                      <th className="px-3 py-2 text-right font-medium">Rank A</th>
                      <th className="px-3 py-2 text-right font-medium">Rank B</th>
                      <th className="px-3 py-2 text-right font-medium">Δ Rank</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/30">
                    {result.analysis.rank_differences.map(rd => (
                      <tr key={rd.chunk_id} className="hover:bg-slate-800/20">
                        <td className="px-3 py-2 font-mono text-slate-400">{rd.chunk_id.slice(0, 12)}</td>
                        <td className="px-3 py-2 text-right text-blue-400">#{rd.rank_a}</td>
                        <td className="px-3 py-2 text-right text-purple-400">#{rd.rank_b}</td>
                        <td className={`px-3 py-2 text-right font-medium ${
                          rd.rank_diff === 0 ? 'text-slate-500' :
                          rd.rank_diff > 0 ? 'text-red-400' : 'text-emerald-400'
                        }`}>
                          {rd.rank_diff > 0 ? '+' : ''}{rd.rank_diff}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ── Batch Evaluation Panel ───────────────────────────────────────────────── */
/* ═══════════════════════════════════════════════════════════════════════════ */

function BatchPanel() {
  const [queries, setQueries] = useState<Array<{ query: string; ground_truth_chunk_ids: string[] }>>([
    { query: '', ground_truth_chunk_ids: [] },
  ]);
  const [overrides, setOverrides] = useState<ExperimentParams>({ ...DEFAULT_PARAMS });
  const [result, setResult] = useState<BatchEvalResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const addQuery = () => setQueries(prev => [...prev, { query: '', ground_truth_chunk_ids: [] }]);
  const removeQuery = (idx: number) => setQueries(prev => prev.filter((_, i) => i !== idx));
  const updateQuery = (idx: number, val: string) =>
    setQueries(prev => prev.map((q, i) => (i === idx ? { ...q, query: val } : q)));

  const runEval = async () => {
    const validQueries = queries.filter(q => q.query.trim());
    if (!validQueries.length) return;

    setLoading(true);
    setResult(null);
    try {
      const res = await api.post<BatchEvalResponse>('/experiments/batch-evaluate', {
        queries: validQueries,
        overrides,
      });
      setResult(res.data);
    } catch (err) {
      console.error('Batch eval failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Query list */}
      <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-slate-200">Test Queries</h3>
          <button
            onClick={addQuery}
            className="flex items-center gap-1.5 text-xs text-blue-400 hover:text-blue-300 transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            Add Query
          </button>
        </div>

        <div className="space-y-2">
          {queries.map((q, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <span className="text-xs text-slate-500 w-5 text-right">{idx + 1}</span>
              <input
                type="text"
                value={q.query}
                onChange={e => updateQuery(idx, e.target.value)}
                placeholder="Enter test query…"
                className="flex-1 bg-slate-800/60 border border-slate-700/60 rounded-lg px-3 py-1.5 text-sm
                           text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2
                           focus:ring-purple-500/40 transition-all"
              />
              {queries.length > 1 && (
                <button
                  onClick={() => removeQuery(idx)}
                  className="p-1.5 text-slate-500 hover:text-red-400 transition-colors"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          ))}
        </div>

        {/* Overrides */}
        <ParamEditor label="Parameters" params={overrides} onChange={setOverrides} accent="purple" />

        <button
          onClick={runEval}
          disabled={loading || queries.every(q => !q.query.trim())}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-500 text-white text-sm font-medium
                     rounded-lg transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed
                     hover:scale-[1.02] active:scale-95"
        >
          {loading ? <Loader size="sm" /> : <Zap className="w-4 h-4" />}
          {loading ? 'Evaluating…' : `Evaluate ${queries.filter(q => q.query.trim()).length} Queries`}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6 animate-fade-in-up">
          {/* Metric cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <MiniCard
              label="Avg Recall@5"
              value={result.metrics.avg_recall_at_5 != null ? result.metrics.avg_recall_at_5.toFixed(3) : 'N/A'}
              color="text-emerald-400"
            />
            <MiniCard
              label="Avg MRR"
              value={result.metrics.avg_mrr != null ? result.metrics.avg_mrr.toFixed(3) : 'N/A'}
              color="text-blue-400"
            />
            <MiniCard
              label="Avg Latency"
              value={`${(result.metrics.avg_latency * 1000).toFixed(0)}ms`}
              color="text-slate-200"
            />
            <MiniCard
              label="P95 Latency"
              value={`${(result.metrics.p95_latency * 1000).toFixed(0)}ms`}
              color={result.metrics.p95_latency > 1 ? 'text-red-400' : 'text-slate-200'}
            />
          </div>

          {/* Per-query table */}
          <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 overflow-hidden">
            <div className="px-5 py-3 border-b border-slate-800/60">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-purple-400" />
                Per-Query Results ({result.total_queries})
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-800/40 text-slate-500">
                    <th className="px-5 py-2 text-left font-medium">#</th>
                    <th className="px-3 py-2 text-left font-medium">Query</th>
                    <th className="px-3 py-2 text-right font-medium">Chunks</th>
                    <th className="px-3 py-2 text-right font-medium">Latency</th>
                    <th className="px-3 py-2 text-right font-medium">Recall@5</th>
                    <th className="px-3 py-2 text-right font-medium">MRR</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/30">
                  {result.per_query.map((pq, i) => (
                    <tr key={i} className="hover:bg-slate-800/20">
                      <td className="px-5 py-2 text-slate-500">{i + 1}</td>
                      <td className="px-3 py-2 text-slate-300 truncate max-w-[250px]">{String(pq.query ?? '')}</td>
                      <td className="px-3 py-2 text-right text-slate-400">{String(pq.chunk_count ?? 0)}</td>
                      <td className="px-3 py-2 text-right text-slate-400">
                        {((Number(pq.retrieval_latency) || 0) * 1000).toFixed(0)}ms
                      </td>
                      <td className="px-3 py-2 text-right text-emerald-400">
                        {pq.recall_at_5 != null ? Number(pq.recall_at_5).toFixed(3) : '—'}
                      </td>
                      <td className="px-3 py-2 text-right text-blue-400">
                        {pq.mrr != null ? Number(pq.mrr).toFixed(3) : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ── Shared Sub-Components ────────────────────────────────────────────────── */
/* ═══════════════════════════════════════════════════════════════════════════ */

function ParamEditor({
  label,
  params,
  onChange,
  accent = 'blue',
}: {
  label: string;
  params: ExperimentParams;
  onChange: (p: ExperimentParams) => void;
  accent?: 'blue' | 'purple';
}) {
  const borderColor = accent === 'blue' ? 'border-blue-500/30' : 'border-purple-500/30';
  const labelColor = accent === 'blue' ? 'text-blue-400' : 'text-purple-400';

  const updateParam = (key: keyof ExperimentParams, raw: string) => {
    const field = PARAM_FIELDS.find(f => f.key === key);
    if (!field) return;
    const val = field.type === 'float' ? parseFloat(raw) : parseInt(raw, 10);
    if (isNaN(val)) return;
    onChange({ ...params, [key]: val });
  };

  return (
    <div className={`border ${borderColor} rounded-lg p-3 bg-slate-800/20`}>
      <p className={`text-xs font-semibold ${labelColor} mb-2`}>{label}</p>
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
        {PARAM_FIELDS.map(f => (
          <div key={f.key}>
            <label className="text-[10px] text-slate-500 block mb-0.5">{f.label}</label>
            <input
              type="number"
              step={f.type === 'float' ? 0.01 : 1}
              value={params[f.key] ?? f.default}
              onChange={e => updateParam(f.key, e.target.value)}
              className="w-full bg-slate-900/60 border border-slate-700/40 rounded px-2 py-1 text-xs
                         text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500/40"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function ChunkList({
  label,
  chunks,
  latency,
  otherIds,
  accent = 'blue',
}: {
  label: string;
  chunks: Array<{ chunk_id: string; source: string; reranker_score: number; rank: number; text_preview: string }>;
  latency: number;
  otherIds: Set<string>;
  accent?: 'blue' | 'purple';
}) {
  const borderColor = accent === 'blue' ? 'border-blue-500/30' : 'border-purple-500/30';
  const headColor = accent === 'blue' ? 'text-blue-400' : 'text-purple-400';

  return (
    <div className={`bg-slate-900/60 rounded-xl border ${borderColor} overflow-hidden`}>
      <div className="px-4 py-3 border-b border-slate-800/60 flex items-center justify-between">
        <h3 className={`text-sm font-semibold ${headColor} flex items-center gap-2`}>
          <Layers className="w-4 h-4" />
          {label}
        </h3>
        <span className="text-[11px] text-slate-500">{chunks.length} chunks · {(latency * 1000).toFixed(0)}ms</span>
      </div>
      <div className="max-h-[400px] overflow-y-auto divide-y divide-slate-800/30">
        {chunks.map(c => {
          const inOther = otherIds.has(c.chunk_id);
          return (
            <div key={c.chunk_id} className="px-4 py-2.5 hover:bg-slate-800/20 transition-colors">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-500">#{c.rank}</span>
                <span className="text-xs font-mono text-slate-400">{c.chunk_id.slice(0, 10)}</span>
                {inOther ? (
                  <span className="text-[9px] px-1.5 py-0.5 bg-emerald-500/15 text-emerald-400 rounded">overlap</span>
                ) : (
                  <span className="text-[9px] px-1.5 py-0.5 bg-amber-500/15 text-amber-400 rounded">unique</span>
                )}
                <span className="text-[11px] text-slate-500 ml-auto">{c.reranker_score.toFixed(3)}</span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1 truncate">{c.source}</p>
              {c.text_preview && (
                <p className="text-[11px] text-slate-600 mt-0.5 truncate">{c.text_preview}</p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function MiniCard({
  label,
  value,
  color = 'text-slate-200',
}: {
  label: string;
  value: string | number;
  color?: string;
}) {
  return (
    <div className="bg-slate-900/60 rounded-xl border border-slate-800/60 p-3 hover:border-slate-700/60 transition-colors">
      <p className="text-[11px] text-slate-500 mb-0.5">{label}</p>
      <p className={`text-base font-semibold ${color}`}>{value}</p>
    </div>
  );
}
