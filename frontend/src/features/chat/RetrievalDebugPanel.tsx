import { useState } from 'react';
import type { DebugInfo, DebugResult, EffectiveParams, EntityInfo, ImageBranchInfo } from '@/shared/types/api.types';

interface Props {
  debugInfo: DebugInfo | null;
  isOpen: boolean;
  onClose: () => void;
}

const STAGES = [
  { key: 'vector_results', label: 'Vector Top-K', color: 'text-blue-400', bg: 'bg-blue-500/10' },
  { key: 'bm25_results', label: 'BM25 Top-K', color: 'text-amber-400', bg: 'bg-amber-500/10' },
  { key: 'rrf_fused', label: 'RRF Merged', color: 'text-purple-400', bg: 'bg-purple-500/10' },
  { key: 'reranked', label: 'Reranked Final', color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
  { key: 'dropped', label: 'Dropped', color: 'text-red-400', bg: 'bg-red-500/10' },
] as const;

type StageKey = typeof STAGES[number]['key'];

/* ── Origin badge ─────────────────────────────────────────── */

const ORIGIN_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  vector: { bg: 'bg-blue-500/20', text: 'text-blue-300', label: 'VEC' },
  bm25:   { bg: 'bg-amber-500/20', text: 'text-amber-300', label: 'BM25' },
  both:   { bg: 'bg-purple-500/20', text: 'text-purple-300', label: 'BOTH' },
  entity: { bg: 'bg-cyan-500/20', text: 'text-cyan-300', label: 'ENT' },
};

function OriginBadge({ origin }: { origin?: string | null }) {
  if (!origin) return <span className="text-slate-600">—</span>;
  const style = ORIGIN_STYLES[origin] || { bg: 'bg-slate-700', text: 'text-slate-400', label: origin };
  return (
    <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider ${style.bg} ${style.text}`}>
      {style.label}
    </span>
  );
}

/* ── Effective Params bar ─────────────────────────────────── */

function EffectiveParamsBar({ params }: { params?: EffectiveParams | null }) {
  if (!params) return null;
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 px-4 py-1.5 border-b border-slate-800/50 text-[10px] text-slate-500">
      <span>RRF k=<b className="text-slate-300">{params.rrf_k}</b></span>
      <span>Rerank top=<b className="text-slate-300">{params.rerank_count}</b></span>
      <span>Threshold=<b className="text-slate-300">{params.rerank_threshold}</b></span>
      <span>Min results=<b className="text-slate-300">{params.rerank_min_results}</b></span>
      <span>Vec top-k=<b className="text-slate-300">{params.vector_top_k}</b></span>
      <span>BM25 top-k=<b className="text-slate-300">{params.bm25_top_k}</b></span>
    </div>
  );
}

/* ── Entity Pipeline Visualization ────────────────────────── */

function EntityPipelineBar({ entityInfo }: { entityInfo?: EntityInfo | null }) {
  if (!entityInfo || entityInfo.extracted_entities.length === 0) return null;

  return (
    <div className="px-4 py-2 border-b border-slate-800/50">
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-[10px] font-semibold text-cyan-400 uppercase tracking-wider">Entity Pipeline</span>
        <span className="text-[10px] text-slate-600">
          {entityInfo.extracted_entities.length} entities → {entityInfo.injected_count} chunks injected
        </span>
      </div>

      {/* Flow visualization: Query → Entities → Chunks */}
      <div className="flex items-center gap-1.5 text-[10px]">
        <span className="text-slate-500">Query</span>
        <Arrow />
        <div className="flex flex-wrap gap-1">
          {entityInfo.extracted_entities.map((e, i) => (
            <span
              key={i}
              className="px-1.5 py-0.5 rounded bg-cyan-500/15 text-cyan-300 font-medium"
            >
              {e}
            </span>
          ))}
        </div>
        <Arrow />
        <div className="flex flex-wrap gap-1">
          {entityInfo.injected_chunks.map((c, i) => (
            <span
              key={i}
              className="px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400/70 font-mono"
              title={`Matched "${c.entity_match}" in ${c.source}`}
            >
              {c.chunk_id.slice(0, 8)}…
            </span>
          ))}
          {entityInfo.injected_count === 0 && (
            <span className="text-slate-600 italic">no matches</span>
          )}
        </div>
      </div>
    </div>
  );
}

function Arrow() {
  return (
    <svg className="w-3 h-3 text-slate-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
    </svg>
  );
}

/* ── Retrieval Mode Badge ─────────────────────────────────── */

function RetrievalModeBadge({ mode }: { mode?: string | null }) {
  if (!mode || mode === 'text') return null;
  const styles: Record<string, { bg: string; text: string; label: string }> = {
    image: { bg: 'bg-pink-500/20', text: 'text-pink-300', label: '🖼️ Image Query' },
    audio: { bg: 'bg-green-500/20', text: 'text-green-300', label: '🎙️ Audio Query' },
  };
  const s = styles[mode] || { bg: 'bg-slate-700', text: 'text-slate-300', label: mode };
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-semibold ${s.bg} ${s.text}`}>
      {s.label}
    </span>
  );
}

/* ── Image Branch Info ────────────────────────────────────── */

function ImageBranchBar({ info }: { info?: ImageBranchInfo | null }) {
  if (!info || (info.image_chunk_count === 0 && info.linked_text_count === 0)) return null;

  const total = info.image_chunk_count + info.linked_text_count;
  const imagePct = total > 0 ? ((info.image_chunk_count / total) * 100).toFixed(0) : '0';
  const textPct = total > 0 ? ((info.linked_text_count / total) * 100).toFixed(0) : '0';

  return (
    <div className="px-4 py-2 border-b border-slate-800/50">
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-[10px] font-semibold text-pink-400 uppercase tracking-wider">Image Branch</span>
        <span className="text-[10px] text-slate-600">
          {info.image_chunk_count} image + {info.linked_text_count} linked → {info.image_branch_contribution} contributed
        </span>
      </div>

      {/* Contribution bar */}
      <div className="flex h-2 rounded-full overflow-hidden bg-slate-800">
        <div
          className="bg-pink-500/60 transition-all"
          style={{ width: `${imagePct}%` }}
          title={`Image: ${imagePct}%`}
        />
        <div
          className="bg-blue-500/60 transition-all"
          style={{ width: `${textPct}%` }}
          title={`Linked text: ${textPct}%`}
        />
      </div>

      <div className="flex justify-between mt-1 text-[9px] text-slate-600">
        <span>Image: {imagePct}%</span>
        <span>Linked text: {textPct}%</span>
      </div>
    </div>
  );
}

/* ── Main component ───────────────────────────────────────── */

export default function RetrievalDebugPanel({ debugInfo, isOpen, onClose }: Props) {
  const [activeStage, setActiveStage] = useState<StageKey>('reranked');

  if (!isOpen || !debugInfo) return null;

  const items = debugInfo[activeStage] || [];
  const activeConf = STAGES.find((s) => s.key === activeStage)!;
  const showOrigin = activeStage === 'rrf_fused' || activeStage === 'reranked' || activeStage === 'dropped';
  const showSourceRanks = showOrigin;

  return (
    <div className="border-t border-slate-800 bg-slate-950/80 backdrop-blur-sm">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <BugIcon className="w-4 h-4 text-amber-400" />
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Retrieval Debug
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors"
        >
          <CloseIcon className="w-4 h-4" />
        </button>
      </div>

      {/* Effective parameters */}
      <EffectiveParamsBar params={debugInfo.effective_params} />

      {/* Retrieval mode badge */}
      {debugInfo.retrieval_mode && debugInfo.retrieval_mode !== 'text' && (
        <div className="px-4 py-1.5 border-b border-slate-800/50">
          <RetrievalModeBadge mode={debugInfo.retrieval_mode} />
        </div>
      )}

      {/* Entity injection pipeline */}
      <EntityPipelineBar entityInfo={debugInfo.entity_info} />

      {/* Image branch contribution */}
      <ImageBranchBar info={debugInfo.image_branch_info} />

      {/* Stage tabs */}
      <div className="flex gap-1 px-4 py-2 overflow-x-auto">
        {STAGES.map((stage) => {
          const count = (debugInfo[stage.key] || []).length;
          const isActive = activeStage === stage.key;
          return (
            <button
              key={stage.key}
              onClick={() => setActiveStage(stage.key)}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg
                         whitespace-nowrap transition-colors
                         ${isActive
                           ? `${stage.bg} ${stage.color} ring-1 ring-current/20`
                           : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/50'}`}
            >
              {stage.label}
              <span className={`tabular-nums ${isActive ? 'opacity-80' : 'opacity-50'}`}>
                ({count})
              </span>
            </button>
          );
        })}
      </div>

      {/* Results table */}
      <div className="max-h-64 overflow-y-auto overflow-x-auto px-4 pb-3
                       scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
        {items.length === 0 ? (
          <p className="text-xs text-slate-600 py-4 text-center">No results at this stage.</p>
        ) : (
          <table className="w-full text-xs text-left">
            <thead className="text-[10px] text-slate-500 uppercase sticky top-0 bg-slate-950/95">
              <tr>
                <th className="py-1.5 pr-3">#</th>
                {showOrigin && <th className="py-1.5 pr-3">Origin</th>}
                <th className="py-1.5 pr-3">Chunk ID</th>
                <th className="py-1.5 pr-3">Source</th>
                <th className="py-1.5 pr-3 text-right">Page</th>
                <th className="py-1.5 pr-3 text-right">Score</th>
                {activeStage !== 'vector_results' && activeStage !== 'bm25_results' && (
                  <th className="py-1.5 pr-3 text-right">Rank</th>
                )}
                {showSourceRanks && <th className="py-1.5 pr-3 text-right">Vec#</th>}
                {showSourceRanks && <th className="py-1.5 pr-3 text-right">BM25#</th>}
                {(activeStage === 'reranked' || activeStage === 'dropped') && (
                  <th className="py-1.5 text-right">Reranker</th>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {items.map((item, idx) => (
                <ResultRow
                  key={`${item.chunk_id}-${idx}`}
                  item={item}
                  index={idx}
                  stageKey={activeStage}
                  color={activeConf.color}
                  showOrigin={showOrigin}
                  showSourceRanks={showSourceRanks}
                />
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

/* ── Result Row ───────────────────────────────────────────── */

function ResultRow({
  item,
  index,
  stageKey,
  color,
  showOrigin,
  showSourceRanks,
}: {
  item: DebugResult;
  index: number;
  stageKey: StageKey;
  color: string;
  showOrigin: boolean;
  showSourceRanks: boolean;
}) {
  return (
    <tr className="hover:bg-slate-800/30 transition-colors">
      <td className={`py-1.5 pr-3 tabular-nums font-medium ${color}`}>{index + 1}</td>
      {showOrigin && (
        <td className="py-1.5 pr-3">
          <OriginBadge origin={item.origin} />
        </td>
      )}
      <td className="py-1.5 pr-3 font-mono text-slate-400" title={item.chunk_id}>
        {item.chunk_id}
      </td>
      <td className="py-1.5 pr-3 text-slate-300 max-w-[180px] truncate" title={item.source}>
        {item.source}
      </td>
      <td className="py-1.5 pr-3 text-right tabular-nums text-slate-400">
        {item.page != null ? item.page : '—'}
      </td>
      <td className="py-1.5 pr-3 text-right tabular-nums text-slate-300">
        {item.score.toFixed(4)}
      </td>
      {stageKey !== 'vector_results' && stageKey !== 'bm25_results' && (
        <td className="py-1.5 pr-3 text-right tabular-nums text-slate-400">
          {item.rank != null ? item.rank : '—'}
        </td>
      )}
      {showSourceRanks && (
        <td className="py-1.5 pr-3 text-right tabular-nums text-blue-400/70">
          {item.vector_rank != null ? item.vector_rank : '—'}
        </td>
      )}
      {showSourceRanks && (
        <td className="py-1.5 pr-3 text-right tabular-nums text-amber-400/70">
          {item.bm25_rank != null ? item.bm25_rank : '—'}
        </td>
      )}
      {(stageKey === 'reranked' || stageKey === 'dropped') && (
        <td className="py-1.5 text-right tabular-nums text-slate-300">
          {item.reranker_score != null ? item.reranker_score.toFixed(4) : '—'}
        </td>
      )}
    </tr>
  );
}

/* ── Icons ────────────────────────────────────────────────── */

function BugIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 12.75c1.148 0 2.278.08 3.383.237 1.037.146 1.866.966 1.866 2.013 0 3.728-2.35 6.75-5.25 6.75S6.75 18.728 6.75 15c0-1.046.83-1.867 1.866-2.013A24.204 24.204 0 0 1 12 12.75Zm0 0c2.883 0 5.647.508 8.207 1.44a23.91 23.91 0 0 1-3.48 2.31m-9.454 0a23.91 23.91 0 0 1-3.48-2.31C6.353 13.258 9.117 12.75 12 12.75Zm0 0V8.25m4.5 2.25a4.5 4.5 0 1 0-9 0v2.25m9-2.25h3M7.5 10.5H4.5m3-3V4.875A2.625 2.625 0 0 1 10.125 2.25h3.75A2.625 2.625 0 0 1 16.5 4.875V7.5" />
    </svg>
  );
}

function CloseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18 18 6M6 6l12 12" />
    </svg>
  );
}
