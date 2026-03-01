import { useState } from 'react';
import type { DocumentSummary } from './KnowledgeBasePage';
import type { DocumentAnalytics } from '@/shared/types/api.types';
import { Search, FileText, Image, Volume2 } from 'lucide-react';

interface Props {
  documents: DocumentSummary[];
  analytics: Map<string, DocumentAnalytics>;
  onViewChunks: (source: string) => void;
  onDelete: (source: string) => void;
  onReindex: (source: string) => void;
  deletingDoc: string | null;
  reindexingDoc: string | null;
}

export default function DocumentTable({
  documents,
  analytics,
  onViewChunks,
  onDelete,
  onReindex,
  deletingDoc,
  reindexingDoc,
}: Props) {
  const [confirmSource, setConfirmSource] = useState<string | null>(null);
  const [search, setSearch] = useState('');

  const filtered = documents.filter((d) =>
    d.source.toLowerCase().includes(search.toLowerCase()) ||
    d.department.toLowerCase().includes(search.toLowerCase()) ||
    d.tags.some((t) => t.toLowerCase().includes(search.toLowerCase()))
  );

  const handleDeleteClick = (source: string) => {
    if (confirmSource === source) {
      onDelete(source);
      setConfirmSource(null);
    } else {
      setConfirmSource(source);
    }
  };

  return (
    <div>
      {/* Search */}
      <div className="mb-4">
        <div className="relative max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search by filename, department, or tags…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-sm bg-slate-800 border border-slate-700/80
                       rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none
                       focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500"
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-slate-800/80">
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-slate-400 uppercase bg-slate-800/50 tracking-wider">
            <tr>
              <th className="px-4 py-3">Document</th>
              <th className="px-4 py-3">Modality</th>
              <th className="px-4 py-3 text-center">Chunks</th>
              <th className="px-4 py-3 text-center">Tokens</th>
              <th className="px-4 py-3 text-center">Hits</th>
              <th className="px-4 py-3 text-center">Avg Rank</th>
              <th className="px-4 py-3">Last Queried</th>
              <th className="px-4 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {filtered.map((doc) => {
              const stats = analytics.get(doc.source);
              return (
              <tr
                key={doc.source}
                className="hover:bg-slate-800/40 transition-all duration-200"
              >
                {/* Document name */}
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <FileIcon modality={doc.modality} />
                    <span className="font-medium text-slate-200 truncate max-w-[240px]" title={doc.source}>
                      {doc.source}
                    </span>
                  </div>
                </td>

                {/* Modality badge */}
                <td className="px-4 py-3">
                  <ModalityBadge modality={doc.modality} />
                </td>

                {/* Chunk count */}
                <td className="px-4 py-3 text-center text-slate-300 tabular-nums">
                  {doc.chunk_count}
                </td>

                {/* Token count */}
                <td className="px-4 py-3 text-center text-slate-300 tabular-nums">
                  {doc.total_tokens.toLocaleString()}
                </td>

                {/* Retrieval hits */}
                <td className="px-4 py-3 text-center tabular-nums">
                  {stats ? (
                    <span className="text-amber-400 font-medium">{stats.retrieval_count}</span>
                  ) : (
                    <span className="text-slate-600">0</span>
                  )}
                </td>

                {/* Avg rank position */}
                <td className="px-4 py-3 text-center tabular-nums">
                  {stats && stats.avg_rank_position > 0 ? (
                    <span className={`font-medium ${stats.avg_rank_position <= 3 ? 'text-emerald-400' : stats.avg_rank_position <= 6 ? 'text-yellow-400' : 'text-slate-400'}`}>
                      #{stats.avg_rank_position.toFixed(1)}
                    </span>
                  ) : (
                    <span className="text-slate-600">—</span>
                  )}
                </td>

                {/* Last queried */}
                <td className="px-4 py-3 text-xs text-slate-400">
                  {stats && stats.last_queried > 0
                    ? new Date(stats.last_queried * 1000).toLocaleString()
                    : <span className="text-slate-600">Never</span>
                  }
                </td>

                {/* Actions */}
                <td className="px-4 py-3">
                  <div className="flex items-center justify-end gap-1.5">
                    <button
                      onClick={() => onViewChunks(doc.source)}
                      className="px-2 py-1.5 text-xs font-medium rounded-md
                                 bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 transition-colors"
                    >
                      Chunks
                    </button>
                    <button
                      onClick={() => onReindex(doc.source)}
                      disabled={reindexingDoc === doc.source}
                      className="px-2 py-1.5 text-xs font-medium rounded-md
                                 bg-cyan-600/15 text-cyan-400 hover:bg-cyan-600/25 transition-colors
                                 disabled:opacity-50"
                    >
                      {reindexingDoc === doc.source ? 'Re-indexing…' : 'Re-index'}
                    </button>
                    <button
                      onClick={() => handleDeleteClick(doc.source)}
                      onBlur={() => setConfirmSource(null)}
                      disabled={deletingDoc === doc.source}
                      className={`px-2 py-1.5 text-xs font-medium rounded-md transition-colors
                        ${confirmSource === doc.source
                          ? 'bg-red-600 text-white hover:bg-red-700'
                          : 'bg-red-600/10 text-red-400 hover:bg-red-600/20'}
                        disabled:opacity-50`}
                    >
                      {deletingDoc === doc.source
                        ? 'Deleting…'
                        : confirmSource === doc.source
                          ? 'Confirm?'
                          : 'Delete'}
                    </button>
                  </div>
                </td>
              </tr>
              );
            })}

            {filtered.length === 0 && (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-slate-500">
                  {search ? 'No documents match your search.' : 'No documents indexed yet.'}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Summary bar */}
      {filtered.length > 0 && (
        <div className="mt-3 text-xs text-slate-500">
          Showing {filtered.length} of {documents.length} document{documents.length !== 1 ? 's' : ''}
        </div>
      )}
    </div>
  );
}

/* ── Helpers ──────────────────────────────────────────────── */

function ModalityBadge({ modality }: { modality: string }) {
  const colors: Record<string, string> = {
    document: 'bg-blue-600/20 text-blue-400',
    image: 'bg-emerald-600/20 text-emerald-400',
    audio: 'bg-purple-600/20 text-purple-400',
  };
  return (
    <span className={`px-2 py-0.5 text-xs rounded-full font-medium ${colors[modality] || 'bg-slate-700 text-slate-300'}`}>
      {modality}
    </span>
  );
}

function FileIcon({ modality }: { modality: string }) {
  const cls = 'w-4 h-4 flex-shrink-0';
  if (modality === 'audio') return <Volume2 className={`${cls} text-purple-400`} />;
  if (modality === 'image') return <Image className={`${cls} text-emerald-400`} />;
  return <FileText className={`${cls} text-blue-400`} />;
}
