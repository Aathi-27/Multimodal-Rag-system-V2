import { useCallback, useEffect, useState } from 'react';
import api from '@/shared/api/axios';
import type { DocumentAnalytics } from '@/shared/types/api.types';
import { RefreshCw, Database, FileText, Layers, Hash } from 'lucide-react';
import Loader from '@/shared/components/Loader';
import ErrorMessage from '@/shared/components/ErrorMessage';
import { SkeletonCard, SkeletonRow } from '@/shared/components/Skeleton';
import DocumentTable from './DocumentTable';
import ChunkViewer from './ChunkViewer';

/* ── Types ─────────────────────────────────────────────────── */

export interface DocumentSummary {
  source: string;
  modality: string;
  department: string;
  tags: string[];
  chunk_count: number;
  total_tokens: number;
  upload_id: string;
}

export interface ChunkDetail {
  chunk_id: string;
  text: string;
  chunk_index: number;
  page_start: number | null;
  token_count: number;
}

/* ── Component ─────────────────────────────────────────────── */

export default function KnowledgeBasePage() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [analytics, setAnalytics] = useState<Map<string, DocumentAnalytics>>(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Chunk viewer state
  const [selectedDoc, setSelectedDoc] = useState<string | null>(null);
  const [chunks, setChunks] = useState<ChunkDetail[]>([]);
  const [chunksLoading, setChunksLoading] = useState(false);

  // Delete confirmation state
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);

  // Re-index state
  const [reindexingDoc, setReindexingDoc] = useState<string | null>(null);

  /* ── Fetch documents + analytics ────────────────────────── */
  const fetchDocuments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [docRes, analyticsRes] = await Promise.allSettled([
        api.get<{ documents: DocumentSummary[]; total: number }>('/documents'),
        api.get<{ documents: DocumentAnalytics[] }>('/analytics'),
      ]);
      if (docRes.status === 'fulfilled') {
        setDocuments(docRes.value.data.documents);
      } else {
        throw docRes.reason;
      }
      if (analyticsRes.status === 'fulfilled') {
        const map = new Map<string, DocumentAnalytics>();
        for (const a of analyticsRes.value.data.documents) {
          map.set(a.source, a);
        }
        setAnalytics(map);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  /* ── View chunks ────────────────────────────────────────── */
  const handleViewChunks = async (source: string) => {
    setSelectedDoc(source);
    setChunksLoading(true);
    try {
      const res = await api.get<{ source: string; chunks: ChunkDetail[]; total_chunks: number }>(
        `/documents/${encodeURIComponent(source)}/chunks`,
      );
      setChunks(res.data.chunks);
    } catch {
      setChunks([]);
    } finally {
      setChunksLoading(false);
    }
  };

  const handleCloseChunks = () => {
    setSelectedDoc(null);
    setChunks([]);
  };

  /* ── Delete document ────────────────────────────────────── */
  const handleDelete = async (source: string) => {
    setDeletingDoc(source);
    try {
      await api.delete(`/documents/${encodeURIComponent(source)}`);
      // Remove from local state immediately
      setDocuments((prev) => prev.filter((d) => d.source !== source));
      setAnalytics((prev) => { const next = new Map(prev); next.delete(source); return next; });
      // Close chunk viewer if this doc was open
      if (selectedDoc === source) handleCloseChunks();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to delete document');
    } finally {
      setDeletingDoc(null);
    }
  };

  /* ── Re-index document ──────────────────────────────────── */
  const handleReindex = async (source: string) => {
    setReindexingDoc(source);
    try {
      await api.post(`/documents/${encodeURIComponent(source)}/reindex`);
      // Refresh list after a short delay to let background task start
      setTimeout(() => {
        fetchDocuments();
        setReindexingDoc(null);
      }, 2000);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to re-index document');
      setReindexingDoc(null);
    }
  };

  /* ── Derived stats ────────────────────────────────────────── */
  const totalChunks = documents.reduce((sum, d) => sum + d.chunk_count, 0);
  const totalTokens = documents.reduce((sum, d) => sum + d.total_tokens, 0);
  const avgTokens = totalChunks > 0 ? Math.round(totalTokens / totalChunks) : 0;

  /* ── Render ─────────────────────────────────────────────── */
  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800/80 bg-slate-900/40 flex-shrink-0">
        <div>
          <h1 className="text-xl font-semibold text-slate-100">Knowledge Base</h1>
          <p className="text-xs text-slate-500 mt-0.5">
            Manage your indexed documents, chunks, and analytics
          </p>
        </div>
        <button
          onClick={fetchDocuments}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-2 text-sm rounded-lg
                     bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white
                     disabled:opacity-50 transition-colors border border-slate-700/50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {loading && !documents.length ? (
          <div className="space-y-5">
            {/* Skeleton stat cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)}
            </div>
            {/* Skeleton table rows */}
            <div className="rounded-xl border border-slate-800/80 overflow-hidden">
              <table className="w-full">
                <tbody>
                  {Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} cols={6} />)}
                </tbody>
              </table>
            </div>
          </div>
        ) : error ? (
          <ErrorMessage message={error} onRetry={fetchDocuments} />
        ) : documents.length === 0 ? (
          <EmptyState />
        ) : (
          <>
            {/* Stats summary row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
              <StatCard icon={FileText} label="Documents" value={documents.length} />
              <StatCard icon={Layers} label="Total Chunks" value={totalChunks} />
              <StatCard icon={Hash} label="Total Tokens" value={totalTokens.toLocaleString()} />
              <StatCard icon={Hash} label="Avg Tokens / Chunk" value={avgTokens} />
            </div>

            <DocumentTable
              documents={documents}
              analytics={analytics}
              onViewChunks={handleViewChunks}
              onDelete={handleDelete}
              onReindex={handleReindex}
              deletingDoc={deletingDoc}
              reindexingDoc={reindexingDoc}
            />
          </>
        )}
      </div>

      {/* Chunk Viewer Slide-over */}
      {selectedDoc && (
        <ChunkViewer
          source={selectedDoc}
          chunks={chunks}
          loading={chunksLoading}
          onClose={handleCloseChunks}
        />
      )}
    </div>
  );
}

/* ── Empty State ──────────────────────────────────────────── */

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-64 text-center">
      <Database className="w-12 h-12 text-slate-600 mb-4" />
      <h2 className="text-lg font-medium text-slate-300">No documents yet</h2>
      <p className="text-sm text-slate-500 mt-1 max-w-sm">
        Upload documents through the Upload page to build your knowledge base.
        They'll appear here once indexed.
      </p>
    </div>
  );
}

/* ── Stat Card ────────────────────────────────────────────── */

function StatCard({ icon: Icon, label, value }: { icon: React.ComponentType<{ className?: string }>; label: string; value: string | number }) {
  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 px-4 py-3 flex items-center gap-3">
      <div className="w-9 h-9 rounded-lg bg-slate-800 flex items-center justify-center flex-shrink-0">
        <Icon className="w-4 h-4 text-slate-400" />
      </div>
      <div>
        <p className="text-lg font-semibold text-slate-200 leading-tight">{String(value)}</p>
        <p className="text-[11px] text-slate-500">{label}</p>
      </div>
    </div>
  );
}
