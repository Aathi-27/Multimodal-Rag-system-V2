import { useCallback, useEffect, useState } from 'react';
import api from '@/shared/api/axios';
import type {
  HealthResponse,
  IndexHealthResponse,
  MetricsResponse,
  ResourceStatus,
  VersionListResponse,
  VersionInfo,
} from '@/shared/types/api.types';
import {
  BarChart3,
  Timer,
  Monitor,
  Package,
  Cpu,
  MemoryStick,
  HardDrive,
  Zap,
  RefreshCw,
  Brain,
  ScanSearch,
  Eye,
  AudioLines,
} from 'lucide-react';
import Loader from '@/shared/components/Loader';
import ErrorMessage from '@/shared/components/ErrorMessage';
import { SkeletonCard } from '@/shared/components/Skeleton';

const POLL_INTERVAL = 15_000; // 15 seconds

export default function SystemStatusPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [indexHealth, setIndexHealth] = useState<IndexHealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [resources, setResources] = useState<ResourceStatus | null>(null);
  const [versions, setVersions] = useState<VersionListResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastChecked, setLastChecked] = useState<Date | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'resources' | 'versions'>('overview');

  const fetchAll = useCallback(async () => {
    try {
      const [healthRes, indexRes, metricsRes, resourcesRes, versionsRes] = await Promise.allSettled([
        api.get<HealthResponse>('/health'),
        api.get<IndexHealthResponse>('/index/health'),
        api.get<MetricsResponse>('/metrics'),
        api.get<ResourceStatus>('/resources'),
        api.get<VersionListResponse>('/versions'),
      ]);
      if (healthRes.status === 'fulfilled') setHealth(healthRes.value.data);
      if (indexRes.status === 'fulfilled') setIndexHealth(indexRes.value.data);
      if (metricsRes.status === 'fulfilled') setMetrics(metricsRes.value.data);
      if (resourcesRes.status === 'fulfilled') setResources(resourcesRes.value.data);
      if (versionsRes.status === 'fulfilled') setVersions(versionsRes.value.data);
      setError('');
      setLastChecked(new Date());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to fetch system status');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
    let timerId: ReturnType<typeof setInterval> | null = setInterval(fetchAll, POLL_INTERVAL);

    const onVisibility = () => {
      if (document.hidden) {
        if (timerId !== null) { clearInterval(timerId); timerId = null; }
      } else {
        fetchAll();
        timerId = setInterval(fetchAll, POLL_INTERVAL);
      }
    };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      if (timerId !== null) clearInterval(timerId);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [fetchAll]);

  if (loading) {
    return (
      <div className="h-full overflow-y-auto">
        <div className="mx-auto max-w-[1400px] px-6 py-6">
          <div className="mb-5">
            <h1 className="text-xl font-semibold text-slate-100">System Status</h1>
            <p className="text-sm text-slate-500 mt-0.5">Loading…</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)}
          </div>
        </div>
      </div>
    );
  }

  const tabs = [
    { key: 'overview' as const, label: 'Overview', icon: BarChart3 },
    { key: 'metrics' as const, label: 'Latency & Metrics', icon: Timer },
    { key: 'resources' as const, label: 'Resources', icon: Monitor },
    { key: 'versions' as const, label: 'Index Versions', icon: Package },
  ];

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1400px] px-6 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-5">
          <div>
            <h1 className="text-xl font-semibold text-slate-100">System Status</h1>
            <p className="text-sm text-slate-500 mt-0.5">
              Live monitoring — refreshes every 15s
            </p>
          </div>
          {lastChecked && (
            <span className="text-xs text-slate-600">
              Last check: {lastChecked.toLocaleTimeString()}
            </span>
          )}
        </div>

        {error && (
          <div className="mb-5">
            <ErrorMessage message={error} onRetry={fetchAll} />
          </div>
        )}

        {/* Tab bar */}
        <div className="flex gap-1 mb-5 border-b border-slate-800/80 pb-px">
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setActiveTab(t.key)}
              className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                activeTab === t.key
                  ? 'bg-slate-800 text-slate-100 border-b-2 border-blue-500'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-slate-900/50'
              }`}
            >
              <t.icon className="w-4 h-4" />
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        {activeTab === 'overview' && (
          <OverviewTab health={health} indexHealth={indexHealth} />
        )}
        {activeTab === 'metrics' && (
          <MetricsTab metrics={metrics} />
        )}
        {activeTab === 'resources' && (
          <ResourcesTab resources={resources} />
        )}
        {activeTab === 'versions' && (
          <VersionsTab versions={versions} onRefresh={fetchAll} />
        )}
      </div>
    </div>
  );
}


/* ══════════════════════════════════════════════════════════════════════════════
   OVERVIEW TAB (original health + index health)
   ══════════════════════════════════════════════════════════════════════════ */

function OverviewTab({
  health,
  indexHealth,
}: {
  health: HealthResponse | null;
  indexHealth: IndexHealthResponse | null;
}) {
  if (!health) return null;

  return (
    <>
      <OverallBanner status={health.status} uptime={health.uptime} />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-5">
        <StatusCard title="Vector Database" subtitle="Qdrant" status={health.qdrant} statusLabel={health.qdrant} icon={HardDrive} />
        <StatusCard title="Language Model" subtitle="Qwen2.5-1.5B" status={health.llm} statusLabel={health.llm} icon={Cpu} />
        <StatusCard title="BM25 Index" subtitle="Keyword search" status={health.bm25} statusLabel={health.bm25} icon={BarChart3} />
        <StatusCard title="Corpus Size" subtitle="Indexed chunks" status="info" statusLabel={`${health.corpus_size} chunks`} icon={Package} />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-4">
        <StatusCard title="Embeddings" subtitle="BGE-small-en-v1.5" status={health.embeddings} statusLabel={health.embeddings} icon={Brain} />
        <StatusCard title="Reranker" subtitle="BGE-reranker-base" status={health.reranker} statusLabel={health.reranker} icon={ScanSearch} />
        <StatusCard title="CLIP Vision" subtitle="ViT-B/32" status={health.clip} statusLabel={health.clip} icon={Eye} />
        <StatusCard title="Whisper STT" subtitle="faster-whisper-small" status={health.whisper} statusLabel={health.whisper} icon={AudioLines} />
      </div>

      {indexHealth && (
        <div className="mt-5 rounded-xl border border-slate-800/80 bg-slate-900/50 p-5">
          <h2 className="text-sm font-medium text-slate-300 mb-4 flex items-center gap-2">
            <Package className="w-4 h-4 text-slate-400" /> Index Health
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <MetricBox label="Documents" value={indexHealth.total_documents} />
            <MetricBox label="Total Chunks" value={indexHealth.total_chunks} />
            <MetricBox label="Total Tokens" value={indexHealth.total_tokens.toLocaleString()} />
            <MetricBox label="Avg Tokens / Chunk" value={indexHealth.avg_tokens_per_chunk} />
            <MetricBox label="BM25 Chunks" value={indexHealth.bm25_chunk_count} />
            <MetricBox label="BM25 Vocabulary" value={indexHealth.bm25_vocab_size.toLocaleString()} />
            <MetricBox label="Embedding Dim" value={indexHealth.embedding_dimension} />
            <MetricBox label="Qdrant Collection" value={indexHealth.qdrant_collection} />
          </div>
          {indexHealth.largest_document && (
            <div className="mt-4 pt-3 border-t border-slate-800 flex items-center gap-2 text-xs text-slate-500">
              <span>Largest document:</span>
              <span className="text-slate-300 font-medium">{indexHealth.largest_document}</span>
              <span>({indexHealth.largest_document_chunks} chunks)</span>
            </div>
          )}
        </div>
      )}
    </>
  );
}


/* ══════════════════════════════════════════════════════════════════════════════
   METRICS TAB (per-stage latency histograms)
   ══════════════════════════════════════════════════════════════════════════ */

function MetricsTab({ metrics }: { metrics: MetricsResponse | null }) {
  if (!metrics) {
    return <p className="text-slate-500 text-sm">Metrics not available yet.</p>;
  }

  const stages = [
    { label: 'End-to-End Query', data: metrics.query_latency, color: 'bg-blue-500' },
    { label: 'Retrieval', data: metrics.retrieval_latency, color: 'bg-blue-500' },
    { label: 'Reranking', data: metrics.rerank_latency, color: 'bg-purple-500' },
    { label: 'LLM Generation', data: metrics.generation_latency, color: 'bg-emerald-500' },
  ];

  return (
    <div className="space-y-6">
      {/* Counters */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 p-4 text-center">
          <p className="text-2xl font-bold text-slate-100">{metrics.queries_total}</p>
          <p className="text-xs text-slate-500 mt-1">Total Queries</p>
        </div>
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 p-4 text-center">
          <p className="text-2xl font-bold text-slate-100">{metrics.uploads_total}</p>
          <p className="text-xs text-slate-500 mt-1">Total Uploads</p>
        </div>
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 p-4 text-center">
          <p className="text-2xl font-bold text-red-400">{metrics.upload_errors}</p>
          <p className="text-xs text-slate-500 mt-1">Upload Errors</p>
        </div>
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 p-4 text-center">
          <p className="text-2xl font-bold text-slate-100">{metrics.corpus_size}</p>
          <p className="text-xs text-slate-500 mt-1">Corpus Size</p>
        </div>
      </div>

      {/* Per-stage latency cards */}
      <div className="space-y-3">
        <h2 className="text-sm font-medium text-slate-300 flex items-center gap-2">
          <Timer className="w-4 h-4 text-slate-400" /> Per-Stage Latency
        </h2>
        {stages.map((s) => (
          <LatencyCard key={s.label} label={s.label} data={s.data} color={s.color} />
        ))}
      </div>
    </div>
  );
}

function LatencyCard({
  label,
  data,
  color,
}: {
  label: string;
  data: { avg: number; p95: number; count: number };
  color: string;
}) {
  // Scale bar to max 30s (reasonable upper bound)
  const maxBar = 30;
  const avgPct = Math.min((data.avg / maxBar) * 100, 100);
  const p95Pct = Math.min((data.p95 / maxBar) * 100, 100);

  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-slate-200">{label}</h3>
        <span className="text-xs text-slate-500">{data.count} samples</span>
      </div>
      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 w-8">avg</span>
          <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-500 ${color}`} style={{ width: `${avgPct}%` }} />
          </div>
          <span className="text-xs font-mono text-slate-300 w-16 text-right">{data.avg.toFixed(3)}s</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500 w-8">p95</span>
          <div className="flex-1 h-2 rounded-full bg-slate-800 overflow-hidden">
            <div className={`h-full rounded-full transition-all duration-500 ${color} opacity-60`} style={{ width: `${p95Pct}%` }} />
          </div>
          <span className="text-xs font-mono text-slate-300 w-16 text-right">{data.p95.toFixed(3)}s</span>
        </div>
      </div>
    </div>
  );
}


/* ══════════════════════════════════════════════════════════════════════════════
   RESOURCES TAB (CPU, RAM, GPU, Disk)
   ══════════════════════════════════════════════════════════════════════════ */

function ResourcesTab({ resources }: { resources: ResourceStatus | null }) {
  if (!resources) {
    return <p className="text-slate-500 text-sm">Resource data not available.</p>;
  }

  return (
    <div className="space-y-4">
      {/* CPU + RAM */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <ResourceGauge
          icon={Cpu}
          title="CPU"
          value={resources.cpu_percent}
          unit="%"
          color={gaugeColor(resources.cpu_percent)}
        />
        <ResourceGauge
          icon={MemoryStick}
          title="RAM"
          value={resources.ram_percent}
          unit="%"
          subtitle={`${resources.ram_used_mb.toFixed(0)} / ${resources.ram_total_mb.toFixed(0)} MB`}
          color={gaugeColor(resources.ram_percent)}
        />
      </div>

      {/* GPU */}
      {resources.gpu_name && (
        <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 p-5">
          <h2 className="text-sm font-medium text-slate-300 mb-3 flex items-center gap-2">
            <Zap className="w-4 h-4 text-slate-400" /> GPU — {resources.gpu_name}
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <ResourceGauge
              icon={Zap}
              title="GPU Utilization"
              value={resources.gpu_utilization}
              unit="%"
              color={gaugeColor(resources.gpu_utilization)}
              compact
            />
            <ResourceGauge
              icon={MemoryStick}
              title="GPU Memory"
              value={resources.gpu_memory_percent}
              unit="%"
              subtitle={`${resources.gpu_memory_used_mb.toFixed(0)} / ${resources.gpu_memory_total_mb.toFixed(0)} MB`}
              color={gaugeColor(resources.gpu_memory_percent)}
              compact
            />
          </div>
        </div>
      )}

      {/* Disk */}
      <ResourceGauge
        icon={HardDrive}
        title="Disk (D:)"
        value={resources.disk_percent}
        unit="%"
        subtitle={`${resources.disk_used_gb.toFixed(1)} / ${resources.disk_total_gb.toFixed(1)} GB`}
        color={gaugeColor(resources.disk_percent)}
      />
    </div>
  );
}

function gaugeColor(pct: number): string {
  if (pct >= 90) return 'bg-red-500';
  if (pct >= 70) return 'bg-yellow-500';
  return 'bg-emerald-500';
}

function ResourceGauge({
  icon: Icon,
  title,
  value,
  unit,
  subtitle,
  color,
  compact = false,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  value: number;
  unit: string;
  subtitle?: string;
  color: string;
  compact?: boolean;
}) {
  return (
    <div className={`rounded-xl border border-slate-800/80 bg-slate-900/50 p-4 ${compact ? '' : ''}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-slate-200 flex items-center gap-2">
          <Icon className="w-4 h-4 text-slate-400" />
          {title}
        </h3>
        <span className="text-lg font-bold text-slate-100">
          {value.toFixed(1)}{unit}
        </span>
      </div>
      <div className="h-2 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ease-out ${color}`}
          style={{ width: `${Math.min(value, 100)}%` }}
        />
      </div>
      {subtitle && (
        <p className="text-xs text-slate-500 mt-1.5">{subtitle}</p>
      )}
    </div>
  );
}


/* ══════════════════════════════════════════════════════════════════════════════
   VERSIONS TAB (index version management)
   ══════════════════════════════════════════════════════════════════════════ */

function VersionsTab({
  versions,
  onRefresh,
}: {
  versions: VersionListResponse | null;
  onRefresh: () => void;
}) {
  const [switching, setSwitching] = useState<string | null>(null);

  if (!versions) {
    return <p className="text-slate-500 text-sm">Version data not available.</p>;
  }

  const handleSwitch = async (version: string) => {
    if (!confirm(`Switch active index to ${version}? Services may need a restart.`)) return;
    setSwitching(version);
    try {
      await api.post(`/versions/${version}/switch`);
      onRefresh();
    } catch {
      alert('Failed to switch version.');
    } finally {
      setSwitching(null);
    }
  };

  const handleDelete = async (version: string) => {
    if (!confirm(`Delete version ${version}? This is irreversible.`)) return;
    try {
      await api.delete(`/versions/${version}`);
      onRefresh();
    } catch {
      alert('Failed to delete version.');
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-slate-300 flex items-center gap-2">
          <Package className="w-4 h-4 text-slate-400" /> Index Versions · Active: <span className="text-blue-400 font-semibold">{versions.current_version || 'none'}</span>
        </h2>
      </div>

      {versions.versions.length === 0 ? (
        <p className="text-slate-500 text-sm">No index versions found.</p>
      ) : (
        <div className="rounded-xl border border-slate-800/80 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-900/80 text-slate-400 text-xs uppercase tracking-wider">
                <th className="text-left px-4 py-3 font-medium">Version</th>
                <th className="text-right px-4 py-3 font-medium">Size</th>
                <th className="text-center px-4 py-3 font-medium">Status</th>
                <th className="text-right px-4 py-3 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/50">
              {versions.versions.map((v: VersionInfo) => (
                <tr key={v.version} className="hover:bg-slate-900/30 transition-colors">
                  <td className="px-4 py-3 font-medium text-slate-200">{v.version}</td>
                  <td className="px-4 py-3 text-right text-slate-400 tabular-nums">
                    {v.size_mb.toFixed(1)} MB
                  </td>
                  <td className="px-4 py-3 text-center">
                    {v.is_active ? (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-900/50 text-emerald-400 border border-emerald-700/50">
                        active
                      </span>
                    ) : (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-500 border border-slate-700">
                        inactive
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex gap-2 justify-end">
                      {!v.is_active && (
                        <>
                          <button
                            onClick={() => handleSwitch(v.version)}
                            disabled={switching === v.version}
                            className="text-xs px-2.5 py-1 rounded-lg bg-blue-600/20 text-blue-400
                                       hover:bg-blue-600/30 transition-colors disabled:opacity-50"
                          >
                            {switching === v.version ? 'Switching…' : 'Activate'}
                          </button>
                          <button
                            onClick={() => handleDelete(v.version)}
                            className="text-xs px-2.5 py-1 rounded-lg bg-red-900/20 text-red-400
                                       hover:bg-red-900/30 transition-colors"
                          >
                            Delete
                          </button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}


/* ═══════════════════════════════════════════════════════════════════════════
   SHARED UI COMPONENTS
   ═══════════════════════════════════════════════════════════════════════ */

function OverallBanner({ status, uptime }: { status: string; uptime: string }) {
  const color =
    status === 'healthy'
      ? 'border-emerald-700/50 bg-emerald-950/30'
      : status === 'degraded'
        ? 'border-yellow-700/50 bg-yellow-950/30'
        : 'border-red-700/50 bg-red-950/30';

  const dot =
    status === 'healthy'
      ? 'bg-emerald-400'
      : status === 'degraded'
        ? 'bg-yellow-400'
        : 'bg-red-400';

  const label =
    status === 'healthy'
      ? 'All systems operational'
      : status === 'degraded'
        ? 'System degraded'
        : 'System error';

  return (
    <div className={`flex items-center gap-4 rounded-xl border p-5 ${color}`}>
      <span className="relative flex h-3 w-3">
        <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${dot}`} />
        <span className={`relative inline-flex rounded-full h-3 w-3 ${dot}`} />
      </span>
      <div>
        <p className="text-sm font-semibold text-slate-100">{label}</p>
        <p className="text-xs text-slate-500">Uptime: {uptime}</p>
      </div>
    </div>
  );
}

function StatusCard({
  title,
  subtitle,
  status,
  statusLabel,
  icon: Icon,
}: {
  title: string;
  subtitle: string;
  status: string;
  statusLabel: string;
  icon: React.ComponentType<{ className?: string }>;
}) {
  const isGood = ['connected', 'ready', 'loaded', 'info'].includes(status);
  const isWarn = ['not_loaded', 'downloaded'].includes(status);
  const badgeColor = isGood
    ? 'bg-emerald-900/50 text-emerald-400 border-emerald-700/50'
    : isWarn
      ? 'bg-yellow-900/50 text-yellow-400 border-yellow-700/50'
      : 'bg-red-900/50 text-red-400 border-red-700/50';

  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 p-4">
      <div className="flex items-start justify-between">
        <div className="w-9 h-9 rounded-lg bg-slate-800 flex items-center justify-center">
          <Icon className="w-4 h-4 text-slate-400" />
        </div>
        <span className={`text-xs px-2 py-0.5 rounded-full border ${badgeColor}`}>
          {statusLabel}
        </span>
      </div>
      <h3 className="text-sm font-medium text-slate-200 mt-3">{title}</h3>
      <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>
    </div>
  );
}

function MetricBox({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-sm font-semibold text-slate-200 mt-0.5">{String(value)}</p>
    </div>
  );
}
