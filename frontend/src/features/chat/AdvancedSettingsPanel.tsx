import { useCallback, useEffect, useState } from 'react';
import axios from '@/shared/api/axios';
import type { EffectiveParams } from '@/shared/types/api.types';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

interface SettingsState extends EffectiveParams {
  overrides: Record<string, unknown>;
}

const PARAMS: {
  key: keyof EffectiveParams;
  label: string;
  min: number;
  max: number;
  step: number;
  type: 'int' | 'float';
  description: string;
}[] = [
  { key: 'rrf_k', label: 'RRF k', min: 1, max: 200, step: 1, type: 'int', description: 'Fusion constant — higher values reduce the influence of top-ranked items' },
  { key: 'rerank_threshold', label: 'Rerank Threshold', min: 0, max: 1, step: 0.01, type: 'float', description: 'Minimum reranker score to keep a chunk (normalized)' },
  { key: 'rerank_count', label: 'Rerank Count', min: 1, max: 100, step: 1, type: 'int', description: 'How many RRF candidates to send to the cross-encoder' },
  { key: 'rerank_min_results', label: 'Min Results', min: 1, max: 50, step: 1, type: 'int', description: 'Minimum chunks to retain even if below threshold' },
  { key: 'vector_top_k', label: 'Vector Top-K', min: 1, max: 200, step: 1, type: 'int', description: 'Number of candidates from vector search' },
  { key: 'bm25_top_k', label: 'BM25 Top-K', min: 1, max: 200, step: 1, type: 'int', description: 'Number of candidates from keyword search' },
];

export default function AdvancedSettingsPanel({ isOpen, onClose }: Props) {
  const [settings, setSettings] = useState<SettingsState | null>(null);
  const [localValues, setLocalValues] = useState<Record<string, number>>({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);

  /* ── Load current settings ────────────────────────────────── */
  const loadSettings = useCallback(async () => {
    try {
      const { data } = await axios.get<SettingsState>('/settings/retrieval');
      setSettings(data);
      const vals: Record<string, number> = {};
      for (const p of PARAMS) {
        vals[p.key] = data[p.key];
      }
      setLocalValues(vals);
      setDirty(false);
    } catch {
      setError('Failed to load settings');
    }
  }, []);

  useEffect(() => {
    if (isOpen) loadSettings();
  }, [isOpen, loadSettings]);

  if (!isOpen) return null;

  /* ── Handle slider change ─────────────────────────────────── */
  const handleChange = (key: string, value: number) => {
    setLocalValues((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
  };

  /* ── Save ──────────────────────────────────────────────────── */
  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const patch: Record<string, number> = {};
      for (const p of PARAMS) {
        if (localValues[p.key] !== undefined) {
          patch[p.key] = p.type === 'int'
            ? Math.round(localValues[p.key])
            : localValues[p.key];
        }
      }
      const { data } = await axios.patch<SettingsState>('/settings/retrieval', patch);
      setSettings(data);
      setDirty(false);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Save failed';
      setError(msg);
    } finally {
      setSaving(false);
    }
  };

  /* ── Reset to defaults ─────────────────────────────────────── */
  const handleReset = async () => {
    setSaving(true);
    setError(null);
    try {
      await axios.delete('/settings/retrieval');
      await loadSettings();
    } catch {
      setError('Reset failed');
    } finally {
      setSaving(false);
    }
  };

  const hasOverrides = settings && Object.keys(settings.overrides || {}).length > 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-slate-900 border border-slate-700/80 rounded-xl shadow-2xl shadow-black/40 w-full max-w-lg mx-4 max-h-[85vh] flex flex-col animate-fade-in-up">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <TuneIcon className="w-5 h-5 text-indigo-400" />
            <h2 className="text-sm font-semibold text-slate-200">Advanced Retrieval Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-slate-800 transition-colors"
          >
            <CloseIcon className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5 scrollbar-thin scrollbar-thumb-slate-700">
          {error && (
            <div className="text-xs text-red-400 bg-red-500/10 px-3 py-2 rounded">{error}</div>
          )}

          {hasOverrides && (
            <div className="text-[10px] text-amber-400 bg-amber-500/10 px-3 py-1.5 rounded flex items-center gap-1.5">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400" />
              Runtime overrides active — values differ from config.yaml defaults
            </div>
          )}

          {PARAMS.map((p) => {
            const value = localValues[p.key] ?? 0;
            const isOverridden = settings?.overrides && p.key in settings.overrides;
            return (
              <div key={p.key}>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs font-medium text-slate-300">
                    {p.label}
                    {isOverridden && (
                      <span className="ml-1.5 text-[9px] text-amber-400 font-bold uppercase">overridden</span>
                    )}
                  </label>
                  <span className="text-xs font-mono text-indigo-300 tabular-nums">
                    {p.type === 'float' ? value.toFixed(2) : value}
                  </span>
                </div>
                <input
                  type="range"
                  min={p.min}
                  max={p.max}
                  step={p.step}
                  value={value}
                  onChange={(e) => handleChange(p.key, parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer
                             accent-indigo-500"
                />
                <p className="text-[10px] text-slate-600 mt-0.5">{p.description}</p>
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-slate-800">
          <button
            onClick={handleReset}
            disabled={saving}
            className="text-xs text-slate-500 hover:text-slate-300 transition-colors disabled:opacity-50"
          >
            Reset to defaults
          </button>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 border border-slate-700
                         rounded-lg transition-all duration-150 active:scale-95"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={saving || !dirty}
              className="px-4 py-1.5 text-xs font-medium text-white bg-indigo-600 hover:bg-indigo-500
                         rounded-lg transition-all duration-150 disabled:opacity-50 disabled:cursor-not-allowed
                         active:scale-95"
            >
              {saving ? 'Saving…' : 'Apply'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Icons ────────────────────────────────────────────────── */

function TuneIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75" />
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
