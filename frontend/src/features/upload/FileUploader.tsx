import { useCallback, useRef, useState } from 'react';
import api from '@/shared/api/axios';
import type { UploadResponse } from '@/shared/types/api.types';
import Loader from '@/shared/components/Loader';

const ACCEPTED = '.pdf,.docx,.pptx,.txt,.png,.jpg,.jpeg,.mp3,.wav,.m4a,.ogg,.flac';
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100 MB

interface FileUploadState {
  file: File;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  uploadId?: string;
  error?: string;
  modality?: string;
  chunks?: number;
}

interface FileUploaderProps {
  onUploaded?: (uploadId: string) => void;
}

export default function FileUploader({ onUploaded }: FileUploaderProps) {
  const [uploads, setUploads] = useState<FileUploadState[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  /* ─── File selection ───────────────────────────────────────────────── */

  const addFiles = useCallback(
    (files: FileList | File[]) => {
      const allowedExts = ACCEPTED.split(',');

      Array.from(files).forEach((file) => {
        // Size gate
        if (file.size > MAX_FILE_SIZE) {
          setUploads((prev) => [
            ...prev,
            {
              file,
              progress: 0,
              status: 'failed',
              error: `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max is ${MAX_FILE_SIZE / 1024 / 1024} MB.`,
            },
          ]);
          return;
        }

        // Extension gate
        const ext = '.' + (file.name.split('.').pop()?.toLowerCase() || '');
        if (!allowedExts.includes(ext)) {
          setUploads((prev) => [
            ...prev,
            { file, progress: 0, status: 'failed', error: `Unsupported format: ${ext}` },
          ]);
          return;
        }

        uploadFile(file);
      });
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  /* ─── Upload a single file ─────────────────────────────────────────── */

  const uploadFile = async (file: File) => {
    const id = crypto.randomUUID();
    const entry: FileUploadState = { file, progress: 0, status: 'uploading' };

    setUploads((prev) => [...prev, entry]);
    const idx = uploads.length; // capture current index

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await api.post<UploadResponse>('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (e) => {
          const pct = e.total ? Math.round((e.loaded / e.total) * 100) : 0;
          setUploads((prev) =>
            prev.map((u, i) => (u.file === file && u.status === 'uploading' ? { ...u, progress: pct } : u)),
          );
        },
      });

      const uploadId = res.data.upload_id;

      setUploads((prev) =>
        prev.map((u) =>
          u.file === file
            ? { ...u, status: 'processing', progress: 100, uploadId, modality: res.data.modality }
            : u,
        ),
      );

      // Poll for completion
      pollStatus(file, uploadId);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Upload failed';
      setUploads((prev) =>
        prev.map((u) => (u.file === file ? { ...u, status: 'failed', error: message } : u)),
      );
    }
  };

  /* ─── Poll ingestion status ────────────────────────────────────────── */

  const pollStatus = async (file: File, uploadId: string) => {
    const MAX_POLLS = 120; // 2 minutes max
    for (let i = 0; i < MAX_POLLS; i++) {
      await new Promise((r) => setTimeout(r, 2000));

      try {
        const res = await api.get<{
          task_id: string;
          status: string;
          error?: string;
          chunks?: number;
        }>(`/status/${uploadId}`);

        if (res.data.status === 'completed') {
          setUploads((prev) =>
            prev.map((u) =>
              u.uploadId === uploadId
                ? { ...u, status: 'completed', chunks: res.data.chunks }
                : u,
            ),
          );
          onUploaded?.(uploadId);
          return;
        }

        if (res.data.status === 'failed') {
          setUploads((prev) =>
            prev.map((u) =>
              u.uploadId === uploadId
                ? { ...u, status: 'failed', error: res.data.error || 'Ingestion failed' }
                : u,
            ),
          );
          return;
        }
      } catch {
        // Network error — keep polling
      }
    }

    // Timeout
    setUploads((prev) =>
      prev.map((u) =>
        u.uploadId === uploadId ? { ...u, status: 'failed', error: 'Polling timed out' } : u,
      ),
    );
  };

  /* ─── Drag & drop handlers ─────────────────────────────────────────── */

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };
  const onDragLeave = () => setDragActive(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  };

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`
          relative cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors
          ${dragActive ? 'border-blue-500 bg-blue-950/20' : 'border-slate-700 hover:border-slate-600 bg-slate-900/50'}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED}
          className="hidden"
          onChange={(e) => e.target.files && addFiles(e.target.files)}
        />

        <svg className="mx-auto w-10 h-10 text-slate-600 mb-3" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
        </svg>

        <p className="text-sm text-slate-300 font-medium">
          Drop files here or <span className="text-blue-400">browse</span>
        </p>
        <p className="text-xs text-slate-500 mt-1">
          PDF, DOCX, PPTX, TXT, PNG, JPG, MP3, WAV — max 100 MB
        </p>
      </div>

      {/* Upload list */}
      {uploads.length > 0 && (
        <div className="space-y-2">
          {uploads.map((u, i) => (
            <UploadRow key={i} upload={u} />
          ))}
        </div>
      )}
    </div>
  );
}

/* ─── Upload row ─────────────────────────────────────────────────────────── */

function UploadRow({ upload }: { upload: FileUploadState }) {
  const { file, progress, status, error, modality, chunks } = upload;

  const statusColor = {
    uploading: 'text-blue-400',
    processing: 'text-yellow-400',
    completed: 'text-emerald-400',
    failed: 'text-red-400',
  }[status];

  const icon = {
    uploading: '↑',
    processing: '⟳',
    completed: '✓',
    failed: '✕',
  }[status];

  return (
    <div className="flex items-center gap-3 rounded-lg bg-slate-800/60 border border-slate-700/50 px-4 py-3">
      {/* File icon */}
      <FileTypeIcon filename={file.name} />

      {/* Details */}
      <div className="flex-1 min-w-0">
        <p className="text-sm text-slate-200 truncate">{file.name}</p>
        <div className="flex items-center gap-2 mt-0.5">
          <span className={`text-xs font-medium ${statusColor}`}>
            {icon} {status === 'uploading' ? `${progress}%` : status}
          </span>
          {modality && (
            <span className="text-xs text-slate-500">• {modality}</span>
          )}
          {status === 'completed' && chunks != null && (
            <span className="text-xs text-slate-500">• {chunks} chunks</span>
          )}
          {status === 'processing' && <Loader size="sm" />}
        </div>
        {error && <p className="text-xs text-red-400 mt-0.5">{error}</p>}
      </div>

      {/* Progress bar (during upload) */}
      {status === 'uploading' && (
        <div className="w-20 h-1.5 rounded-full bg-slate-700 overflow-hidden">
          <div
            className="h-full bg-blue-500 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  );
}

function FileTypeIcon({ filename }: { filename: string }) {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  const colors: Record<string, string> = {
    pdf: 'bg-red-900/50 text-red-400',
    docx: 'bg-blue-900/50 text-blue-400',
    pptx: 'bg-orange-900/50 text-orange-400',
    txt: 'bg-slate-800 text-slate-400',
    png: 'bg-emerald-900/50 text-emerald-400',
    jpg: 'bg-emerald-900/50 text-emerald-400',
    jpeg: 'bg-emerald-900/50 text-emerald-400',
    mp3: 'bg-purple-900/50 text-purple-400',
    wav: 'bg-purple-900/50 text-purple-400',
  };

  return (
    <div className={`flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center text-xs font-bold uppercase ${colors[ext] || 'bg-slate-800 text-slate-400'}`}>
      {ext}
    </div>
  );
}
