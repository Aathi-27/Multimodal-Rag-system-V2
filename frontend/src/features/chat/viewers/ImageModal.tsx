import { useEffect, useState } from 'react';
import { X, ZoomIn, ZoomOut, Download, RotateCw } from 'lucide-react';

interface ImageModalProps {
  fileId: string;
  fileName: string;
  onClose: () => void;
}

export default function ImageModal({ fileId, fileName, onClose }: ImageModalProps) {
  const [scale, setScale] = useState(1);
  const [rotation, setRotation] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Keyboard controls
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === '+' || e.key === '=') setScale((s) => Math.min(4, s + 0.25));
      if (e.key === '-') setScale((s) => Math.max(0.25, s - 0.25));
      if (e.key === 'r') setRotation((r) => (r + 90) % 360);
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  // Close on backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col bg-slate-950/95 backdrop-blur-sm animate-fade-in"
      onClick={handleBackdropClick}
    >
      {/* Header toolbar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/80 bg-slate-900/80">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-emerald-900/30 border border-emerald-700/40">
            <span className="text-xs font-medium text-emerald-300">Image</span>
          </div>
          <span className="text-sm font-medium text-slate-200 truncate max-w-[300px]">{fileName}</span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setScale((s) => Math.max(0.25, s - 0.25))}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs text-slate-400 w-12 text-center">{Math.round(scale * 100)}%</span>
          <button
            onClick={() => setScale((s) => Math.min(4, s + 0.25))}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          <div className="w-px h-5 bg-slate-700 mx-1" />

          <button
            onClick={() => setRotation((r) => (r + 90) % 360)}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <RotateCw className="w-4 h-4" />
          </button>

          <a
            href={`/files/${fileId}`}
            download={fileName}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <Download className="w-4 h-4" />
          </a>

          <button
            onClick={onClose}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Image area */}
      <div
        className="flex-1 overflow-auto flex items-center justify-center p-6"
        onClick={handleBackdropClick}
      >
        {loading && (
          <div className="absolute flex items-center gap-2 text-slate-400">
            <div className="w-5 h-5 border-2 border-slate-600 border-t-emerald-400 rounded-full animate-spin" />
            <span className="text-sm">Loading image…</span>
          </div>
        )}

        {error && (
          <div className="text-center">
            <p className="text-red-400 text-sm">⚠ Failed to load image</p>
            <button
              onClick={onClose}
              className="mt-4 text-xs text-slate-400 hover:text-white underline"
            >
              Close
            </button>
          </div>
        )}

        <img
          src={`/files/${fileId}`}
          alt={fileName}
          onLoad={() => setLoading(false)}
          onError={() => {
            setLoading(false);
            setError(true);
          }}
          className="max-w-full max-h-full object-contain shadow-2xl shadow-black/50 rounded-md transition-transform duration-200"
          style={{
            transform: `scale(${scale}) rotate(${rotation}deg)`,
            display: error ? 'none' : 'block',
          }}
        />
      </div>
    </div>
  );
}
