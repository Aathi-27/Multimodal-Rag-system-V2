import { useCallback, useEffect, useRef, useState } from 'react';
import { X, ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Download } from 'lucide-react';
import * as pdfjsLib from 'pdfjs-dist';

// Configure PDF.js worker
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

interface PDFViewerProps {
  fileId: string;
  fileName: string;
  initialPage?: number;
  onClose: () => void;
}

export default function PDFViewer({ fileId, fileName, initialPage = 1, onClose }: PDFViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [pdf, setPdf] = useState<pdfjsLib.PDFDocumentProxy | null>(null);
  const [currentPage, setCurrentPage] = useState(initialPage);
  const [totalPages, setTotalPages] = useState(0);
  const [scale, setScale] = useState(1.4);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const renderTaskRef = useRef<pdfjsLib.RenderTask | null>(null);

  // Load PDF document
  useEffect(() => {
    let cancelled = false;

    async function loadPdf() {
      try {
        setLoading(true);
        setError(null);
        const doc = await pdfjsLib.getDocument(`/files/${fileId}`).promise;
        if (cancelled) return;
        setPdf(doc);
        setTotalPages(doc.numPages);
        // Clamp initial page
        if (initialPage > doc.numPages) setCurrentPage(doc.numPages);
        else if (initialPage < 1) setCurrentPage(1);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load PDF');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadPdf();
    return () => { cancelled = true; };
  }, [fileId, initialPage]);

  // Render current page
  const renderPage = useCallback(async () => {
    if (!pdf || !canvasRef.current) return;

    try {
      // Cancel any in-flight render
      if (renderTaskRef.current) {
        renderTaskRef.current.cancel();
        renderTaskRef.current = null;
      }

      const page = await pdf.getPage(currentPage);
      const viewport = page.getViewport({ scale });
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = viewport.width;
      canvas.height = viewport.height;

      const task = page.render({ canvasContext: ctx, viewport });
      renderTaskRef.current = task;
      await task.promise;
      renderTaskRef.current = null;
    } catch (err) {
      // Ignore cancelled render tasks
      if ((err as Error)?.message?.includes('Rendering cancelled')) return;
    }
  }, [pdf, currentPage, scale]);

  useEffect(() => {
    renderPage();
  }, [renderPage]);

  // Keyboard navigation
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        setCurrentPage((p) => Math.max(1, p - 1));
      }
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        setCurrentPage((p) => Math.min(totalPages, p + 1));
      }
      if (e.key === '+' || e.key === '=') setScale((s) => Math.min(3, s + 0.2));
      if (e.key === '-') setScale((s) => Math.max(0.4, s - 0.2));
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [totalPages, onClose]);

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-slate-950/95 backdrop-blur-sm animate-fade-in">
      {/* Header toolbar */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/80 bg-slate-900/80">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-red-900/30 border border-red-700/40">
            <span className="text-xs font-medium text-red-300">PDF</span>
          </div>
          <span className="text-sm font-medium text-slate-200 truncate max-w-[300px]">{fileName}</span>
        </div>

        {/* Page navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={currentPage <= 1}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-30 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <div className="flex items-center gap-1.5">
            <input
              type="number"
              min={1}
              max={totalPages}
              value={currentPage}
              onChange={(e) => {
                const p = parseInt(e.target.value);
                if (p >= 1 && p <= totalPages) setCurrentPage(p);
              }}
              className="w-12 text-center text-sm bg-slate-800 border border-slate-700 rounded-md px-1 py-1 text-slate-200 focus:border-blue-500 outline-none"
            />
            <span className="text-xs text-slate-500">/ {totalPages}</span>
          </div>

          <button
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage >= totalPages}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 disabled:opacity-30 transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Zoom + actions */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setScale((s) => Math.max(0.4, s - 0.2))}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <span className="text-xs text-slate-400 w-12 text-center">{Math.round(scale * 100)}%</span>
          <button
            onClick={() => setScale((s) => Math.min(3, s + 0.2))}
            className="p-1.5 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <ZoomIn className="w-4 h-4" />
          </button>

          <div className="w-px h-5 bg-slate-700 mx-1" />

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

      {/* Canvas area */}
      <div className="flex-1 overflow-auto flex items-start justify-center p-6">
        {loading && (
          <div className="flex items-center gap-2 text-slate-400 mt-20">
            <div className="w-5 h-5 border-2 border-slate-600 border-t-blue-400 rounded-full animate-spin" />
            <span className="text-sm">Loading PDF…</span>
          </div>
        )}

        {error && (
          <div className="text-center mt-20">
            <p className="text-red-400 text-sm">⚠ {error}</p>
            <button
              onClick={onClose}
              className="mt-4 text-xs text-slate-400 hover:text-white underline"
            >
              Close
            </button>
          </div>
        )}

        {!loading && !error && (
          <canvas
            ref={canvasRef}
            className="shadow-2xl shadow-black/50 rounded-md"
          />
        )}
      </div>
    </div>
  );
}
