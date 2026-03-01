import { FileText, ImageIcon, Mic } from 'lucide-react';
import FileUploader from './FileUploader';

export default function UploadPage() {
  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-2xl px-6 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-xl font-semibold text-slate-100">Upload Documents</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Upload files to index them into the RAG pipeline. Supported formats include
            documents, images, and audio files.
          </p>
        </div>

        <FileUploader />

        {/* Info cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-6">
          <InfoCard
            icon={FileText}
            title="Documents"
            desc="PDF, DOCX, PPTX, TXT — parsed via Docling, chunked, and indexed."
          />
          <InfoCard
            icon={ImageIcon}
            title="Images"
            desc="PNG, JPG — OCR via EasyOCR extracts visible text for search."
          />
          <InfoCard
            icon={Mic}
            title="Audio"
            desc="MP3, WAV — transcribed via Whisper with timestamps."
          />
        </div>
      </div>
    </div>
  );
}

function InfoCard({ icon: Icon, title, desc }: { icon: React.ComponentType<{ className?: string }>; title: string; desc: string }) {
  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 p-4
                    hover:bg-slate-800/40 hover:border-slate-700/60 transition-all duration-200
                    hover:scale-[1.01]">
      <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center mb-3">
        <Icon className="w-4 h-4 text-slate-400" />
      </div>
      <h3 className="text-sm font-medium text-slate-200">{title}</h3>
      <p className="text-xs text-slate-500 mt-1 leading-relaxed">{desc}</p>
    </div>
  );
}
