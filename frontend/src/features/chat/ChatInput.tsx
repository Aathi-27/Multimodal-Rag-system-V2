import { FormEvent, useState, useRef, useEffect, ChangeEvent } from 'react';
import { Image, Mic, X } from 'lucide-react';

export type QueryMode = 'text' | 'image' | 'audio';

interface ChatInputProps {
  onSend: (message: string) => void;
  onImageQuery?: (file: File, prompt?: string) => void;
  onAudioQuery?: (file: File) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSend, onImageQuery, onAudioQuery, disabled }: ChatInputProps) {
  const [value, setValue] = useState('');
  const [mode, setMode] = useState<QueryMode>('text');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    }
  }, [value]);

  // Clean up preview URL
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    if (mode === 'image' && file.type.startsWith('image/')) {
      setPreviewUrl(URL.createObjectURL(file));
    } else {
      setPreviewUrl(null);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (disabled) return;

    if (mode === 'text') {
      const trimmed = value.trim();
      if (!trimmed) return;
      onSend(trimmed);
      setValue('');
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
    } else if (mode === 'image' && selectedFile && onImageQuery) {
      onImageQuery(selectedFile, value.trim() || undefined);
      setValue('');
      clearFile();
    } else if (mode === 'audio' && selectedFile && onAudioQuery) {
      onAudioQuery(selectedFile);
      clearFile();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const switchMode = (newMode: QueryMode) => {
    setMode(newMode);
    clearFile();
    setValue('');
  };

  const canSubmit =
    mode === 'text'
      ? !!value.trim()
      : !!selectedFile;

  return (
    <form onSubmit={handleSubmit} className="border-t border-slate-800/80 bg-slate-900/60 p-4">
      <div className="mx-auto max-w-3xl space-y-3">
        {/* Mode selector */}
        <div className="flex items-center gap-2">
          <ModeButton active={mode === 'text'} onClick={() => switchMode('text')} label="Text" />
          <ModeButton active={mode === 'image'} onClick={() => switchMode('image')} label="Image" icon={<Image className="w-3.5 h-3.5" />} />
          <ModeButton active={mode === 'audio'} onClick={() => switchMode('audio')} label="Audio" icon={<Mic className="w-3.5 h-3.5" />} />
        </div>

        {/* File upload area (image/audio modes) */}
        {mode !== 'text' && (
          <div className="relative">
            {selectedFile ? (
              <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-slate-800/50 border border-slate-700/50">
                {previewUrl && (
                  <img src={previewUrl} alt="preview" className="w-12 h-12 rounded object-cover" />
                )}
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-slate-300 truncate">{selectedFile.name}</p>
                  <p className="text-[10px] text-slate-500">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                <button
                  type="button"
                  onClick={clearFile}
                  className="text-slate-500 hover:text-slate-300 transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <label className="flex items-center justify-center gap-2 px-4 py-4 rounded-lg
                border-2 border-dashed border-slate-700 hover:border-slate-500
                cursor-pointer transition-colors">
                {mode === 'image' ? (
                  <Image className="w-5 h-5 text-slate-500" />
                ) : (
                  <Mic className="w-5 h-5 text-slate-500" />
                )}
                <span className="text-sm text-slate-500">
                  {mode === 'image' ? 'Click to select an image' : 'Click to select an audio file'}
                </span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={mode === 'image' ? 'image/*' : 'audio/*'}
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </label>
            )}
          </div>
        )}

        {/* Text input + send button */}
        <div className="flex items-end gap-3">
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            placeholder={
              mode === 'text'
                ? 'Ask a question about your documents…'
                : mode === 'image'
                  ? 'Optional: describe what to find…'
                  : 'Audio will be transcribed automatically'
            }
            className="flex-1 resize-none rounded-xl border border-slate-700/80 bg-slate-900
                       px-4 py-3 text-sm text-slate-100 placeholder-slate-500
                       focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 outline-none
                       disabled:opacity-50 transition-all duration-200 min-h-[44px]
                       hover:border-slate-600"
          />
          <button
            type="submit"
            disabled={disabled || !canSubmit}
            className="flex-shrink-0 flex items-center justify-center w-11 h-11 rounded-xl
                       bg-blue-600 text-white hover:bg-blue-500 active:scale-95
                       disabled:opacity-40 disabled:cursor-not-allowed
                       transition-all duration-150"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 12 3.269 3.125A59.769 59.769 0 0 1 21.485 12 59.768 59.768 0 0 1 3.27 20.875L5.999 12Zm0 0h7.5" />
            </svg>
          </button>
        </div>
      </div>
    </form>
  );
}

/* ─── Mode selector button ────────────────────────────────────────────────── */

function ModeButton({
  active,
  onClick,
  label,
  icon,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
  icon?: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all
        ${active
          ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
          : 'text-slate-500 hover:text-slate-300 border border-transparent hover:bg-slate-800/50'
        }`}
    >
      {icon}
      {label}
    </button>
  );
}
