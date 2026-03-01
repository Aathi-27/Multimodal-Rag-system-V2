import { useEffect, useRef, useState } from 'react';
import { X, Play, Pause, Volume2, VolumeX, Download } from 'lucide-react';

interface AudioPlayerProps {
  fileId: string;
  fileName: string;
  initialTimestamp?: number;  // seconds
  speaker?: string | null;
  onClose: () => void;
}

export default function AudioPlayer({
  fileId,
  fileName,
  initialTimestamp = 0,
  speaker,
  onClose,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [volume, setVolume] = useState(0.8);
  const [isLoaded, setIsLoaded] = useState(false);

  // Seek to initial timestamp when loaded
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoaded = () => {
      setDuration(audio.duration);
      setIsLoaded(true);
      if (initialTimestamp > 0) {
        audio.currentTime = initialTimestamp;
        setCurrentTime(initialTimestamp);
      }
      audio.play().then(() => setIsPlaying(true)).catch(() => {});
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleEnded = () => {
      setIsPlaying(false);
    };

    audio.addEventListener('loadedmetadata', handleLoaded);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoaded);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [initialTimestamp]);

  // Keyboard controls
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === ' ') {
        e.preventDefault();
        togglePlay();
      }
      if (e.key === 'ArrowLeft') seekBy(-5);
      if (e.key === 'ArrowRight') seekBy(5);
      if (e.key === 'm') setIsMuted((m) => !m);
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const seekBy = (seconds: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = Math.max(0, Math.min(duration, audio.currentTime + seconds));
  };

  const handleProgressClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    const bar = progressRef.current;
    if (!audio || !bar) return;

    const rect = bar.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    audio.currentTime = pct * duration;
  };

  const handleVolumeChange = (val: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    setVolume(val);
    audio.volume = val;
    if (val === 0) setIsMuted(true);
    else setIsMuted(false);
  };

  const toggleMute = () => {
    const audio = audioRef.current;
    if (!audio) return;
    setIsMuted((m) => {
      audio.muted = !m;
      return !m;
    });
  };

  const formatTime = (s: number) => {
    if (!isFinite(s)) return '0:00';
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-lg mx-4 bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800/60">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-purple-900/30 border border-purple-700/40">
              <span className="text-xs font-medium text-purple-300">Audio</span>
            </div>
            <div>
              <p className="text-sm font-medium text-slate-200 truncate max-w-[250px]">{fileName}</p>
              {speaker && (
                <p className="text-xs text-slate-500">Speaker: {speaker}</p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
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

        {/* Waveform / Progress */}
        <div className="px-5 pt-6 pb-2">
          <div
            ref={progressRef}
            onClick={handleProgressClick}
            className="relative h-2 bg-slate-800 rounded-full cursor-pointer group"
          >
            <div
              className="absolute inset-y-0 left-0 bg-purple-500 rounded-full transition-all duration-100"
              style={{ width: `${progressPct}%` }}
            />
            <div
              className="absolute top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-white rounded-full shadow-lg
                         opacity-0 group-hover:opacity-100 transition-opacity"
              style={{ left: `calc(${progressPct}% - 7px)` }}
            />
          </div>

          {/* Time labels */}
          <div className="flex justify-between mt-1.5">
            <span className="text-[10px] text-slate-500">{formatTime(currentTime)}</span>
            <span className="text-[10px] text-slate-500">{formatTime(duration)}</span>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between px-5 py-4">
          {/* Seek backward */}
          <button
            onClick={() => seekBy(-10)}
            className="text-xs text-slate-400 hover:text-white transition-colors"
          >
            −10s
          </button>

          {/* Play/Pause */}
          <button
            onClick={togglePlay}
            disabled={!isLoaded}
            className="w-12 h-12 rounded-full bg-purple-600 hover:bg-purple-500 text-white
                       flex items-center justify-center transition-all active:scale-95
                       disabled:opacity-40"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
          </button>

          {/* Seek forward */}
          <button
            onClick={() => seekBy(10)}
            className="text-xs text-slate-400 hover:text-white transition-colors"
          >
            +10s
          </button>
        </div>

        {/* Volume */}
        <div className="flex items-center gap-2 px-5 pb-4">
          <button onClick={toggleMute} className="text-slate-400 hover:text-white transition-colors">
            {isMuted || volume === 0 ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={isMuted ? 0 : volume}
            onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
            className="flex-1 h-1 appearance-none bg-slate-700 rounded-full accent-purple-500"
          />
        </div>

        {/* Hidden audio element */}
        <audio ref={audioRef} src={`/files/${fileId}`} preload="metadata" />
      </div>
    </div>
  );
}
