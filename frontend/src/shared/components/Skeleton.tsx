/**
 * Skeleton — shimmer placeholder for loading states.
 *
 * Usage:
 *   <Skeleton className="h-4 w-48" />           → single text line
 *   <Skeleton className="h-10 w-full" rounded /> → card/row placeholder
 *   <SkeletonCard />                             → stat card placeholder
 *   <SkeletonRow />                              → table row placeholder
 */

interface SkeletonProps {
  className?: string;
  rounded?: boolean;
}

export default function Skeleton({ className = 'h-4 w-full', rounded = false }: SkeletonProps) {
  return (
    <div
      className={`animate-pulse bg-slate-800 ${rounded ? 'rounded-xl' : 'rounded-md'} ${className}`}
    />
  );
}

/* ── Preset: stat card skeleton ──────────────────────────── */

export function SkeletonCard() {
  return (
    <div className="rounded-xl border border-slate-800/80 bg-slate-900/50 px-4 py-3 flex items-center gap-3">
      <Skeleton className="w-9 h-9 flex-shrink-0" rounded />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-5 w-16" />
        <Skeleton className="h-3 w-24" />
      </div>
    </div>
  );
}

/* ── Preset: table row skeleton ──────────────────────────── */

export function SkeletonRow({ cols = 5 }: { cols?: number }) {
  return (
    <tr>
      {Array.from({ length: cols }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <Skeleton className={`h-4 ${i === 0 ? 'w-48' : 'w-16'}`} />
        </td>
      ))}
    </tr>
  );
}

/* ── Preset: list item skeleton ──────────────────────────── */

export function SkeletonListItem() {
  return (
    <div className="flex items-center gap-3 rounded-lg bg-slate-800/30 border border-slate-800 px-4 py-3">
      <Skeleton className="w-9 h-9 flex-shrink-0" rounded />
      <div className="flex-1 space-y-2">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-3 w-1/2" />
      </div>
    </div>
  );
}
