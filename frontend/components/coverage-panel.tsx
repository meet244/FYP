'use client'

import Link from 'next/link'
import { CheckCircle2, Circle, FileQuestion } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { useCoverage } from '@/lib/api/hooks'
import { timestamp } from '@/lib/format'

/**
 * Syllabus coverage is a free consequence of the method — every note already
 * carries the unit retrieval picked for its spans, so this is a group-by, not a
 * separate classification pass.
 */
export function CoveragePanel({ subjectId }: { subjectId: string }) {
  const { data, error, isLoading } = useCoverage(subjectId)

  if (isLoading) return <Skeleton className="h-64 rounded-xl" />

  if (error || !data) {
    return (
      <p className="rounded-xl border border-border/60 bg-card p-8 text-center text-sm text-muted-foreground">
        Coverage is unavailable — check the backend is running.
      </p>
    )
  }

  if (data.total_units === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border bg-card/40 p-12 text-center">
        <FileQuestion className="mx-auto mb-3 h-8 w-8 text-muted-foreground" />
        <h4 className="mb-1.5 font-semibold text-foreground">No syllabus to track against</h4>
        <p className="mx-auto max-w-sm text-sm leading-relaxed text-muted-foreground">
          Upload the syllabus PDF and coverage fills in on its own as lectures are processed.
        </p>
      </div>
    )
  }

  const pct = Math.round((data.coverage_fraction ?? 0) * 100)

  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-border/60 bg-card p-5">
        <div className="mb-3 flex items-end justify-between">
          <div>
            <p className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              Syllabus delivered
            </p>
            <p className="text-3xl font-black tabular-nums leading-none tracking-tighter text-foreground">
              {pct}%
            </p>
          </div>
          <p className="text-sm text-muted-foreground">
            <span className="font-semibold text-foreground">{data.covered_units}</span> of{' '}
            {data.total_units} units
          </p>
        </div>
        <Progress value={pct} className="h-2" />
        {data.unattributed_notes > 0 && (
          <p className="mt-3 text-xs text-muted-foreground">
            {data.unattributed_notes} note{data.unattributed_notes === 1 ? '' : 's'} could not be
            attributed to a unit — usually material taught outside the syllabus.
          </p>
        )}
      </div>

      {data.covered.length > 0 && (
        <div>
          <h4 className="mb-2 flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-muted-foreground">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" /> Covered
          </h4>
          <div className="space-y-2">
            {data.covered.map((u) => (
              <div key={u.unit_id} className="rounded-xl border border-border/60 bg-card p-3.5">
                <div className="mb-2 flex items-center gap-2">
                  <span className="shrink-0 rounded-md bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                    {u.unit_key}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                    {u.title}
                  </span>
                  <Badge variant="secondary" className="shrink-0 text-[10px]">
                    {u.note_count} note{u.note_count === 1 ? '' : 's'}
                  </Badge>
                </div>

                <div className="space-y-1 pl-1">
                  {u.notes.map((n) => (
                    <Link
                      key={n.note_id}
                      href={`/lectures/${n.lecture_id}?note=${n.note_id}`}
                      className="flex items-baseline gap-2 rounded-md px-1.5 py-1 text-xs transition-colors hover:bg-muted/60"
                    >
                      <span className="font-mono text-[10px] text-primary">
                        {timestamp(n.start_s)}
                      </span>
                      <span className="min-w-0 flex-1 truncate text-muted-foreground">
                        {n.topic}
                      </span>
                      <span className="shrink-0 truncate text-[10px] text-muted-foreground/70">
                        {n.lecture_title}
                      </span>
                    </Link>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.outstanding.length > 0 && (
        <div>
          <h4 className="mb-2 flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-muted-foreground">
            <Circle className="h-3.5 w-3.5" /> Outstanding
          </h4>
          <div className="space-y-1.5">
            {data.outstanding.map((u) => (
              <div
                key={u.unit_id}
                className="flex items-center gap-2 rounded-xl border border-dashed border-border/60 px-3.5 py-2.5"
              >
                <span className="shrink-0 rounded-md bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                  {u.unit_key}
                </span>
                <span className="min-w-0 flex-1 truncate text-sm text-muted-foreground">
                  {u.title}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
