'use client'

import { useRef } from 'react'
import { CheckCircle2, Circle, Upload } from 'lucide-react'
import { toast } from 'sonner'

import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { uploadSyllabus } from '@/lib/api/client'
import { useCoverage, useSyllabus } from '@/lib/api/hooks'
import { cn } from '@/lib/utils'
import type { Job } from '@/lib/api/types'

export function SyllabusRail({
  subjectId,
  jobs = [],
  onQueued,
}: {
  subjectId: string
  jobs?: Job[]
  onQueued: (job: Job) => void
}) {
  const { data: syllabus, isLoading: sylLoading } = useSyllabus(subjectId)
  const { data: coverage } = useCoverage(subjectId)
  const inputRef = useRef<HTMLInputElement>(null)

  const pct = Math.round((coverage?.coverage_fraction ?? 0) * 100)
  const covered = new Set(coverage?.covered.map((u) => u.unit_id) ?? [])
  const sylJob = jobs.find(
    (j) => j.kind === 'ingest_syllabus' && (j.status === 'queued' || j.status === 'running')
  )

  const onFile = async (file: File) => {
    try {
      const job = await uploadSyllabus(subjectId, file)
      onQueued(job)
      toast.success('Parsing syllabus')
    } catch (err) {
      toast.error('Syllabus upload failed', {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-border px-4 py-4">
        <div className="mb-3 flex items-end justify-between gap-2">
          <div>
            <p className="kicker mb-1.5">Syllabus</p>
            <p className="font-display text-[2rem] leading-none tracking-tight text-foreground">
              {coverage?.total_units ? `${pct}%` : '—'}
            </p>
          </div>
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="flex items-center gap-1 border border-border px-2 py-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground hover:border-foreground/30 hover:text-foreground"
          >
            <Upload className="h-3 w-3" />
            {syllabus ? 'Replace' : 'PDF'}
          </button>
          <input
            ref={inputRef}
            type="file"
            accept="application/pdf"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) onFile(file)
              e.target.value = ''
            }}
          />
        </div>
        {sylJob && (
          <p className="mt-2 text-[11px] text-muted-foreground">
            {sylJob.message ?? 'Parsing…'} · {Math.round(sylJob.progress * 100)}%
          </p>
        )}
        {coverage && coverage.total_units > 0 && (
          <>
            <Progress value={pct} className="h-px rounded-none" />
            <p className="mt-1.5 text-[11px] text-muted-foreground">
              {coverage.covered_units}/{coverage.total_units} units in lectures
            </p>
          </>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
        {sylLoading ? (
          <Skeleton className="h-24 rounded-none" />
        ) : !syllabus ? (
          <p className="px-1 py-6 text-center text-[11px] leading-relaxed text-muted-foreground">
            Upload the official PDF. Chat can still use voice and files; SGCD needs this.
          </p>
        ) : (
          <ol className="space-y-1">
            {syllabus.units.map((u, i) => {
              const done = covered.has(u.id)
              return (
                <li
                  key={u.id}
                  className="flex items-start gap-2 rounded-lg px-1.5 py-1.5 hover:bg-muted/40"
                >
                  {done ? (
                    <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-foreground" />
                  ) : (
                    <Circle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground/40" />
                  )}
                  <div className="min-w-0">
                    <p className="text-[10px] font-mono text-muted-foreground">{u.unit_key}</p>
                    <p className={cn('text-xs leading-snug', done ? 'text-foreground' : 'text-muted-foreground')}>
                      {i + 1}. {u.title}
                    </p>
                  </div>
                </li>
              )
            })}
          </ol>
        )}
      </div>
    </div>
  )
}
