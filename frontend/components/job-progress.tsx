'use client'

import { AlertCircle, CheckCircle2, X } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type { Job } from '@/lib/api/types'
import { cancelJob, retryJob } from '@/lib/api/client'
import { jobProgress } from '@/lib/job-progress'

function titleFor(job: Job) {
  if (job.kind === 'ingest_syllabus') return 'Syllabus'
  if (job.kind === 'ingest_material') return 'File'
  if (job.kind === 'generate_notes') return 'Notes'
  return 'Lecture'
}

export function JobProgress({
  job,
  onDismiss,
  className,
  onQueued,
}: {
  job: Job
  onDismiss?: (id: string) => void
  className?: string
  onQueued?: (job: Job) => void
}) {
  const [busy, setBusy] = useState(false)
  const done = job.status === 'succeeded'
  const failed = job.status === 'failed'
  const cancelled = job.status === 'cancelled'
  const progress = jobProgress(job)
  const pct = Math.round(progress.value * 100)
  const action = async (retry: boolean) => {
    setBusy(true)
    try {
      const next = await (retry ? retryJob(job.id) : cancelJob(job.id))
      if (retry) { onQueued?.(next); onDismiss?.(job.id) }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Could not update job')
    } finally { setBusy(false) }
  }

  return (
    <motion.div
      layout
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, height: 0 }}
      className={cn(
        'relative flex items-center gap-2 overflow-hidden border-b border-border px-3 py-1.5 text-xs',
        className
      )}
    >
      {failed ? (
        <AlertCircle className="h-3 w-3 shrink-0 text-destructive" />
      ) : done ? (
        <CheckCircle2 className="h-3 w-3 shrink-0 text-muted-foreground" />
      ) : (
        <span className="h-1.5 w-1.5 shrink-0 animate-pulse bg-foreground" />
      )}
      <span className="shrink-0 text-foreground">{titleFor(job)}</span>
      <span className="min-w-0 flex-1 truncate text-muted-foreground">
        {failed ? job.error : cancelled ? 'Cancelled' : job.message ?? job.stage ?? job.status}
      </span>
      {!failed && !done && !cancelled && (
        <span className="font-mono tabular-nums text-muted-foreground" title={progress.label}>
          {pct}%{progress.suffix !== 'overall' && <span className="ml-1 font-sans">{progress.suffix}</span>}
        </span>
      )}
      {onQueued && (failed || cancelled) && <Button size="sm" variant="ghost" className="h-6 text-xs" disabled={busy} onClick={() => action(true)}>Retry</Button>}
      {(job.status === 'queued' || job.status === 'running') && <Button size="sm" variant="ghost" className="h-6 text-xs" disabled={busy} onClick={() => action(false)}>Cancel</Button>}
      {onDismiss && (done || failed || cancelled) && (
        <Button
          variant="ghost"
          size="icon"
          className="h-5 w-5 text-muted-foreground"
          onClick={() => onDismiss(job.id)}
          aria-label="Dismiss"
        >
          <X className="h-3 w-3" />
        </Button>
      )}
      {!failed && !done && !cancelled && (
        <span
          className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-foreground/35"
          style={{ width: `${Math.max(4, pct)}%` }}
        />
      )}
    </motion.div>
  )
}

export function JobList({
  jobs,
  onDismiss,
  onQueued,
}: {
  jobs: Job[]
  onDismiss?: (id: string) => void
  onQueued?: (job: Job) => void
}) {
  if (jobs.length === 0) return null
  return (
    <div>
      <AnimatePresence initial={false}>
        {jobs.map((j) => (
          <JobProgress key={j.id} job={j} onDismiss={onDismiss} onQueued={onQueued} />
        ))}
      </AnimatePresence>
    </div>
  )
}
