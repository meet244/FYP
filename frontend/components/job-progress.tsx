'use client'

import { AlertCircle, CheckCircle2, FileText, Mic, Search, Sparkles, Volume2, X } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'

import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'
import type { Job } from '@/lib/api/types'

/**
 * The stages `jobs/handlers.py` reports, in order. Showing them as a track
 * rather than a bare percentage matters here: transcription is the long stage,
 * and a student watching 40% for four minutes needs to know it is decoding and
 * not stuck.
 */
const LECTURE_STAGES = [
  { key: 'audio', label: 'Audio', icon: Volume2 },
  { key: 'transcribing', label: 'SGCD decode', icon: Mic },
  { key: 'notes', label: 'Notes', icon: Sparkles },
  { key: 'indexing', label: 'Index', icon: Search },
] as const

const SYLLABUS_STAGES = [
  { key: 'parsing', label: 'Reading PDF', icon: FileText },
  { key: 'writing', label: 'Writing units', icon: Sparkles },
] as const

function stagesFor(kind: Job['kind']) {
  return kind === 'ingest_syllabus' ? SYLLABUS_STAGES : LECTURE_STAGES
}

export function JobProgress({
  job,
  onDismiss,
  className,
}: {
  job: Job
  onDismiss?: (id: string) => void
  className?: string
}) {
  const stages = stagesFor(job.kind)
  const done = job.status === 'succeeded'
  const failed = job.status === 'failed'
  const currentIdx = stages.findIndex((s) => s.key === job.stage)

  const title = job.kind === 'ingest_syllabus' ? 'Parsing syllabus' : 'Processing lecture'

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, height: 0, marginBottom: 0 }}
      className={cn(
        'overflow-hidden rounded-xl border p-3.5',
        failed
          ? 'border-destructive/30 bg-destructive/5'
          : done
            ? 'border-emerald-500/30 bg-emerald-500/5'
            : 'border-primary/25 bg-primary/5',
        className
      )}
    >
      <div className="mb-2.5 flex items-center gap-2">
        {failed ? (
          <AlertCircle className="h-4 w-4 shrink-0 text-destructive" />
        ) : done ? (
          <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-500" />
        ) : (
          <span className="relative flex h-2 w-2 shrink-0">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
          </span>
        )}

        <span className="text-sm font-semibold text-foreground">
          {failed ? `${title} failed` : done ? `${title.replace(/ing\b/, 'ed')}` : title}
        </span>

        {!failed && !done && (
          <span className="ml-auto font-mono text-xs tabular-nums text-muted-foreground">
            {Math.round(job.progress * 100)}%
          </span>
        )}

        {onDismiss && (done || failed) && (
          <Button
            variant="ghost"
            size="icon"
            className="ml-auto h-6 w-6 text-muted-foreground"
            onClick={() => onDismiss(job.id)}
            aria-label="Dismiss"
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>

      {failed ? (
        <p className="font-mono text-xs leading-relaxed text-destructive">{job.error}</p>
      ) : (
        <>
          <Progress value={job.progress * 100} className="mb-2.5 h-1.5" />

          <div className="flex items-center gap-1">
            {stages.map((s, i) => {
              const Icon = s.icon
              const active = i === currentIdx && !done
              const passed = done || (currentIdx >= 0 && i < currentIdx)
              return (
                <div
                  key={s.key}
                  className={cn(
                    'flex items-center gap-1.5 rounded-md px-1.5 py-1 text-[10px] font-medium transition-colors',
                    active
                      ? 'bg-primary/15 text-primary'
                      : passed
                        ? 'text-emerald-600 dark:text-emerald-400'
                        : 'text-muted-foreground/50'
                  )}
                >
                  <Icon className={cn('h-3 w-3', active && 'animate-pulse')} />
                  <span className="hidden sm:inline">{s.label}</span>
                </div>
              )
            })}
          </div>

          {job.message && (
            <p className="mt-2 truncate text-xs text-muted-foreground">{job.message}</p>
          )}
        </>
      )}
    </motion.div>
  )
}

export function JobList({
  jobs,
  onDismiss,
}: {
  jobs: Job[]
  onDismiss?: (id: string) => void
}) {
  if (jobs.length === 0) return null
  return (
    <div className="mb-4 space-y-2">
      <AnimatePresence initial={false}>
        {jobs.map((j) => (
          <JobProgress key={j.id} job={j} onDismiss={onDismiss} />
        ))}
      </AnimatePresence>
    </div>
  )
}
