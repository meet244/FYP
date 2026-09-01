'use client'

import Link from 'next/link'
import {
  AlertTriangle,
  Clock,
  Gauge,
  Layers,
  Mic,
  RefreshCw,
  ShieldCheck,
  Trash2,
} from 'lucide-react'
import { toast } from 'sonner'
import { mutate } from 'swr'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { UploadLectureDialog } from '@/components/upload-lecture-dialog'
import { deleteLecture, reprocessLecture } from '@/lib/api/client'
import { keys, useLectures } from '@/lib/api/hooks'
import { duration, relativeTime } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { Job, Lecture, LectureStatus } from '@/lib/api/types'

const STATUS: Record<LectureStatus, { label: string; className: string; pulse?: boolean }> = {
  uploaded: { label: 'Queued', className: 'border-border text-muted-foreground', pulse: true },
  transcribing: {
    label: 'Transcribing',
    className: 'border-primary/30 bg-primary/10 text-primary',
    pulse: true,
  },
  transcribed: {
    label: 'Transcribed',
    className: 'border-blue-500/30 bg-blue-500/10 text-blue-600 dark:text-blue-400',
  },
  summarising: {
    label: 'Writing notes',
    className: 'border-primary/30 bg-primary/10 text-primary',
    pulse: true,
  },
  ready: {
    label: 'Ready',
    className: 'border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400',
  },
}

function Stat({
  icon: Icon,
  value,
  tip,
  className,
}: {
  icon: React.ElementType
  value: string
  tip: string
  className?: string
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className={cn('flex items-center gap-1.5 text-xs text-muted-foreground', className)}>
          <Icon className="h-3.5 w-3.5" />
          {value}
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-56 text-xs">
        {tip}
      </TooltipContent>
    </Tooltip>
  )
}

function LectureRow({ lecture }: { lecture: Lecture }) {
  const status = STATUS[lecture.status] ?? STATUS.uploaded
  const stats = lecture.asr_stats

  const reprocess = async () => {
    try {
      await reprocessLecture(lecture.id)
      await mutate(keys.lectures(lecture.subject_id))
      toast.success('Re-decoding queued', {
        description: 'The same audio conditioned on the current syllabus.',
      })
    } catch (err) {
      toast.error('Could not reprocess', {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const remove = async () => {
    try {
      await deleteLecture(lecture.id)
      await mutate(keys.lectures(lecture.subject_id))
      await mutate(keys.coverage(lecture.subject_id))
      toast.success(`Deleted “${lecture.title}”`)
    } catch (err) {
      toast.error('Could not delete', {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  return (
    <div className="group relative rounded-xl border border-border/60 bg-card p-4 transition-colors hover:border-primary/40">
      <Link href={`/lectures/${lecture.id}`} className="block">
        <div className="mb-2 flex items-start gap-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10">
            <Mic className="h-4 w-4 text-primary" />
          </div>

          <div className="min-w-0 flex-1">
            <h4 className="truncate text-sm font-semibold text-foreground">{lecture.title}</h4>
            <p className="text-xs text-muted-foreground">
              {relativeTime(lecture.recorded_at ?? lecture.created_at)} · {duration(lecture.duration_s)}
            </p>
          </div>

          <Badge variant="outline" className={cn('shrink-0 gap-1.5 text-[10px]', status.className)}>
            {status.pulse && (
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
            )}
            {status.label}
          </Badge>
        </div>

        {stats && (
          <TooltipProvider delayDuration={200}>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pl-12">
              <Stat
                icon={Layers}
                value={`${stats.n_spans} spans · ${stats.mean_span_s}s`}
                tip="Mean span length. Conditioning only helps on lecture-length spans — below ~20s it regresses."
              />
              {stats.realtime_factor != null && (
                <Stat
                  icon={Gauge}
                  value={`${stats.realtime_factor}× real time`}
                  tip={`Decoded in ${Math.round(stats.elapsed_s)}s.`}
                />
              )}
              <Stat
                icon={ShieldCheck}
                value={`${(stats.safeguard_fallback_rate * 100).toFixed(0)}% reverted`}
                tip={`${stats.safeguard_fallbacks} of ${stats.n_spans} spans fell back to the unconditioned hypothesis. A high rate means the thresholds need refitting for this checkpoint.`}
                className={cn(stats.safeguard_fallback_rate > 0.25 && 'text-amber-500')}
              />
              {!stats.conditioned && (
                <Stat
                  icon={AlertTriangle}
                  value="Unconditioned"
                  tip="Decoded without a syllabus — a plain single-pass baseline. Reprocess now that one exists."
                  className="text-amber-500"
                />
              )}
            </div>
          </TooltipProvider>
        )}

        {lecture.error && (
          <p className="mt-2 pl-12 text-xs leading-relaxed text-amber-600 dark:text-amber-400">
            {lecture.error}
          </p>
        )}
      </Link>

      <div className="absolute right-2 top-2 flex gap-0.5 opacity-0 transition-opacity group-hover:opacity-100">
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 text-muted-foreground"
          onClick={reprocess}
          title="Re-decode against the current syllabus"
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>

        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7 text-muted-foreground hover:text-destructive"
              title="Delete lecture"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete “{lecture.title}”?</AlertDialogTitle>
              <AlertDialogDescription>
                Removes the transcript, notes, normalised audio and vector index entries. The
                original upload stays on disk.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={remove}
                className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </div>
  )
}

export function LectureList({
  subjectId,
  hasSyllabus,
  onQueued,
}: {
  subjectId: string
  hasSyllabus: boolean
  onQueued: (job: Job) => void
}) {
  const { data: lectures, isLoading } = useLectures(subjectId)

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[0, 1].map((i) => (
          <Skeleton key={i} className="h-24 rounded-xl" />
        ))}
      </div>
    )
  }

  if (!lectures || lectures.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border bg-card/40 p-12 text-center">
        <div className="mx-auto mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10">
          <Mic className="h-6 w-6 text-primary" />
        </div>
        <h4 className="mb-1.5 font-semibold text-foreground">No lectures yet</h4>
        <p className="mx-auto mb-5 max-w-sm text-sm leading-relaxed text-muted-foreground">
          {hasSyllabus
            ? 'Upload a classroom recording. Two decode passes, then notes and an index — it runs for a few minutes.'
            : 'Upload the syllabus first. Without one the pipeline runs a single unconditioned pass and none of the method applies.'}
        </p>
        <UploadLectureDialog
          subjectId={subjectId}
          hasSyllabus={hasSyllabus}
          onQueued={onQueued}
        />
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {lectures.map((l) => (
        <LectureRow key={l.id} lecture={l} />
      ))}
    </div>
  )
}
