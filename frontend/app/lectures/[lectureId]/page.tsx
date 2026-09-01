'use client'

import { Suspense, use, useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { useSearchParams } from 'next/navigation'
import {
  AlertTriangle,
  ArrowLeft,
  FileText,
  Gauge,
  Layers,
  RefreshCw,
  ServerCrash,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import { toast } from 'sonner'

import { AppHeader } from '@/components/app-header'
import { AudioPlayer, type AudioPlayerHandle } from '@/components/audio-player'
import { JobList } from '@/components/job-progress'
import { NotesView } from '@/components/notes-view'
import { TranscriptView } from '@/components/transcript-view'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { reprocessLecture } from '@/lib/api/client'
import {
  revalidateAfterJob,
  useJobTracker,
  useLecture,
  useNotes,
  useSyllabus,
  useTranscript,
} from '@/lib/api/hooks'
import { duration } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { AsrStats } from '@/lib/api/types'

function StatCard({
  icon: Icon,
  label,
  value,
  tip,
  warn,
}: {
  icon: React.ElementType
  label: string
  value: string
  tip: string
  warn?: boolean
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div
          className={cn(
            'rounded-xl border bg-card p-3',
            warn ? 'border-amber-500/30' : 'border-border/60'
          )}
        >
          <div className="mb-1 flex items-center gap-1.5">
            <Icon className={cn('h-3.5 w-3.5', warn ? 'text-amber-500' : 'text-primary')} />
            <span className="text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
              {label}
            </span>
          </div>
          <p
            className={cn(
              'text-lg font-black tabular-nums leading-none tracking-tight',
              warn ? 'text-amber-500' : 'text-foreground'
            )}
          >
            {value}
          </p>
        </div>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-64 text-xs leading-relaxed">
        {tip}
      </TooltipContent>
    </Tooltip>
  )
}

function AsrStatsGrid({ stats }: { stats: AsrStats }) {
  const scriptMix = Object.entries(stats.script_mix)
    .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`)
    .join(' · ')

  return (
    <TooltipProvider delayDuration={200}>
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-4">
        <StatCard
          icon={Layers}
          label="Spans"
          value={`${stats.n_spans}`}
          tip={`Mean ${stats.mean_span_s}s. Conditioning only helps on lecture-length spans — at ~5.7s it regresses WER by 5.11 points, at ~26s it gains 6.23.`}
          warn={stats.mean_span_s < 20}
        />
        <StatCard
          icon={Gauge}
          label="Speed"
          value={stats.realtime_factor ? `${stats.realtime_factor}×` : '—'}
          tip={`Decoded in ${Math.round(stats.elapsed_s)}s of wall clock, two passes over the whole recording.`}
        />
        <StatCard
          icon={ShieldCheck}
          label="Reverted"
          value={`${(stats.safeguard_fallback_rate * 100).toFixed(0)}%`}
          tip={`${stats.safeguard_fallbacks} of ${stats.n_spans} spans fell back to the unconditioned hypothesis. Safeguard thresholds do not transfer across checkpoints — a high rate means refit them for this model.`}
          warn={stats.safeguard_fallback_rate > 0.25}
        />
        <StatCard
          icon={FileText}
          label="Script mix"
          value={scriptMix || '—'}
          tip="Descriptive check that the dual-script convention survived — not a WER proxy. A collapse in the Latin share is the Devanagari-transliteration failure mode."
        />
      </div>
    </TooltipProvider>
  )
}

function LectureView({ lectureId }: { lectureId: string }) {
  const search = useSearchParams()
  const focusNoteId = search.get('note')
  const seekParam = search.get('t')

  const { data: lecture, error } = useLecture(lectureId)
  const { data: transcript } = useTranscript(lectureId)
  const { data: notes } = useNotes(lectureId)
  const { data: syllabus } = useSyllabus(lecture?.subject_id ?? null)
  const { jobs, track, dismiss } = useJobTracker(revalidateAfterJob)

  const playerRef = useRef<AudioPlayerHandle>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const seeked = useRef(false)

  // A citation arrives as ?t=<seconds>; honour it once the audio can accept a seek.
  useEffect(() => {
    if (seeked.current || !seekParam || !transcript) return
    const t = Number(seekParam)
    if (Number.isFinite(t)) {
      seeked.current = true
      setCurrentTime(t)
      // One tick, so the <audio> element has mounted with its metadata.
      const id = setTimeout(() => playerRef.current?.seek(t), 300)
      return () => clearTimeout(id)
    }
  }, [seekParam, transcript])

  const reprocess = async () => {
    try {
      track(await reprocessLecture(lectureId))
      toast.success('Re-decoding queued')
    } catch (err) {
      toast.error('Could not reprocess', {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <AppHeader />
        <div className="mx-auto max-w-2xl px-6 py-24 text-center">
          <ServerCrash className="mx-auto mb-3 h-8 w-8 text-destructive" />
          <h2 className="mb-1 font-semibold text-foreground">Lecture unavailable</h2>
          <p className="mb-6 text-sm text-muted-foreground">{error.message}</p>
          <Button asChild variant="outline">
            <Link href="/">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back to subjects
            </Link>
          </Button>
        </div>
      </div>
    )
  }

  const stats = lecture?.asr_stats ?? transcript?.asr_stats ?? null

  return (
    <div className="min-h-screen bg-background">
      <AppHeader>
        <div className="flex min-w-0 items-center gap-2">
          <Button asChild variant="ghost" size="icon" className="h-7 w-7 shrink-0">
            <Link
              href={lecture ? `/subjects/${lecture.subject_id}` : '/'}
              aria-label="Back to subject"
            >
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          {lecture ? (
            <span className="truncate text-sm font-semibold text-foreground">{lecture.title}</span>
          ) : (
            <Skeleton className="h-4 w-40" />
          )}
        </div>
      </AppHeader>

      <main className="mx-auto max-w-4xl px-5 py-6">
        <JobList jobs={jobs} onDismiss={dismiss} />

        <div className="mb-4 flex flex-wrap items-center gap-2">
          {lecture && (
            <>
              <Badge variant="secondary" className="text-[10px]">
                {duration(lecture.duration_s)}
              </Badge>
              <Badge variant="outline" className="text-[10px] capitalize">
                {lecture.status}
              </Badge>
              {stats && !stats.conditioned && (
                <Badge
                  variant="outline"
                  className="gap-1 border-amber-500/30 text-[10px] text-amber-600 dark:text-amber-400"
                >
                  <AlertTriangle className="h-2.5 w-2.5" /> unconditioned
                </Badge>
              )}
            </>
          )}
          <Button
            variant="outline"
            size="sm"
            className="ml-auto h-7 gap-1.5 text-xs"
            onClick={reprocess}
          >
            <RefreshCw className="h-3 w-3" /> Reprocess
          </Button>
        </div>

        {lecture?.error && (
          <p className="mb-4 rounded-xl border border-amber-500/30 bg-amber-500/5 p-3 text-xs leading-relaxed text-amber-600 dark:text-amber-400">
            {lecture.error}
          </p>
        )}

        {lecture && (
          <AudioPlayer
            ref={playerRef}
            lectureId={lectureId}
            title={lecture.title}
            onTime={setCurrentTime}
            className="mb-4"
          />
        )}

        {stats && (
          <div className="mb-6">
            <AsrStatsGrid stats={stats} />
          </div>
        )}

        <Tabs defaultValue={focusNoteId ? 'notes' : 'transcript'}>
          <TabsList className="mb-4">
            <TabsTrigger value="transcript" className="gap-1.5 text-xs">
              <FileText className="h-3.5 w-3.5" /> Transcript
              {transcript && (
                <span className="ml-0.5 text-muted-foreground">{transcript.spans.length}</span>
              )}
            </TabsTrigger>
            <TabsTrigger value="notes" className="gap-1.5 text-xs">
              <Sparkles className="h-3.5 w-3.5" /> Notes
              {notes && notes.length > 0 && (
                <span className="ml-0.5 text-muted-foreground">{notes.length}</span>
              )}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="transcript" className="mt-0">
            {transcript ? (
              <TranscriptView
                spans={transcript.spans}
                units={syllabus?.units ?? []}
                currentTime={currentTime}
                onSeek={(s) => playerRef.current?.seek(s)}
              />
            ) : (
              <div className="space-y-2">
                {[0, 1, 2, 3].map((i) => (
                  <Skeleton key={i} className="h-12 rounded-lg" />
                ))}
              </div>
            )}
          </TabsContent>

          <TabsContent value="notes" className="mt-0">
            {notes ? (
              <NotesView
                notes={notes}
                units={syllabus?.units ?? []}
                focusNoteId={focusNoteId}
                onSeek={(s) => playerRef.current?.seek(s)}
              />
            ) : (
              <div className="space-y-3">
                {[0, 1].map((i) => (
                  <Skeleton key={i} className="h-40 rounded-xl" />
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

export default function LecturePage({
  params,
}: {
  params: Promise<{ lectureId: string }>
}) {
  const { lectureId } = use(params)
  // useSearchParams needs a Suspense boundary to prerender — the `?t=` and
  // `?note=` params arrive from chat citations and coverage links.
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-background">
          <AppHeader />
          <main className="mx-auto max-w-4xl space-y-4 px-5 py-6">
            <Skeleton className="h-14 rounded-xl" />
            <Skeleton className="h-20 rounded-xl" />
            <Skeleton className="h-64 rounded-xl" />
          </main>
        </div>
      }
    >
      <LectureView lectureId={lectureId} />
    </Suspense>
  )
}
