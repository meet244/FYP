'use client'

import { useEffect, useId, useState } from 'react'
import { Check, ChevronDown, ChevronUp, LoaderCircle } from 'lucide-react'

import { cn } from '@/lib/utils'
import { useLecture, useMaterials } from '@/lib/api/hooks'
import type { Job } from '@/lib/api/types'
import { jobProgress } from '@/lib/job-progress'

function titleFor(job: Job) {
  if (job.status === 'cancelling') return 'Stopping safely'
  if (job.status === 'queued') return 'Waiting in the queue'
  const titles: Record<string, string> = {
    audio: 'Preparing your recording',
    transcribing: 'Turning speech into text',
    notes: 'Creating your study notes',
    indexing: 'Making your source searchable',
    extracting: 'Reading your document',
    parsing: 'Reading your syllabus',
    writing: 'Organizing syllabus topics',
  }
  return titles[job.stage ?? ''] ?? 'Preparing your source'
}

function stageLine(job: Job) {
  if (job.status === 'cancelling') return 'The current step will finish before processing stops.'
  if (job.status === 'queued') return 'Processing starts when the current source is finished.'
  const segment = job.message?.match(/(first|second) pass (\d+)\/(\d+)/i)
  if (segment) return `Segment ${segment[2]} of ${segment[3]}${segment[1].toLowerCase() === 'second' ? ' · checking syllabus context' : ''}`
  if (/loading|download/i.test(job.message ?? '')) return 'Loading the speech model. The first run may download model files.'
  const details: Record<string, string> = {
    audio: 'Preparing the audio for accurate speech recognition.',
    transcribing: 'Recognizing speech with your selected model.',
    notes: 'Organizing the transcript into useful study notes.',
    indexing: 'Connecting this source to your subject’s chat.',
    extracting: 'Extracting the content from your file.',
    parsing: 'Finding the topics and structure in your syllabus.',
    writing: 'Saving the topics for retrieval and coverage.',
  }
  return details[job.stage ?? ''] ?? 'Your source is being prepared in the background.'
}

function stepsFor(job: Job) {
  if (job.kind === 'ingest_material') return [
    { label: 'Read file', stages: ['extracting'] },
    { label: 'Ready for chat', stages: ['indexing'] },
  ]
  if (job.kind === 'ingest_syllabus') return [
    { label: 'Read PDF', stages: ['parsing'] },
    { label: 'Organize topics', stages: ['writing'] },
    { label: 'Ready for chat', stages: ['indexing'] },
  ]
  if (job.kind === 'generate_notes') return [
    { label: 'Study notes', stages: ['notes'] },
    { label: 'Ready for chat', stages: ['indexing'] },
  ]
  return [
    { label: 'Prepare audio', stages: ['audio'] },
    { label: 'Transcript', stages: ['transcribing'] },
    { label: 'Study notes', stages: ['notes'] },
    { label: 'Ready for chat', stages: ['indexing'] },
  ]
}

export function SgcdMachine({ compact, label, detail, progress, paused = false }: {
  compact?: boolean
  label?: string
  detail?: string
  progress?: number
  paused?: boolean
}) {
  const pct = progress != null ? Math.round(Math.min(1, Math.max(0, progress)) * 100) : null
  return (
    <div className={cn('sgcd-machine', compact && 'sgcd-machine-compact', paused && 'sgcd-machine-paused')}
      role="status" aria-live="polite" aria-atomic="true">
      <div className="sgcd-orbit" aria-hidden="true">
        <span className="sgcd-ring sgcd-ring-outer" />
        <span className="sgcd-ring sgcd-ring-inner" />
        <span className="sgcd-core">
          <span className="sgcd-mark">CS</span>
          <span className="sgcd-wave">
            {Array.from({ length: 9 }, (_, index) => <i key={index} style={{ '--i': index } as React.CSSProperties} />)}
          </span>
        </span>
      </div>
      <div className="sgcd-copy">
        {label && <p className="sgcd-label">{label}</p>}
        {detail && <p className="sgcd-detail">{detail}</p>}
        {pct != null && <div className="sgcd-pct">
          <span className="font-mono tabular-nums">{pct}%</span>
          <div className="sgcd-bar" role="progressbar" aria-label={label ?? 'Processing'} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
            <span style={{ width: `${pct}%` }} />
          </div>
        </div>}
      </div>
    </div>
  )
}

function ProcessingPanel({ job, jobs, minimized, onToggle }: {
  job: Job
  jobs: Job[]
  minimized: boolean
  onToggle: () => void
}) {
  const contentId = useId()
  const { data: lecture } = useLecture(job.lecture_id)
  const { data: materials } = useMaterials(job.kind === 'ingest_material' ? job.subject_id : null)
  const name = lecture?.title ?? materials?.find(material => material.id === job.material_id)?.title
    ?? (job.kind === 'ingest_syllabus' ? 'Subject syllabus' : 'Your source')
  const queued = jobs.filter(item => item.status === 'queued').length
  const running = jobs.filter(item => ['running', 'cancelling'].includes(item.status)).length
  const steps = stepsFor(job)
  const current = job.status === 'queued' ? -1 : steps.findIndex(step => step.stages.includes(job.stage ?? ''))
  const progress = jobProgress(job)
  const pct = Math.round(progress.value * 100)
  const model = job.asr_config?.model?.split('/').pop()?.replace(/-hf$/, '')

  return (
    <aside aria-label="Source processing" className={cn('processing-panel', minimized && 'processing-panel-minimized')}>
      <button type="button" onClick={onToggle} aria-expanded={!minimized} aria-controls={contentId}
        aria-label={minimized ? 'Expand processing panel' : 'Minimize processing panel'}
        className="processing-panel-toggle">
        <span className={cn('processing-dot', running > 0 && 'processing-dot-active')} aria-hidden="true" />
        <span className="min-w-0 flex-1 truncate text-left">
          {minimized ? `${titleFor(job)} · ${pct}%` : running > 0 ? `${running} source${running === 1 ? '' : 's'} processing` : 'Sources queued'}
        </span>
        {queued > 0 && <span className="processing-queue">{queued} queued</span>}
        {minimized ? <ChevronUp className="h-3.5 w-3.5 shrink-0" /> : <ChevronDown className="h-3.5 w-3.5 shrink-0" />}
      </button>
      <div id={contentId} className="processing-panel-body" hidden={minimized}>
        <div className="processing-overview">
        <p className="processing-source" title={name}>{name}</p>
        <SgcdMachine compact label={titleFor(job)} detail={stageLine(job)} paused={job.status !== 'running'} />
        {model && <p className="processing-model">{model} · {job.asr_config?.language === 'hi' ? 'Hindi + English' : job.asr_config?.language === 'en' ? 'English' : 'Auto language'}</p>}
        </div>
        <div className="processing-milestones">
        <div className="processing-progress-label"><span>{progress.label}</span><span className="font-mono tabular-nums text-foreground">{pct}%</span></div>
        <div className="sgcd-bar" role="progressbar" aria-label={`${progress.label}: ${name}`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
          <span style={{ width: `${pct}%` }} />
        </div>
        <ol className="processing-steps" aria-label="Processing stages">
          {steps.map((step, index) => <li key={step.label} className={cn('processing-step', index < current && 'processing-step-done', index === current && 'processing-step-current')}
            aria-current={index === current ? 'step' : undefined}>
            <span className="processing-step-icon" aria-hidden="true">
              {index < current ? <Check className="h-3 w-3" /> : index === current ? <LoaderCircle className={cn('h-3 w-3', job.status === 'running' && 'processing-spinner')} /> : <span className="h-1 w-1 rounded-full bg-current" />}
            </span>
            <span>{step.label}</span>
          </li>)}
        </ol>
        <p className="processing-footnote">Keep chatting while we work. Your new sources appear when ready.</p>
        </div>
      </div>
    </aside>
  )
}

export function ProcessingStatus({ jobs }: { jobs: Job[] }) {
  const [minimized, setMinimized] = useState(false)
  useEffect(() => {
    const narrow = window.matchMedia('(max-width: 767px), (max-height: 700px)')
    const adapt = () => setMinimized(narrow.matches)
    adapt()
    narrow.addEventListener('change', adapt)
    return () => narrow.removeEventListener('change', adapt)
  }, [])
  const active = jobs.filter(job => ['queued', 'running', 'cancelling'].includes(job.status))
  const job = active.find(item => item.status === 'running')
    ?? active.find(item => item.status === 'cancelling') ?? active[0]
  if (!job) return null
  return <ProcessingPanel job={job} jobs={active} minimized={minimized} onToggle={() => setMinimized(value => !value)} />
}
