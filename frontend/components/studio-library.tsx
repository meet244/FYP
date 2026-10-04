'use client'

import { useRef, useState } from 'react'
import {
  FileAudio,
  FileImage,
  FileText,
  Search,
  Trash2,
  Upload,
} from 'lucide-react'
import { toast } from 'sonner'
import { mutate } from 'swr'

import { Button } from '@/components/ui/button'
import { ASRSelector } from '@/components/asr-selector'
import { Skeleton } from '@/components/ui/skeleton'
import { AUDIO_FILE_ACCEPT, DOCUMENT_FILE_ACCEPT, type SubjectUploads } from '@/hooks/use-subject-uploads'
import type { SourceSelection } from '@/components/source-viewer'
import { SyllabusPanel } from '@/components/syllabus-panel'
import { CoveragePanel } from '@/components/coverage-panel'
import {
  deleteLecture,
  deleteMaterial,
} from '@/lib/api/client'
import { keys, useLectures, useMaterials } from '@/lib/api/hooks'
import { duration, relativeTime } from '@/lib/format'
import type { Job, Lecture, Material } from '@/lib/api/types'
import { jobProgress } from '@/lib/job-progress'

function activeJob(jobs: Job[], pred: (j: Job) => boolean) {
  return jobs.find(
    (j) => ['queued', 'running', 'cancelling'].includes(j.status) && pred(j)
  )
}

function stageLabel(job: Job) {
  const map: Record<string, string> = {
    audio: 'Audio',
    transcribing: 'Transcribing',
    notes: 'Notes',
    indexing: 'Indexing',
    extracting: 'Extracting',
    parsing: 'Reading',
    writing: 'Writing',
  }
  return map[job.stage ?? ''] ?? job.stage ?? job.status
}

type Row = {
  id: string
  name: string
  kind: 'voice' | 'image' | 'pdf' | 'doc'
  source: SourceSelection
  status: string
  meta: string
  job?: Job
  onDelete: () => void
}

function kindIcon(kind: Row['kind']) {
  if (kind === 'voice') return FileAudio
  if (kind === 'image') return FileImage
  return FileText
}

function kindLabel(kind: Row['kind']) {
  if (kind === 'voice') return 'Audio'
  if (kind === 'image') return 'Image'
  if (kind === 'pdf') return 'PDF'
  return 'Doc'
}

export function StudioLibrary({
  subjectId,
  kind,
  hasSyllabus,
  jobs = [],
  uploads,
  onQueued,
  onOpenSource,
}: {
  subjectId: string
  kind: 'files' | 'audio'
  hasSyllabus: boolean
  jobs?: Job[]
  uploads: SubjectUploads
  onQueued?: (job: Job) => void
  onOpenSource?: (source: SourceSelection) => void
}) {
  const isAudio = kind === 'audio'
  const { data: lectures, isLoading: lecLoading } = useLectures(isAudio ? subjectId : null)
  const { data: materials, isLoading: matLoading } = useMaterials(isAudio ? null : subjectId)
  const inputRef = useRef<HTMLInputElement>(null)
  const [query, setQuery] = useState('')
  const { asr, setAsr, busy, ingest } = uploads

  const removeMaterial = async (m: Material) => {
    try {
      await deleteMaterial(m.id)
      await mutate(keys.materials(subjectId))
      await mutate(keys.subject(subjectId))
    } catch (err) {
      toast.error('Could not delete', {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const removeLecture = async (l: Lecture) => {
    try {
      await deleteLecture(l.id)
      await mutate(keys.lectures(subjectId))
      await mutate(keys.coverage(subjectId))
    } catch (err) {
      toast.error('Could not delete', {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const rows: Row[] = [
    ...(lectures ?? []).map((l) => {
      const job = activeJob(jobs, (j) => j.kind === 'process_lecture' && j.lecture_id === l.id)
      return {
        id: l.id,
        name: l.title,
        kind: 'voice' as const,
        source: { kind: 'transcript', lecture_id: l.id, lecture_title: l.title },
        status: job ? `${stageLabel(job)} ${Math.round(jobProgress(job).value * 100)}%` : l.status,
        meta: [duration(l.duration_s), relativeTime(l.created_at)].filter(Boolean).join(' · '),
        job,
        onDelete: () => removeLecture(l),
      }
    }),
    ...(materials ?? []).map((m) => {
      const job = activeJob(jobs, (j) => j.kind === 'ingest_material' && j.material_id === m.id)
      return {
        id: m.id,
        name: m.title,
        kind: m.kind,
        source: { kind: m.kind, material_id: m.id, material_title: m.title },
        status: job ? `${stageLabel(job)} ${Math.round(jobProgress(job).value * 100)}%` : m.status,
        meta: [
          m.n_chunks ? `${m.n_chunks} chunks` : null,
          relativeTime(m.created_at),
        ]
          .filter(Boolean)
          .join(' · '),
        job,
        onDelete: () => removeMaterial(m),
      }
    }),
  ].sort((a, b) => a.name.localeCompare(b.name))

  const visible = query.trim()
    ? rows.filter((r) => r.name.toLowerCase().includes(query.trim().toLowerCase()))
    : rows

  const loading = isAudio ? lecLoading : matLoading

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="max-h-[65%] shrink-0 space-y-2 overflow-y-auto border-b border-border p-2">
        {isAudio && <ASRSelector value={asr} onChange={setAsr} disabled={busy} />}
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="flex w-full items-center gap-2 border border-dashed border-border px-2.5 py-2 text-left text-xs transition-colors hover:border-foreground/25 disabled:opacity-60"
        >
          <Upload className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="min-w-0 truncate text-muted-foreground">
            {busy ? 'Uploading…' : isAudio ? 'Add audio recordings' : 'Add files'}
          </span>
        </button>
        <div className="relative">
          <Search className="absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={isAudio ? 'Search recordings' : 'Search files'}
            aria-label={isAudio ? 'Search recordings' : 'Search files'}
            className="h-8 w-full border border-border bg-background pl-7 pr-2 text-xs outline-none focus:border-foreground/30"
          />
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple
          className="hidden"
          accept={isAudio ? AUDIO_FILE_ACCEPT : DOCUMENT_FILE_ACCEPT}
          onChange={(e) => {
            ingest(Array.from(e.target.files ?? []))
            e.target.value = ''
          }}
        />
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          {isAudio ? 'Recordings are transcribed and indexed for chat. Drop audio anywhere on the page.' : 'Add your syllabus, PDFs, PowerPoint (.pptx), Word (.docx), text, CSV or images. Drop files anywhere.'}
        </p>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="sticky top-0 z-10 grid grid-cols-[minmax(0,1fr)_44px_4.5rem] gap-1 border-b border-border bg-muted/50 px-2 py-1.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          <span>Name</span>
          <span>Kind</span>
          <span className="text-right">Status</span>
        </div>

        {loading ? (
          <div className="space-y-px p-2">
            <Skeleton className="h-9" />
            <Skeleton className="h-9" />
            <Skeleton className="h-9" />
          </div>
        ) : visible.length === 0 ? (
          <p className="px-3 py-8 text-center text-[11px] leading-relaxed text-muted-foreground">
            {rows.length === 0
              ? isAudio ? 'No recordings yet. Add audio to get a transcript and ask questions about it.' : 'No files yet. Add documents to ask questions grounded in your course material.'
              : 'No files match that search.'}
          </p>
        ) : (
          <ul>
            {visible.map((row) => {
              const Icon = kindIcon(row.kind)
              const inner = (
                <>
                  <span className="flex min-w-0 items-center gap-2">
                    <Icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                    <span className="min-w-0">
                      <span className="block truncate text-[12px] text-foreground">{row.name}</span>
                      <span className="block truncate text-[10px] text-muted-foreground">{row.meta}</span>
                    </span>
                  </span>
                  <span className="self-center text-[10px] uppercase tracking-wide text-muted-foreground">
                    {kindLabel(row.kind)}
                  </span>
                  <span className="self-center text-right text-[10px] capitalize text-muted-foreground" title={row.job ? jobProgress(row.job).label : undefined}>
                    {row.status}
                  </span>
                </>
              )
              return (
                <li key={row.id} className="group relative border-b border-border">
                  <button type="button" onClick={() => onOpenSource?.(row.source)}
                    className="grid w-full grid-cols-[minmax(0,1fr)_44px_4.5rem] gap-1 px-2 py-2 pr-8 text-left hover:bg-background">
                    {inner}
                  </button>
                  {row.job && (
                    <span
                      className="pointer-events-none absolute inset-x-0 bottom-0 h-px bg-foreground/40"
                      style={{ width: `${Math.max(4, Math.round(jobProgress(row.job).value * 100))}%` }}
                    />
                  )}
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute right-0.5 top-1.5 h-6 w-6 opacity-0 group-hover:opacity-100"
                    onClick={row.onDelete}
                    aria-label={`Delete ${row.name}`}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </li>
              )
            })}
          </ul>
        )}
        {!isAudio && hasSyllabus && <button type="button" onClick={() => onOpenSource?.({ kind: 'unit', unit_title: 'Syllabus' })}
          className="flex w-full items-center gap-2 border-y border-border px-3 py-2 text-left text-xs hover:bg-background">
          <FileText className="h-3.5 w-3.5" /> View configured syllabus
        </button>}
        {!isAudio && onQueued && <details className="border-t border-border p-3">
          <summary className="cursor-pointer text-xs text-muted-foreground">Syllabus settings (optional)</summary>
          <p className="my-3 text-[11px] leading-relaxed text-muted-foreground">Files are searchable automatically. Configure a syllabus here only to enable curriculum coverage and syllabus-guided transcription.</p>
          <SyllabusPanel subjectId={subjectId} onQueued={onQueued} />
          <div className="mt-5"><CoveragePanel subjectId={subjectId} /></div>
        </details>}
      </div>
    </div>
  )
}
