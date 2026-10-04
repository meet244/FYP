'use client'

import { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import useSWR from 'swr'

import { AudioPlayer, type AudioPlayerHandle } from '@/components/audio-player'
import { Markdown } from '@/components/markdown'
import { NotesView } from '@/components/notes-view'
import { TranscriptView } from '@/components/transcript-view'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { getMaterialPreview, materialFileUrl, syllabusFileUrl } from '@/lib/api/client'
import { keys, useLecture, useNotes, useSyllabus, useTranscript } from '@/lib/api/hooks'
import type { Citation } from '@/lib/api/types'

export type SourceSelection = Partial<Citation>

function MaterialPreview({ source }: { source: SourceSelection }) {
  const { data, error } = useSWR(keys.material(source.material_id!), () => getMaterialPreview(source.material_id!), {
    refreshInterval: material => material && ['ready', 'failed'].includes(material.status) ? 0 : 1500,
  })
  const [showText, setShowText] = useState(false)
  if (error) return <p className="p-6 text-sm text-destructive">Could not open this file: {error.message}</p>
  if (!data) return <p className="p-6 text-sm text-muted-foreground">Loading file…</p>

  const url = materialFileUrl(data.id)
  const canShowOriginal = data.kind === 'pdf' || data.kind === 'image'
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center justify-between gap-3 border-b border-border px-4 py-2 text-xs text-muted-foreground">
        <span className="truncate">{data.original_filename ?? data.title} · {data.status === 'ready' ? 'Ready for chat' : data.status}</span>
        {canShowOriginal && <Button size="sm" variant="ghost" onClick={() => setShowText(value => !value)}>
          {showText ? 'View original' : 'View extracted text'}
        </Button>}
      </div>
      {data.error && <p className="px-4 py-2 text-sm text-destructive">{data.error}</p>}
      {!showText && data.kind === 'pdf' ? (
        <iframe title={data.title} src={`${url}#page=${source.page ?? 1}`} className="min-h-0 w-full flex-1 border-0 bg-white" />
      ) : !showText && data.kind === 'image' ? (
        <div className="min-h-0 flex-1 overflow-auto p-4"><img src={url} alt={data.title} className="mx-auto max-h-full max-w-full object-contain" /></div>
      ) : (
        <div className="min-h-0 flex-1 overflow-auto p-6">
          <p className="mb-3 text-xs text-muted-foreground">Extracted document text</p>
          <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-relaxed text-foreground">
            {data.text || 'Text is not available yet. Processing progress is shown in the subject library.'}
          </pre>
        </div>
      )}
    </div>
  )
}

function RecordingPreview({ source }: { source: SourceSelection }) {
  const id = source.lecture_id!
  const { data: lecture, error } = useLecture(id)
  const { data: transcript, error: transcriptError } = useTranscript(id)
  const { data: notes } = useNotes(id)
  const { data: syllabus } = useSyllabus(lecture?.subject_id ?? null)
  const player = useRef<AudioPlayerHandle>(null)
  const [currentTime, setCurrentTime] = useState(source.start_s ?? 0)
  useEffect(() => {
    if (lecture?.duration_s != null && source.start_s != null) player.current?.seek(source.start_s)
  }, [lecture?.duration_s, source.start_s])

  if (error) return <p className="p-6 text-sm text-destructive">Could not open this recording: {error.message}</p>
  if (!lecture) return <p className="p-6 text-sm text-muted-foreground">Loading recording…</p>
  const units = syllabus?.units ?? []
  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-6">
      <div className="mb-4 flex items-center justify-between text-xs text-muted-foreground">
        <span>{lecture.status === 'ready' ? 'Transcribed and indexed for chat' : lecture.status}</span>
        <Link href={`/lectures/${id}${source.start_s != null ? `?t=${source.start_s}` : ''}`} className="underline">Recording details</Link>
      </div>
      {lecture.error && <p className="mb-3 text-sm text-destructive">{lecture.error}</p>}
      {lecture.duration_s != null && <AudioPlayer ref={player} lectureId={id} title={lecture.title} onTime={setCurrentTime} className="mb-5" />}
      {source.note_id && notes && <NotesView notes={notes.filter(note => note.id === source.note_id)} units={units} focusNoteId={source.note_id} onSeek={seconds => player.current?.seek(seconds)} />}
      <h3 className="mb-3 font-display text-xl">Transcript</h3>
      {transcript?.spans.length ? <TranscriptView spans={transcript.spans} units={units} currentTime={currentTime} onSeek={seconds => player.current?.seek(seconds)} /> : (
        <p className="text-sm text-muted-foreground">{transcriptError ? 'The transcript is not available yet.' : 'Waiting for the transcript. Follow processing progress in Audio recordings.'}</p>
      )}
    </div>
  )
}

function SyllabusPreview({ source, subjectId }: { source: SourceSelection; subjectId: string }) {
  const { data, error } = useSyllabus(subjectId)
  if (error) return <p className="p-6 text-sm text-destructive">Could not open the syllabus: {error.message}</p>
  if (!data) return <p className="p-6 text-sm text-muted-foreground">Loading syllabus…</p>
  const unit = data.units.find(item => item.id === source.unit_id)
  return <div className="flex min-h-0 flex-1 flex-col">
    {unit && <div className="max-h-48 overflow-y-auto border-b border-border p-4"><h3 className="mb-2 font-display text-xl">{unit.title}</h3><Markdown>{unit.prose}</Markdown></div>}
    <iframe title="Syllabus PDF" src={syllabusFileUrl(subjectId)} className="min-h-0 w-full flex-1 border-0 bg-white" />
  </div>
}

export function SourceViewer({ source, subjectId, onClose }: {
  source: SourceSelection | null
  subjectId: string
  onClose: () => void
}) {
  const title = source?.material_title ?? source?.lecture_title ?? source?.unit_title ?? 'Source'
  const sourceKey = `${source?.material_id ?? source?.lecture_id ?? source?.unit_id ?? 'syllabus'}-${source?.start_s ?? ''}-${source?.note_id ?? ''}-${source?.n ?? ''}`
  return <Dialog open={!!source} onOpenChange={open => { if (!open) onClose() }}>
    <DialogContent className="flex h-[90dvh] w-[calc(100%-2rem)] max-w-5xl flex-col gap-0 overflow-hidden p-0 sm:max-w-5xl">
      <DialogHeader className="shrink-0 border-b border-border px-5 py-4 pr-12 text-left">
        <DialogTitle className="font-display text-xl font-normal">{title}</DialogTitle>
        <DialogDescription>{source?.n ? `Source ${source.n}${source.timestamp ? ` · ${source.timestamp}` : ''}` : 'Preview this subject source'} · Close to return to your chat.</DialogDescription>
      </DialogHeader>
      {source?.excerpt && <blockquote className="max-h-32 shrink-0 overflow-y-auto border-b border-border bg-muted/40 px-5 py-3 text-sm leading-relaxed"><p className="mb-1 text-xs font-medium text-muted-foreground">Cited passage</p>{source.excerpt}</blockquote>}
      {source && (source.material_id ? <MaterialPreview key={sourceKey} source={source} /> : source.lecture_id ? <RecordingPreview key={sourceKey} source={source} /> : <SyllabusPreview key={sourceKey} subjectId={subjectId} source={source} />)}
    </DialogContent>
  </Dialog>
}
