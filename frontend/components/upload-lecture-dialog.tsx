'use client'

import { useRef, useState } from 'react'
import { FileAudio, Upload, X } from 'lucide-react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Spinner } from '@/components/ui/spinner'
import { uploadLecture } from '@/lib/api/client'
import { bytes } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { Job } from '@/lib/api/types'

/** Mirrors AUDIO_SUFFIXES in `api/lectures.py` — reject before the round trip. */
const AUDIO_SUFFIXES = [
  '.m4a', '.mp3', '.wav', '.aac', '.ogg', '.opus',
  '.flac', '.mp4', '.3gp', '.amr', '.webm',
]

export function UploadLectureDialog({
  subjectId,
  hasSyllabus,
  onQueued,
  trigger,
}: {
  subjectId: string
  hasSyllabus: boolean
  onQueued: (job: Job) => void
  trigger?: React.ReactNode
}) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [recordedAt, setRecordedAt] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const accept = (f: File | undefined) => {
    if (!f) return
    const suffix = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
    if (!AUDIO_SUFFIXES.includes(suffix)) {
      toast.error(`Unsupported format ${suffix}`, {
        description: `Expected one of ${AUDIO_SUFFIXES.join(', ')}`,
      })
      return
    }
    setFile(f)
    if (!title) setTitle(f.name.replace(/\.[^.]+$/, ''))
  }

  const reset = () => {
    setFile(null)
    setTitle('')
    setRecordedAt('')
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return
    setBusy(true)
    try {
      const job = await uploadLecture(subjectId, file, {
        title: title.trim() || undefined,
        // <input type="datetime-local"> already emits ISO 8601 without a zone,
        // which is what the endpoint parses with fromisoformat.
        recordedAt: recordedAt || undefined,
      })
      onQueued(job)
      setOpen(false)
      reset()
      toast.success('Lecture queued', { description: 'Transcription runs in the background.' })
    } catch (err) {
      toast.error('Upload failed', {
        description: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o)
        if (!o) reset()
      }}
    >
      <DialogTrigger asChild>
        {trigger ?? (
          <Button size="sm" className="gap-1.5">
            <Upload className="h-3.5 w-3.5" /> Upload lecture
          </Button>
        )}
      </DialogTrigger>

      <DialogContent className="sm:max-w-lg">
        <form onSubmit={submit}>
          <DialogHeader>
            <DialogTitle>Upload a recording</DialogTitle>
            <DialogDescription>
              {hasSyllabus
                ? 'Decoded in two passes and conditioned on the syllabus units retrieval selects for each span.'
                : 'No syllabus on this subject yet — this will run a single unconditioned pass. Upload the syllabus first, or reprocess afterwards.'}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-5">
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setDragging(true)
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragging(false)
                accept(e.dataTransfer.files[0])
              }}
              onClick={() => inputRef.current?.click()}
              className={cn(
                'cursor-pointer rounded-xl border-2 border-dashed p-6 text-center transition-colors',
                dragging
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:border-primary/50 hover:bg-muted/30'
              )}
            >
              <input
                ref={inputRef}
                type="file"
                className="hidden"
                accept={AUDIO_SUFFIXES.join(',')}
                onChange={(e) => accept(e.target.files?.[0])}
              />

              {file ? (
                <div className="flex items-center gap-3 text-left">
                  <FileAudio className="h-8 w-8 shrink-0 text-primary" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-foreground">{file.name}</p>
                    <p className="text-xs text-muted-foreground">{bytes(file.size)}</p>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={(e) => {
                      e.stopPropagation()
                      setFile(null)
                    }}
                  >
                    <X className="h-3.5 w-3.5" />
                  </Button>
                </div>
              ) : (
                <>
                  <Upload className="mx-auto mb-2 h-7 w-7 text-muted-foreground" />
                  <p className="text-sm font-medium text-foreground">
                    Drop a recording, or click to browse
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    m4a, mp3, wav, aac, opus and more — phone recordings are fine
                  </p>
                </>
              )}
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="lecture-title">Title</Label>
              <Input
                id="lecture-title"
                placeholder="Process scheduling"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="lecture-when">
                Recorded at <span className="text-muted-foreground">(optional)</span>
              </Label>
              <Input
                id="lecture-when"
                type="datetime-local"
                value={recordedAt}
                onChange={(e) => setRecordedAt(e.target.value)}
              />
            </div>
          </div>

          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => setOpen(false)} disabled={busy}>
              Cancel
            </Button>
            <Button type="submit" disabled={busy || !file}>
              {busy && <Spinner className="mr-2 h-3.5 w-3.5" />}
              {busy ? 'Uploading…' : 'Upload & transcribe'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
