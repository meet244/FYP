'use client'

import { useRef, useState } from 'react'
import { FileText, Info, Pencil, Upload, X } from 'lucide-react'
import { toast } from 'sonner'
import { mutate } from 'swr'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import { Spinner } from '@/components/ui/spinner'
import { Textarea } from '@/components/ui/textarea'
import { updateUnit, uploadSyllabus } from '@/lib/api/client'
import { keys, useSyllabus } from '@/lib/api/hooks'
import { cn } from '@/lib/utils'
import type { Job, Unit } from '@/lib/api/types'

function UnitEditor({
  subjectId,
  unit,
  onDone,
}: {
  subjectId: string
  unit: Unit
  onDone: () => void
}) {
  const [title, setTitle] = useState(unit.title)
  const [prose, setProse] = useState(unit.prose)
  const [keywords, setKeywords] = useState(unit.keywords.join(', '))
  const [busy, setBusy] = useState(false)

  const save = async () => {
    setBusy(true)
    try {
      await updateUnit(subjectId, unit.id, {
        title: title.trim(),
        prose: prose.trim(),
        keywords: keywords
          .split(',')
          .map((k) => k.trim())
          .filter(Boolean),
      })
      await mutate(keys.syllabus(subjectId))
      toast.success('Unit updated', {
        description: 'Reprocess a lecture to decode it against the new prose.',
      })
      onDone()
    } catch (err) {
      toast.error('Could not save unit', {
        description: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        <Label htmlFor={`t-${unit.id}`} className="text-xs">
          Title
        </Label>
        <Input
          id={`t-${unit.id}`}
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          className="h-8 text-sm"
        />
      </div>

      <div className="space-y-1.5">
        <Label htmlFor={`p-${unit.id}`} className="text-xs">
          Prose
        </Label>
        <Textarea
          id={`p-${unit.id}`}
          rows={7}
          value={prose}
          onChange={(e) => setProse(e.target.value)}
          className="font-mono text-xs leading-relaxed"
        />
        <p className="flex gap-1.5 text-[11px] leading-relaxed text-muted-foreground">
          <Info className="mt-0.5 h-3 w-3 shrink-0" />
          <span>
            This is the text the decoder conditions on. Keep it fluent narration in the register the
            lecturer actually uses — a comma-separated term list is the published failure mode
            (+19.39 WER).
          </span>
        </p>
      </div>

      <div className="space-y-1.5">
        <Label htmlFor={`k-${unit.id}`} className="text-xs">
          Keywords <span className="text-muted-foreground">(comma separated)</span>
        </Label>
        <Input
          id={`k-${unit.id}`}
          value={keywords}
          onChange={(e) => setKeywords(e.target.value)}
          className="h-8 text-sm"
        />
      </div>

      <div className="flex gap-2 pt-1">
        <Button size="sm" onClick={save} disabled={busy}>
          {busy && <Spinner className="mr-2 h-3 w-3" />}
          Save unit
        </Button>
        <Button size="sm" variant="ghost" onClick={onDone} disabled={busy}>
          Cancel
        </Button>
      </div>
    </div>
  )
}

export function SyllabusPanel({
  subjectId,
  onQueued,
}: {
  subjectId: string
  onQueued: (job: Job) => void
}) {
  const { data: syllabus, error, isLoading } = useSyllabus(subjectId)
  const [editing, setEditing] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const upload = async (f: File | undefined) => {
    if (!f) return
    if (!f.name.toLowerCase().endsWith('.pdf')) {
      toast.error('Syllabus must be a PDF')
      return
    }
    setUploading(true)
    try {
      const job = await uploadSyllabus(subjectId, f)
      onQueued(job)
      toast.success('Syllabus queued', {
        description: 'Rewriting it into code-mixed units takes a moment.',
      })
    } catch (err) {
      toast.error('Upload failed', {
        description: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setUploading(false)
    }
  }

  const dropzone = (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        upload(e.dataTransfer.files[0])
      }}
      onClick={() => !uploading && inputRef.current?.click()}
      className={cn(
        'cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors',
        dragging ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50 hover:bg-muted/30',
        uploading && 'pointer-events-none opacity-60'
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={(e) => upload(e.target.files?.[0])}
      />
      {uploading ? (
        <Spinner className="mx-auto h-7 w-7" />
      ) : (
        <Upload className="mx-auto mb-2 h-7 w-7 text-muted-foreground" />
      )}
      <p className="mt-2 text-sm font-medium text-foreground">
        {syllabus ? 'Replace the syllabus PDF' : 'Drop the syllabus PDF, or click to browse'}
      </p>
      <p className="mx-auto mt-1 max-w-sm text-xs leading-relaxed text-muted-foreground">
        Parsed by an LLM into code-mixed narration, not extracted as raw text — that rewrite is what
        makes conditioning work. Scanned PDFs need OCR first.
      </p>
    </div>
  )

  if (isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-32 rounded-xl" />
        <Skeleton className="h-12 rounded-xl" />
        <Skeleton className="h-12 rounded-xl" />
      </div>
    )
  }

  // 404 is the ordinary "nothing uploaded yet" case, not an error to surface.
  if (error || !syllabus) return dropzone

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 rounded-xl border border-border/60 bg-card p-3">
        <FileText className="h-4 w-4 shrink-0 text-primary" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
          {syllabus.source_filename ?? 'syllabus.pdf'}
        </span>
        <Badge variant="secondary" className="text-[10px]">
          {syllabus.units.length} units
        </Badge>
        <Badge
          variant="outline"
          className={cn(
            'text-[10px]',
            syllabus.provenance === 'real'
              ? 'border-emerald-500/30 text-emerald-600 dark:text-emerald-400'
              : 'border-amber-500/30 text-amber-600 dark:text-amber-400'
          )}
        >
          {syllabus.provenance}
        </Badge>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 gap-1.5 text-xs"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
        >
          {uploading ? <Spinner className="h-3 w-3" /> : <Upload className="h-3 w-3" />}
          Replace
        </Button>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={(e) => upload(e.target.files?.[0])}
        />
      </div>

      <Accordion type="multiple" className="space-y-2">
        {syllabus.units.map((u) => (
          <AccordionItem
            key={u.id}
            value={u.id}
            className="rounded-xl border border-border/60 bg-card px-3.5"
          >
            <AccordionTrigger className="py-3 hover:no-underline">
              <div className="flex min-w-0 flex-1 items-center gap-2.5 text-left">
                <span className="shrink-0 rounded-md bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                  {u.unit_key}
                </span>
                <span className="truncate text-sm font-medium text-foreground">{u.title}</span>
              </div>
            </AccordionTrigger>

            <AccordionContent className="pb-4">
              {editing === u.id ? (
                <UnitEditor subjectId={subjectId} unit={u} onDone={() => setEditing(null)} />
              ) : (
                <div className="space-y-3">
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-muted-foreground">
                    {u.prose}
                  </p>

                  {u.keywords.length > 0 && (
                    <div className="flex flex-wrap gap-1.5">
                      {u.keywords.map((k) => (
                        <Badge key={k} variant="secondary" className="text-[10px] font-normal">
                          {k}
                        </Badge>
                      ))}
                    </div>
                  )}

                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 gap-1.5 text-xs"
                    onClick={() => setEditing(u.id)}
                  >
                    <Pencil className="h-3 w-3" /> Edit prose
                  </Button>
                </div>
              )}
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
    </div>
  )
}
