'use client'

import { useEffect, useRef } from 'react'
import { GraduationCap, Play, Sparkles } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Markdown } from '@/components/markdown'
import { timestamp } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { Note, Unit } from '@/lib/api/types'

export function NotesView({
  notes,
  units,
  focusNoteId,
  onSeek,
}: {
  notes: Note[]
  units: Unit[]
  focusNoteId?: string | null
  onSeek: (seconds: number) => void
}) {
  const focusRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (focusNoteId) focusRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [focusNoteId])

  if (notes.length === 0) {
    return (
      <div className="rounded-xl border border-dashed border-border/60 p-12 text-center">
        <Sparkles className="mx-auto mb-3 h-7 w-7 text-muted-foreground" />
        <h4 className="mb-1.5 font-semibold text-foreground">No notes for this lecture</h4>
        <p className="mx-auto max-w-sm text-sm leading-relaxed text-muted-foreground">
          Note synthesis needs an LLM key. Transcription does not — the transcript above is the
          expensive artefact and it is already safe on disk. Set{' '}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">ANTHROPIC_API_KEY</code>{' '}
          and reprocess.
        </p>
      </div>
    )
  }

  const unitKey = new Map(units.map((u) => [u.id, u.unit_key]))

  return (
    <div className="space-y-3">
      {notes.map((note) => {
        const focused = note.id === focusNoteId
        return (
          <div
            key={note.id}
            ref={focused ? focusRef : undefined}
            className={cn(
              'scroll-mt-4 rounded-xl border bg-card p-4 transition-colors',
              focused ? 'border-primary/50 ring-1 ring-primary/20' : 'border-border/60'
            )}
          >
            <div className="mb-3 flex flex-wrap items-center gap-2">
              {note.start_s != null && (
                <button
                  type="button"
                  onClick={() => onSeek(note.start_s!)}
                  className="flex shrink-0 items-center gap-1 rounded-md bg-primary/10 px-1.5 py-0.5 font-mono text-[11px] text-primary transition-colors hover:bg-primary/20"
                  title="Play from here"
                >
                  <Play className="h-2.5 w-2.5" />
                  {timestamp(note.start_s)}
                </button>
              )}

              <h3 className="min-w-0 flex-1 text-sm font-bold text-foreground">{note.topic}</h3>

              {note.unit_id && unitKey.has(note.unit_id) && (
                <Badge variant="secondary" className="shrink-0 font-mono text-[10px]">
                  {unitKey.get(note.unit_id)}
                </Badge>
              )}
            </div>

            <Markdown>{note.markdown}</Markdown>

            {note.terms.length > 0 && (
              <div className="mt-4 border-t border-border/50 pt-3">
                <p className="mb-2 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                  Key terms
                </p>
                <dl className="space-y-1.5">
                  {note.terms.map((t) => (
                    <div key={t.term} className="text-xs leading-relaxed">
                      <dt className="inline font-semibold text-foreground">{t.term}</dt>
                      <dd className="inline text-muted-foreground"> — {t.definition}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            )}

            {note.outcomes.length > 0 && (
              <div className="mt-4 border-t border-border/50 pt-3">
                <p className="mb-2 flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
                  <GraduationCap className="h-3 w-3" /> Learning outcomes
                </p>
                <ul className="space-y-1.5">
                  {note.outcomes.map((o, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs leading-relaxed">
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-primary/60" />
                      <span className="text-muted-foreground">{o.text}</span>
                      {o.bloom_level && (
                        <Badge
                          variant="outline"
                          className="ml-auto h-4 shrink-0 px-1.5 text-[9px] font-normal"
                        >
                          {o.bloom_level}
                        </Badge>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
