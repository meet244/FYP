'use client'

import { useEffect, useRef } from 'react'
import { GraduationCap, Play } from 'lucide-react'

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
      <div className="border border-dashed border-border p-12 text-center">
        <p className="kicker mb-3">Notes</p>
        <h4 className="mb-1.5 font-display text-2xl text-foreground">None for this lecture</h4>
        <p className="mx-auto max-w-sm text-sm leading-relaxed text-muted-foreground">
          Note synthesis needs an LLM key. Transcription does not — the transcript above is the
          expensive artefact and it is already safe on disk. Set{' '}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">GEMINI_API_KEY</code>{' '}
          and use “Regenerate notes” to retry without transcribing again.
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
              'scroll-mt-4 border bg-card p-5 transition-colors',
              focused ? 'border-foreground/30' : 'border-border'
            )}
          >
            <div className="mb-3 flex flex-wrap items-center gap-2">
              {note.start_s != null && (
                <button
                  type="button"
                  onClick={() => onSeek(note.start_s!)}
                  className="flex shrink-0 items-center gap-1 border border-border px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground transition-colors hover:border-foreground/30 hover:text-foreground"
                  title="Play from here"
                >
                  <Play className="h-2.5 w-2.5" />
                  {timestamp(note.start_s)}
                </button>
              )}

              <h3 className="min-w-0 flex-1 font-display text-lg leading-snug text-foreground">{note.topic}</h3>

              {note.unit_id && unitKey.has(note.unit_id) && (
                <Badge variant="secondary" className="shrink-0 font-mono text-[10px]">
                  {unitKey.get(note.unit_id)}
                </Badge>
              )}
            </div>

            <Markdown>{note.markdown}</Markdown>

            {note.terms.length > 0 && (
              <div className="mt-4 border-t border-border/50 pt-3">
                <p className="kicker mb-2">Key terms</p>
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
                <p className="kicker mb-2 flex items-center gap-1.5">
                  <GraduationCap className="h-3 w-3" /> Learning outcomes
                </p>
                <ul className="space-y-1.5">
                  {note.outcomes.map((o, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs leading-relaxed">
                      <span className="mt-1.5 h-px w-2 shrink-0 bg-foreground/40" />
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
