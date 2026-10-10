'use client'

import { useState } from 'react'
import { Loader2, RotateCw, SlidersHorizontal, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import { mutate } from 'swr'

import { KIND_META, REPORT_FORMATS, STUDIO_KINDS } from '@/components/studio/kinds'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Textarea } from '@/components/ui/textarea'
import { createStudioItem, deleteStudioItem, retryStudioItem } from '@/lib/api/client'
import { keys, useLectures, useStudioItems, useSyllabus } from '@/lib/api/hooks'
import { relativeTime } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { Difficulty, ReportFormat, StudioItem, StudioKind, StudioOptions } from '@/lib/api/types'

type ScopeValue = string // 'subject' | `unit:${id}` | `lecture:${id}`

function parseScope(value: ScopeValue) {
  if (value === 'subject') return { type: 'subject' as const, id: null }
  const [type, id] = value.split(':') as ['unit' | 'lecture', string]
  return { type, id }
}

function errorText(err: unknown) {
  return err instanceof Error ? err.message : String(err)
}

function ScopePicker({ subjectId, value, onChange }: {
  subjectId: string
  value: ScopeValue
  onChange: (v: ScopeValue) => void
}) {
  const { data: syllabus } = useSyllabus(subjectId)
  const { data: lectures } = useLectures(subjectId)
  const ordered = [...(lectures ?? [])].sort((a, b) => a.created_at.localeCompare(b.created_at))
  return (
    <label className="block text-xs text-muted-foreground">
      Generate from
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="mt-1 h-8 w-full border border-border bg-background px-2 text-xs text-foreground"
      >
        <option value="subject">Whole subject</option>
        {!!syllabus?.units.length && (
          <optgroup label="Syllabus units">
            {syllabus.units.map((u) => (
              <option key={u.id} value={`unit:${u.id}`}>
                Unit {u.order_index}: {u.title}
              </option>
            ))}
          </optgroup>
        )}
        {!!ordered.length && (
          <optgroup label="Recordings">
            {ordered.map((l) => (
              <option key={l.id} value={`lecture:${l.id}`}>
                {l.title}
              </option>
            ))}
          </optgroup>
        )}
      </select>
    </label>
  )
}

function CustomizeDialog({ kind, onClose, onGenerate }: {
  kind: StudioKind | null
  onClose: () => void
  onGenerate: (kind: StudioKind, options: StudioOptions) => void
}) {
  const meta = kind ? KIND_META[kind] : null
  const [count, setCount] = useState<number | null>(null)
  const [difficulty, setDifficulty] = useState<Difficulty>('mixed')
  const [format, setFormat] = useState<ReportFormat>('study_guide')
  const [focus, setFocus] = useState('')

  const submit = () => {
    if (!kind) return
    onGenerate(kind, {
      count: meta?.count ? (count ?? meta.count.default) : null,
      difficulty: kind === 'quiz' ? difficulty : null,
      format: kind === 'report' ? format : null,
      focus: focus.trim() || null,
    })
    setFocus('')
    setCount(null)
  }

  return (
    <Dialog open={!!kind} onOpenChange={(open) => { if (!open) onClose() }}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="font-display text-xl font-normal">Customize {meta?.label.toLowerCase()}</DialogTitle>
          <DialogDescription>{meta?.blurb}</DialogDescription>
        </DialogHeader>
        <div className="space-y-4 text-sm">
          {meta?.count && (
            <fieldset>
              <legend className="mb-1.5 text-xs text-muted-foreground">{meta.count.label}</legend>
              <div className="grid grid-cols-4 gap-1.5">
                {meta.count.choices.map((n) => (
                  <button
                    key={n}
                    type="button"
                    onClick={() => setCount(n)}
                    aria-pressed={(count ?? meta.count!.default) === n}
                    className={cn(
                      'h-8 border text-xs',
                      (count ?? meta.count!.default) === n
                        ? 'border-foreground bg-foreground text-background'
                        : 'border-border hover:border-foreground/40'
                    )}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </fieldset>
          )}
          {kind === 'quiz' && (
            <label className="block text-xs text-muted-foreground">
              Difficulty
              <select value={difficulty} onChange={(e) => setDifficulty(e.target.value as Difficulty)}
                className="mt-1 h-8 w-full border border-border bg-background px-2 text-xs text-foreground">
                <option value="mixed">Mixed</option>
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </label>
          )}
          {kind === 'report' && (
            <fieldset>
              <legend className="mb-1.5 text-xs text-muted-foreground">Format</legend>
              <div className="grid grid-cols-2 gap-1.5">
                {REPORT_FORMATS.map((f) => (
                  <button
                    key={f.value}
                    type="button"
                    onClick={() => setFormat(f.value)}
                    aria-pressed={format === f.value}
                    className={cn(
                      'h-8 border text-xs',
                      format === f.value ? 'border-foreground bg-foreground text-background' : 'border-border hover:border-foreground/40'
                    )}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </fieldset>
          )}
          <label className="block text-xs text-muted-foreground">
            What should it focus on? (optional)
            <Textarea
              value={focus}
              onChange={(e) => setFocus(e.target.value)}
              maxLength={500}
              rows={3}
              placeholder="e.g. Only the perceptron learning rule and why XOR fails"
              className="mt-1 text-sm"
            />
          </label>
        </div>
        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>Cancel</Button>
          <Button onClick={submit}>Generate</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function ItemRow({ item, onOpen }: { item: StudioItem; onOpen: (id: string) => void }) {
  const meta = KIND_META[item.kind]
  const Icon = meta.icon
  const busy = item.status === 'pending' || item.status === 'running'

  const remove = async () => {
    try {
      await deleteStudioItem(item.id)
      await mutate(keys.studio(item.subject_id))
    } catch (err) {
      toast.error('Could not delete', { description: errorText(err) })
    }
  }
  const retry = async () => {
    try {
      await retryStudioItem(item.id)
      await mutate(keys.studio(item.subject_id))
    } catch (err) {
      toast.error('Could not retry', { description: errorText(err) })
    }
  }

  return (
    <li className="group relative border-b border-border">
      <button
        type="button"
        disabled={busy}
        onClick={() => onOpen(item.id)}
        className="flex w-full items-start gap-2.5 px-3 py-2.5 pr-14 text-left hover:bg-background disabled:cursor-default disabled:hover:bg-transparent"
      >
        {busy ? (
          <Loader2 className={cn('mt-0.5 h-4 w-4 shrink-0 animate-spin', meta.tint)} />
        ) : (
          <Icon className={cn('mt-0.5 h-4 w-4 shrink-0', meta.tint)} />
        )}
        <span className="min-w-0">
          <span className="block truncate text-[12.5px] text-foreground">{item.title}</span>
          <span className={cn('block truncate text-[11px]', item.status === 'failed' ? 'text-destructive' : 'text-muted-foreground')}>
            {busy
              ? `Generating from ${item.scope.label ?? 'subject'}…`
              : item.status === 'failed'
                ? item.error ?? 'Generation failed'
                : `${meta.label} · ${relativeTime(item.created_at)}`}
          </span>
        </span>
      </button>
      <div className="absolute right-1 top-2 flex opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100">
        {item.status === 'failed' && (
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={retry} aria-label={`Retry ${item.title}`}>
            <RotateCw className="h-3 w-3" />
          </Button>
        )}
        {!busy && (
          <Button variant="ghost" size="icon" className="h-6 w-6" onClick={remove} aria-label={`Delete ${item.title}`}>
            <Trash2 className="h-3 w-3" />
          </Button>
        )}
      </div>
    </li>
  )
}

export function StudioPanel({ subjectId, onOpen }: { subjectId: string; onOpen: (id: string) => void }) {
  const { data: items, isLoading } = useStudioItems(subjectId)
  const [scope, setScope] = useState<ScopeValue>('subject')
  const [customizing, setCustomizing] = useState<StudioKind | null>(null)

  const generate = async (kind: StudioKind, options: StudioOptions = {}) => {
    setCustomizing(null)
    try {
      await createStudioItem(subjectId, { kind, scope: parseScope(scope), options })
      await mutate(keys.studio(subjectId))
    } catch (err) {
      toast.error(`Could not start ${KIND_META[kind].label.toLowerCase()}`, { description: errorText(err) })
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 space-y-3 border-b border-border p-3">
        <h2 className="font-display text-xl leading-none text-foreground">Studio</h2>
        <ScopePicker subjectId={subjectId} value={scope} onChange={setScope} />
        <div className="grid grid-cols-2 gap-1.5">
          {STUDIO_KINDS.map((kind) => {
            const meta = KIND_META[kind]
            const Icon = meta.icon
            return (
              <div key={kind} className="group relative">
                <button
                  type="button"
                  onClick={() => generate(kind)}
                  title={meta.blurb}
                  className="flex h-11 w-full items-center gap-2 border border-border bg-background pl-2.5 pr-7 text-left text-[12.5px] text-foreground transition-colors hover:border-foreground/30"
                >
                  <Icon className={cn('h-4 w-4 shrink-0', meta.tint)} />
                  <span className="truncate">{meta.label}</span>
                </button>
                <button
                  type="button"
                  onClick={() => setCustomizing(kind)}
                  aria-label={`Customize ${meta.label}`}
                  className="absolute right-1 top-1/2 flex h-6 w-6 -translate-y-1/2 items-center justify-center text-muted-foreground opacity-60 hover:text-foreground hover:opacity-100 focus-visible:opacity-100"
                >
                  <SlidersHorizontal className="h-3 w-3" />
                </button>
              </div>
            )
          })}
        </div>
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Tap to generate, or use the sliders to customize. You can also ask in chat, e.g. “make a quiz on unit 2”.
        </p>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {isLoading ? (
          <p className="px-3 py-6 text-center text-[11px] text-muted-foreground">Loading…</p>
        ) : !items?.length ? (
          <p className="px-4 py-8 text-center text-[11px] leading-relaxed text-muted-foreground">
            Generated quizzes, flashcards, mind maps and more are saved here.
          </p>
        ) : (
          <ul>{items.map((item) => <ItemRow key={item.id} item={item} onOpen={onOpen} />)}</ul>
        )}
      </div>

      <CustomizeDialog kind={customizing} onClose={() => setCustomizing(null)} onGenerate={generate} />
    </div>
  )
}
