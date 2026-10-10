'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  BarChart3,
  BookOpen,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleCheck,
  Clock,
  Copy,
  Cpu,
  Expand,
  Layers,
  Lightbulb,
  Minus,
  Network,
  Plus,
  Printer,
  RotateCcw,
  Settings,
  Shuffle,
  SquareFunction,
  Target,
  TriangleAlert,
  X,
} from 'lucide-react'
import { toast } from 'sonner'

import { Markdown } from '@/components/markdown'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import type {
  Citation,
  FlashcardsContent,
  InfographicContent,
  InfographicIcon,
  MindNode,
  MindmapContent,
  QuizContent,
  ReportContent,
  SlidesContent,
} from '@/lib/api/types'

export interface ViewProps<T> {
  content: T
  citations: Citation[]
  onOpenSource: (c: Citation) => void
}

/** Small `[n]` chips that open the cited lecture moment, page or unit. */
export function SourceRefs({ refs, citations, onOpenSource, className }: {
  refs: number[]
  citations: Citation[]
  onOpenSource: (c: Citation) => void
  className?: string
}) {
  const found = refs.map((n) => citations.find((c) => c.n === n)).filter((c): c is Citation => !!c)
  if (!found.length) return null
  return (
    <span className={cn('inline-flex flex-wrap gap-1', className)}>
      {found.map((c) => (
        <button
          key={c.n}
          type="button"
          onClick={(e) => { e.stopPropagation(); onOpenSource(c) }}
          title={c.material_title ?? c.lecture_title ?? c.unit_title ?? 'Source'}
          className="inline-flex h-5 min-w-5 items-center justify-center rounded-full border border-border bg-background px-1.5 font-mono text-[10px] text-muted-foreground hover:border-foreground/40 hover:text-foreground"
        >
          {c.n}
        </button>
      ))}
    </span>
  )
}

function useArrowKeys(onPrev: () => void, onNext: () => void, extra?: (e: KeyboardEvent) => void) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (e.key === 'ArrowLeft') onPrev()
      else if (e.key === 'ArrowRight') onNext()
      else extra?.(e)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onPrev, onNext, extra])
}

// --- quiz -----------------------------------------------------------------------

export function QuizView({ content, citations, onOpenSource }: ViewProps<QuizContent>) {
  const qs = content.questions
  const [index, setIndex] = useState(0)
  const [answers, setAnswers] = useState<(number | null)[]>(() => qs.map(() => null))
  const [hint, setHint] = useState(false)
  const [finished, setFinished] = useState(false)

  const q = qs[index]
  const chosen = answers[index]
  const answered = chosen != null
  const score = answers.filter((a, i) => a === qs[i].answer_index).length
  const done = answers.filter((a) => a != null).length

  const go = useCallback((i: number) => {
    setIndex(Math.max(0, Math.min(qs.length - 1, i)))
    setHint(false)
  }, [qs.length])
  const choose = useCallback((i: number) => {
    if (answers[index] != null || i >= q.options.length) return
    setAnswers((prev) => prev.map((a, j) => (j === index ? i : a)))
  }, [answers, index, q.options.length])

  const onKey = useCallback((e: KeyboardEvent) => {
    const n = Number(e.key)
    if (n >= 1 && n <= 9) choose(n - 1)
  }, [choose])
  useArrowKeys(useCallback(() => go(index - 1), [go, index]), useCallback(() => go(index + 1), [go, index]), onKey)

  if (finished) {
    const pct = Math.round((score / qs.length) * 100)
    return (
      <div className="mx-auto flex max-w-xl flex-col items-center px-6 py-12 text-center">
        <p className="font-display text-6xl text-foreground">{score}<span className="text-muted-foreground">/{qs.length}</span></p>
        <p className="mt-3 text-sm text-muted-foreground">
          {pct >= 80 ? 'Strong result. Review any misses, then move on.' : pct >= 50 ? 'Halfway there. Open the sources on the questions you missed.' : 'Worth another pass. Review the explanations, then retake.'}
        </p>
        <div className="mt-8 w-full space-y-1 text-left">
          {qs.map((item, i) => {
            const right = answers[i] === item.answer_index
            return (
              <button key={i} type="button" onClick={() => { setFinished(false); go(i) }}
                className="flex w-full items-start gap-2 border-b border-border py-2 text-left text-sm hover:bg-muted/50">
                {right ? <Check className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" /> : <X className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />}
                <span className="min-w-0 flex-1">{item.question}</span>
              </button>
            )
          })}
        </div>
        <Button className="mt-8 gap-1.5" onClick={() => { setAnswers(qs.map(() => null)); setFinished(false); go(0) }}>
          <RotateCcw className="h-3.5 w-3.5" /> Retake quiz
        </Button>
      </div>
    )
  }

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col px-5 py-6 sm:px-8 sm:py-8">
      <div className="mb-6 flex items-center gap-3">
        <div className="h-1 flex-1 bg-muted">
          <div className="h-full bg-primary transition-[width]" style={{ width: `${(done / qs.length) * 100}%` }} />
        </div>
        <span className="text-xs tabular-nums text-muted-foreground">{index + 1} of {qs.length}</span>
      </div>

      <h3 className="font-display text-[1.6rem] leading-snug text-foreground">{q.question}</h3>
      {q.difficulty && <p className="mt-1 text-xs capitalize text-muted-foreground">{q.difficulty}</p>}

      <ol className="mt-6 space-y-2">
        {q.options.map((opt, i) => {
          const isAnswer = i === q.answer_index
          const isChosen = i === chosen
          return (
            <li key={i}>
              <button
                type="button"
                disabled={answered}
                onClick={() => choose(i)}
                className={cn(
                  'flex w-full items-start gap-3 border px-4 py-3 text-left text-[15px] transition-colors',
                  !answered && 'border-border bg-background hover:border-foreground/40',
                  answered && isAnswer && 'border-emerald-600/60 bg-emerald-600/10',
                  answered && isChosen && !isAnswer && 'border-destructive/60 bg-destructive/10',
                  answered && !isAnswer && !isChosen && 'border-border opacity-60'
                )}
              >
                <span className="mt-px font-mono text-xs text-muted-foreground">{String.fromCharCode(65 + i)}</span>
                <span className="min-w-0 flex-1">{opt}</span>
                {answered && isAnswer && <Check className="h-4 w-4 shrink-0 text-emerald-600" />}
                {answered && isChosen && !isAnswer && <X className="h-4 w-4 shrink-0 text-destructive" />}
              </button>
            </li>
          )
        })}
      </ol>

      <div className="mt-4 min-h-[4.5rem] text-sm leading-relaxed">
        {answered ? (
          <div>
            <p className="text-foreground">
              <span className={cn('font-medium', chosen === q.answer_index ? 'text-emerald-700 dark:text-emerald-400' : 'text-destructive')}>
                {chosen === q.answer_index ? 'Correct. ' : 'Not quite. '}
              </span>
              {q.explanation}
            </p>
            <SourceRefs refs={q.sources} citations={citations} onOpenSource={onOpenSource} className="mt-2" />
          </div>
        ) : q.hint ? (
          hint ? <p className="text-muted-foreground">{q.hint}</p> : (
            <button type="button" onClick={() => setHint(true)} className="text-muted-foreground underline underline-offset-2 hover:text-foreground">Show hint</button>
          )
        ) : null}
      </div>

      <div className="mt-4 flex items-center justify-between">
        <Button variant="ghost" size="sm" disabled={index === 0} onClick={() => go(index - 1)} className="gap-1">
          <ChevronLeft className="h-4 w-4" /> Previous
        </Button>
        {index < qs.length - 1 ? (
          <Button size="sm" onClick={() => go(index + 1)} className="gap-1">
            Next <ChevronRight className="h-4 w-4" />
          </Button>
        ) : (
          <Button size="sm" onClick={() => setFinished(true)}>See score</Button>
        )}
      </div>
      <p className="mt-6 text-center text-[11px] text-muted-foreground">Keys 1–4 answer, arrow keys move between questions.</p>
    </div>
  )
}

// --- flashcards -------------------------------------------------------------------

export function FlashcardsView({ content, citations, onOpenSource }: ViewProps<FlashcardsContent>) {
  const [order, setOrder] = useState(() => content.cards.map((_, i) => i))
  const [pos, setPos] = useState(0)
  const [flipped, setFlipped] = useState(false)
  const card = content.cards[order[pos]]

  const go = useCallback((next: number) => {
    setFlipped(false)
    setPos((Math.max(0, Math.min(order.length - 1, next))))
  }, [order.length])
  const onKey = useCallback((e: KeyboardEvent) => {
    if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); setFlipped((f) => !f) }
  }, [])
  useArrowKeys(useCallback(() => go(pos - 1), [go, pos]), useCallback(() => go(pos + 1), [go, pos]), onKey)

  const shuffle = () => {
    const next = [...order]
    for (let i = next.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[next[i], next[j]] = [next[j], next[i]]
    }
    setOrder(next)
    setPos(0)
    setFlipped(false)
  }

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col items-center px-5 py-8">
      <button
        type="button"
        onClick={() => setFlipped((f) => !f)}
        aria-label={flipped ? 'Show front' : 'Show answer'}
        className="group w-full [perspective:1600px]"
      >
        <div
          className={cn(
            'relative aspect-[3/2] w-full transition-transform duration-500 [transform-style:preserve-3d] motion-reduce:transition-none',
            flipped && '[transform:rotateY(180deg)]'
          )}
        >
          <div className="absolute inset-0 flex flex-col items-center justify-center border border-border bg-background p-8 [backface-visibility:hidden]">
            <p className="text-center font-display text-[clamp(1.4rem,3.4vw,2.2rem)] leading-tight text-foreground">{card.front}</p>
            <p className="absolute bottom-4 text-[11px] text-muted-foreground">Click or press space to flip</p>
          </div>
          <div className="absolute inset-0 flex flex-col items-center justify-center border border-primary/40 bg-secondary p-8 [backface-visibility:hidden] [transform:rotateY(180deg)]">
            <p className="text-center text-[17px] leading-relaxed text-foreground">{card.back}</p>
            <SourceRefs refs={card.sources} citations={citations} onOpenSource={onOpenSource} className="absolute bottom-4" />
          </div>
        </div>
      </button>

      <div className="mt-6 flex w-full items-center justify-between">
        <Button variant="ghost" size="icon" onClick={() => go(pos - 1)} disabled={pos === 0} aria-label="Previous card">
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <span className="text-sm tabular-nums text-muted-foreground">{pos + 1} / {order.length}</span>
        <Button variant="ghost" size="icon" onClick={() => go(pos + 1)} disabled={pos === order.length - 1} aria-label="Next card">
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
      <Button variant="outline" size="sm" className="mt-3 gap-1.5" onClick={shuffle}>
        <Shuffle className="h-3.5 w-3.5" /> Shuffle
      </Button>
    </div>
  )
}

// --- mind map -----------------------------------------------------------------------

function collectIds(node: MindNode, depth: number, maxDepth: number, into: Set<string>) {
  if (depth < maxDepth && node.children.length) into.add(node.id)
  node.children.forEach((c) => collectIds(c, depth + 1, maxDepth, into))
  return into
}

function MindBranch({ node, depth, open, toggle, selected, select }: {
  node: MindNode
  depth: number
  open: Set<string>
  toggle: (id: string) => void
  selected: string | null
  select: (n: MindNode) => void
}) {
  const expanded = open.has(node.id)
  const hasKids = node.children.length > 0
  return (
    <div className="flex items-center">
      <div className="flex shrink-0 items-center">
        <button
          type="button"
          onClick={() => select(node)}
          className={cn(
            'max-w-[15rem] border px-3 py-1.5 text-left leading-snug transition-colors',
            depth === 0 && 'border-primary bg-primary font-display text-lg text-primary-foreground',
            depth === 1 && 'border-violet-500/50 bg-violet-500/10 text-[14px] font-medium text-foreground',
            depth >= 2 && 'border-border bg-background text-[13px] text-foreground',
            selected === node.id && depth > 0 && 'ring-2 ring-ring/50'
          )}
        >
          {node.label}
        </button>
        {hasKids && (
          <button
            type="button"
            onClick={() => toggle(node.id)}
            aria-label={expanded ? `Collapse ${node.label}` : `Expand ${node.label}`}
            aria-expanded={expanded}
            className="ml-1 flex h-5 w-5 items-center justify-center rounded-full border border-border bg-background text-muted-foreground hover:text-foreground"
          >
            {expanded ? <Minus className="h-3 w-3" /> : <Plus className="h-3 w-3" />}
          </button>
        )}
      </div>
      {hasKids && expanded && (
        <div className="ml-4 flex flex-col">
          {node.children.map((child) => (
            <div
              key={child.id}
              className={cn(
                'relative py-1 pl-6',
                // Elbow connector: horizontal stub plus the vertical spine, trimmed at the ends.
                'before:absolute before:left-0 before:top-1/2 before:w-6 before:border-t before:border-foreground/25',
                'after:absolute after:left-0 after:top-0 after:bottom-0 after:border-l after:border-foreground/25',
                'first:after:top-1/2 last:after:bottom-1/2'
              )}
            >
              <MindBranch node={child} depth={depth + 1} open={open} toggle={toggle} selected={selected} select={select} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export function MindmapView({ content, citations, onOpenSource }: ViewProps<MindmapContent>) {
  const [open, setOpen] = useState(() => collectIds(content.root, 0, 1, new Set()))
  const [selected, setSelected] = useState<MindNode | null>(null)
  const toggle = (id: string) =>
    setOpen((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 gap-1 border-b border-border px-4 py-2">
        <Button variant="ghost" size="sm" onClick={() => setOpen(collectIds(content.root, 0, 99, new Set()))}>Expand all</Button>
        <Button variant="ghost" size="sm" onClick={() => setOpen(new Set([content.root.id]))}>Collapse all</Button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-6">
        <div className="w-max">
          <MindBranch node={content.root} depth={0} open={open} toggle={toggle} selected={selected?.id ?? null} select={setSelected} />
        </div>
      </div>
      {selected && selected.id !== content.root.id && (
        <div className="shrink-0 border-t border-border bg-muted/40 px-5 py-3 text-sm">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <p className="font-medium text-foreground">{selected.label}</p>
              {selected.summary && <p className="mt-1 leading-relaxed text-muted-foreground">{selected.summary}</p>}
              <SourceRefs refs={selected.sources} citations={citations} onOpenSource={onOpenSource} className="mt-2" />
            </div>
            <Button variant="ghost" size="icon" className="h-7 w-7 shrink-0" onClick={() => setSelected(null)} aria-label="Close details">
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

// --- report -----------------------------------------------------------------------

export function ReportView({ content, citations, onOpenSource }: ViewProps<ReportContent>) {
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(`# ${content.title}\n\n${content.markdown}`)
      toast.success('Copied as Markdown')
    } catch {
      toast.error('Clipboard is not available in this browser')
    }
  }
  return (
    <div className="mx-auto w-full max-w-3xl px-5 py-6 sm:px-10 sm:py-8">
      <div className="mb-4 flex justify-end">
        <Button variant="outline" size="sm" className="gap-1.5" onClick={copy}>
          <Copy className="h-3.5 w-3.5" /> Copy Markdown
        </Button>
      </div>
      <Markdown
        onCitation={(n) => {
          const c = citations.find((x) => x.n === n)
          if (c) onOpenSource(c)
        }}
      >
        {content.markdown}
      </Markdown>
    </div>
  )
}

// --- slides -----------------------------------------------------------------------

function escapeHtml(s: string) {
  return s.replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]!)
}

function printDeck(content: SlidesContent) {
  const win = window.open('', '_blank')
  if (!win) {
    toast.error('Allow pop-ups to print or save the deck as PDF')
    return
  }
  const slide = (inner: string) => `<section>${inner}</section>`
  const html = [
    slide(`<h1>${escapeHtml(content.title)}</h1>${content.subtitle ? `<p class="sub">${escapeHtml(content.subtitle)}</p>` : ''}`),
    ...content.slides.map((s) =>
      slide(`<h2>${escapeHtml(s.title)}</h2><ul>${s.bullets.map((b) => `<li>${escapeHtml(b)}</li>`).join('')}</ul>`)
    ),
  ].join('')
  win.document.write(`<!doctype html><html><head><title>${escapeHtml(content.title)}</title><style>
    @page { size: 13.333in 7.5in; margin: 0 }
    body { margin: 0; font-family: Georgia, 'Times New Roman', serif; color: #2a241e }
    section { width: 13.333in; height: 7.5in; box-sizing: border-box; padding: 0.9in 1.1in; page-break-after: always; display: flex; flex-direction: column; justify-content: center; background: #f7f4ee }
    h1 { font-size: 54px; font-weight: 400; margin: 0 } .sub { font-size: 24px; color: #6b6158 }
    h2 { font-size: 40px; font-weight: 400; margin: 0 0 28px } li { font: 24px/1.5 Helvetica, Arial, sans-serif; margin-bottom: 10px }
  </style></head><body>${html}</body></html>`)
  win.document.close()
  win.focus()
  setTimeout(() => win.print(), 300)
}

export function SlidesView({ content, citations, onOpenSource }: ViewProps<SlidesContent>) {
  const total = content.slides.length + 1
  const [i, setI] = useState(0)
  const stage = useRef<HTMLDivElement>(null)
  const go = useCallback((n: number) => setI(Math.max(0, Math.min(total - 1, n))), [total])
  useArrowKeys(useCallback(() => go(i - 1), [go, i]), useCallback(() => go(i + 1), [go, i]))
  const slide = i > 0 ? content.slides[i - 1] : null

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-center justify-end gap-1 border-b border-border px-4 py-2">
        <Button variant="ghost" size="sm" className="gap-1.5" onClick={() => stage.current?.requestFullscreen?.()}>
          <Expand className="h-3.5 w-3.5" /> Present
        </Button>
        <Button variant="ghost" size="sm" className="gap-1.5" onClick={() => printDeck(content)}>
          <Printer className="h-3.5 w-3.5" /> Print or save PDF
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-5 sm:px-8">
        <div
          ref={stage}
          onClick={() => go(i + 1)}
          className="mx-auto flex aspect-video w-full max-w-4xl cursor-pointer flex-col justify-center overflow-hidden border border-border bg-background px-[7%] py-[5%] [container-type:inline-size]"
        >
          {slide ? (
            <>
              <h3 className="font-display text-[5cqw] leading-tight text-foreground">{slide.title}</h3>
              <span className="mt-[2cqw] block h-px w-[10cqw] bg-fuchsia-600/60" />
              <ul className="mt-[3cqw] space-y-[1.4cqw]">
                {slide.bullets.map((b, k) => (
                  <li key={k} className="flex gap-[1.5cqw] text-[2.3cqw] leading-snug text-foreground">
                    <span className="mt-[0.9cqw] h-[0.7cqw] w-[0.7cqw] shrink-0 bg-fuchsia-600/70" />
                    {b}
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <>
              <h3 className="font-display text-[6.5cqw] leading-[1.05] text-foreground">{content.title}</h3>
              {content.subtitle && <p className="mt-[2.5cqw] text-[2.4cqw] text-muted-foreground">{content.subtitle}</p>}
            </>
          )}
        </div>

        <div className="mx-auto mt-3 flex max-w-4xl items-center justify-between">
          <Button variant="ghost" size="icon" onClick={() => go(i - 1)} disabled={i === 0} aria-label="Previous slide">
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm tabular-nums text-muted-foreground">{i + 1} / {total}</span>
          <Button variant="ghost" size="icon" onClick={() => go(i + 1)} disabled={i === total - 1} aria-label="Next slide">
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        {slide?.speaker_notes && (
          <div className="mx-auto mt-3 max-w-4xl border-l-2 border-fuchsia-600/50 pl-4 text-sm leading-relaxed text-muted-foreground">
            <p className="mb-1 text-xs font-medium text-foreground">Speaker notes</p>
            {slide.speaker_notes}
            <SourceRefs refs={slide.sources} citations={citations} onOpenSource={onOpenSource} className="mt-2 flex" />
          </div>
        )}

        <div className="mx-auto mt-5 flex max-w-4xl gap-2 overflow-x-auto pb-2">
          {[content.title, ...content.slides.map((s) => s.title)].map((t, k) => (
            <button
              key={k}
              type="button"
              onClick={() => go(k)}
              className={cn(
                'flex aspect-video w-28 shrink-0 items-end border p-1.5 text-left text-[9px] leading-tight',
                k === i ? 'border-foreground' : 'border-border text-muted-foreground hover:border-foreground/40'
              )}
            >
              <span className="line-clamp-3">{t}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

// --- infographic -------------------------------------------------------------------

const ICONS: Record<InfographicIcon, React.ElementType> = {
  lightbulb: Lightbulb,
  gear: Settings,
  chart: BarChart3,
  book: BookOpen,
  target: Target,
  layers: Layers,
  network: Network,
  warning: TriangleAlert,
  check: CircleCheck,
  clock: Clock,
  cpu: Cpu,
  function: SquareFunction,
}

export function InfographicView({ content, citations, onOpenSource }: ViewProps<InfographicContent>) {
  const accents = useMemo(
    () => ['border-t-emerald-600', 'border-t-sky-600', 'border-t-amber-600', 'border-t-violet-600', 'border-t-rose-600', 'border-t-fuchsia-600'],
    []
  )
  return (
    <div className="mx-auto w-full max-w-4xl px-4 py-6 sm:px-8">
      <article className="border border-border bg-background">
        <header className="bg-primary px-6 py-8 text-primary-foreground sm:px-10">
          <h3 className="font-display text-[clamp(2rem,5vw,3.2rem)] leading-[1.05]">{content.title}</h3>
          {content.subtitle && <p className="mt-2 max-w-2xl text-[15px] opacity-80">{content.subtitle}</p>}
        </header>

        {content.stats.length > 0 && (
          <div className="grid grid-cols-2 border-b border-border sm:grid-cols-4">
            {content.stats.map((s, k) => (
              <div key={k} className="border-border px-5 py-5 [&:not(:last-child)]:border-r">
                <p className="font-display text-3xl text-foreground">{s.value}</p>
                <p className="mt-1 text-xs leading-snug text-muted-foreground">{s.label}</p>
              </div>
            ))}
          </div>
        )}

        <div className="grid gap-px bg-border sm:grid-cols-2">
          {content.sections.map((s, k) => {
            const Icon = ICONS[s.icon] ?? Lightbulb
            return (
              <section key={k} className={cn('border-t-4 bg-background p-5 sm:p-6', accents[k % accents.length])}>
                <div className="flex items-center gap-2.5">
                  <Icon className="h-5 w-5 shrink-0 text-foreground/80" />
                  <h4 className="text-[15px] font-semibold text-foreground">{s.heading}</h4>
                </div>
                <ul className="mt-3 space-y-1.5">
                  {s.points.map((p, j) => (
                    <li key={j} className="flex gap-2 text-sm leading-snug text-foreground/90">
                      <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-foreground/50" />
                      {p}
                    </li>
                  ))}
                </ul>
                <SourceRefs refs={s.sources} citations={citations} onOpenSource={onOpenSource} className="mt-3" />
              </section>
            )
          })}
        </div>

        {content.process && (
          <div className="border-t border-border px-5 py-6 sm:px-8">
            <h4 className="mb-4 text-[15px] font-semibold text-foreground">{content.process.title}</h4>
            <ol className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-stretch">
              {content.process.steps.map((step, k) => (
                <li key={k} className="flex items-center gap-2 sm:flex-1 sm:basis-36">
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-primary text-xs font-semibold text-primary-foreground">
                    {k + 1}
                  </span>
                  <span className="text-sm leading-snug text-foreground">{step}</span>
                  {k < content.process!.steps.length - 1 && <ChevronRight className="hidden h-4 w-4 shrink-0 text-muted-foreground sm:block" />}
                </li>
              ))}
            </ol>
          </div>
        )}

        {content.takeaway && (
          <footer className="border-t border-border bg-secondary px-6 py-5 sm:px-10">
            <p className="font-display text-xl leading-snug text-foreground">{content.takeaway}</p>
          </footer>
        )}
      </article>
    </div>
  )
}
