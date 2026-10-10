'use client'

import { useEffect, useRef, useState } from 'react'
import {
  ArrowUp,
  BookOpen,
  FileText,
  GitCompare,
  ListTree,
  MessageCircle,
  PieChart,
  Search,
  Sparkles,
  WandSparkles,
} from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { toast } from 'sonner'
import { mutate } from 'swr'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Markdown } from '@/components/markdown'
import { SgcdMachine } from '@/components/processing-stage'
import { StudioChatCard } from '@/components/studio/studio-chat-card'
import type { SourceSelection } from '@/components/source-viewer'
import { ask } from '@/lib/api/client'
import { keys, useSessionMessages } from '@/lib/api/hooks'
import { cn } from '@/lib/utils'
import type { ChatMessage, Citation, QueryType } from '@/lib/api/types'

const QUERY_TYPE: Record<QueryType, { label: string; icon: React.ElementType }> = {
  lookup: { label: 'Lookup', icon: Search },
  explain: { label: 'Explain', icon: Sparkles },
  summary: { label: 'Summary', icon: FileText },
  compare: { label: 'Compare', icon: GitCompare },
  quiz: { label: 'Quiz', icon: BookOpen },
  outline: { label: 'Outline', icon: ListTree },
  coverage: { label: 'Coverage', icon: PieChart },
  smalltalk: { label: 'Chat', icon: MessageCircle },
  studio: { label: 'Studio', icon: WandSparkles },
}

const SUGGESTIONS = [
  'What does the syllabus cover?',
  'Explain the last lecture from the recording',
  'What is on the slides or PDFs?',
  'Give me five practice questions',
  'Make a quiz on this subject',
]

function friendlyChatError(err: unknown): string {
  const raw = err instanceof Error ? err.message : String(err)
  if (/quota|RESOURCE_EXHAUSTED|\b429\b/i.test(raw)) {
    const cut = raw.split('{')[0].trim()
    return cut || 'Gemini quota is exhausted. Wait a minute, then try again.'
  }
  return raw.split('{')[0].trim() || 'Question failed'
}

function CitationList({
  citations,
  onOpen,
}: {
  citations: Citation[]
  onOpen: (c: Citation) => void
}) {
  if (citations.length === 0) return null
  return (
    <div className="mt-4 border-t border-border pt-3">
      <p className="kicker mb-2">Sources</p>
      <div className="flex flex-wrap gap-1.5">
        {citations.map((c) => (
          <button
            key={c.n}
            type="button"
            onClick={() => onOpen(c)}
            className="flex items-center gap-1.5 rounded-full border border-border bg-background px-2.5 py-1 text-[11px] transition-colors hover:border-foreground/30"
          >
            <span className="flex h-4 w-4 items-center justify-center rounded-full border border-border font-mono text-[10px] text-foreground">
              {c.n}
            </span>
            <span className="max-w-[10rem] truncate text-foreground">
              {c.kind === 'unit'
                ? (c.unit_title ?? 'Syllabus')
                : c.material_title ?? c.lecture_title ?? 'Source'}
            </span>
            {c.timestamp && (
              <span className="font-mono text-muted-foreground">{c.timestamp}</span>
            )}
            {(c.kind === 'note' ||
              c.kind === 'unit' ||
              c.kind === 'pdf' ||
              c.kind === 'image' ||
              c.kind === 'doc' ||
              c.kind === 'span' ||
              c.kind === 'transcript') && (
              <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                {c.kind === 'span' || c.kind === 'transcript'
                  ? 'voice'
                  : c.kind === 'unit'
                    ? 'syllabus'
                    : c.kind === 'note'
                      ? 'notes'
                      : c.kind}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}

function Composer({
  question,
  setQuestion,
  pending,
  onSend,
  autoFocus,
}: {
  question: string
  setQuestion: (v: string) => void
  pending: boolean
  onSend: () => void
  autoFocus?: boolean
}) {
  return (
    <div className="relative mx-auto w-full min-w-0 max-w-2xl">
      <div className="overflow-hidden rounded-3xl border border-border bg-background shadow-[0_1px_6px_oklch(0.22_0.02_55/0.06)]">
        <Textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              onSend()
            }
          }}
          placeholder="Ask anything about this subject…"
          rows={1}
          autoFocus={autoFocus}
          className="field-sizing-fixed max-h-40 min-h-[52px] w-full min-w-0 resize-none rounded-3xl border-0 bg-transparent py-3.5 pl-4 pr-14 text-[15px] shadow-none focus-visible:ring-0"
        />
        <Button
          size="icon"
          className="absolute bottom-2 right-2 h-8 w-8 rounded-full"
          onClick={onSend}
          disabled={pending || !question.trim()}
          aria-label="Send"
        >
          <ArrowUp className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  )
}

export function ChatPanel({
  subjectId,
  subjectName,
  resetNonce = 0,
  selectedSessionId = null,
  onSessionChange,
  onOpenSource,
  onOpenStudio,
}: {
  subjectId: string
  subjectName?: string
  resetNonce?: number
  selectedSessionId?: string | null
  onSessionChange?: (id: string) => void
  onOpenSource: (source: SourceSelection) => void
  onOpenStudio?: (itemId: string) => void
}) {
  const [sessionId, setSessionId] = useState<string | null>(selectedSessionId)
  const { data: saved, error: historyError, isLoading: historyLoading } = useSessionMessages(selectedSessionId)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const [pending, setPending] = useState(false)
  const scrollerRef = useRef<HTMLDivElement>(null)
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])
  const empty = messages.length === 0 && !pending

  useEffect(() => {
    if (saved) setMessages(saved)
  }, [saved])

  useEffect(() => {
    const el = scrollerRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages, pending])

  useEffect(() => {
    if (!resetNonce || selectedSessionId) return
    setSessionId(null)
    setMessages([])
    setQuestion('')
  }, [resetNonce, selectedSessionId])

  const openCitation = (c: Citation) => {
    onOpenSource(c)
  }

  const send = async (text?: string) => {
    const q = (text ?? question).trim()
    if (!q || pending) return

    const optimistic: ChatMessage = {
      id: `local-${Date.now()}`,
      role: 'user',
      content: q,
      query_type: null,
      citations: null,
      created_at: new Date().toISOString(),
    }
    setMessages((m) => [...m, optimistic])
    setQuestion('')
    setPending(true)

    try {
      const res = await ask(subjectId, q, sessionId)
      if (!mounted.current) {
        void mutate(keys.sessions(subjectId))
        return
      }
      setSessionId(res.session_id)
      setMessages((m) => [
        ...m,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: res.answer,
          query_type: res.query_type,
          citations: res.citations,
          studio_item_id: res.studio_item_id ?? null,
          created_at: new Date().toISOString(),
        },
      ])
      if (res.studio_item_id) void mutate(keys.studio(subjectId))
      await mutate(keys.sessions(subjectId))
      await mutate(keys.session(res.session_id))
      if (mounted.current) onSessionChange?.(res.session_id)
    } catch (err) {
      if (!mounted.current) return
      setMessages((m) => m.filter((x) => x.id !== optimistic.id))
      setQuestion(q)
      toast.error('Question failed', {
        description: friendlyChatError(err),
      })
    } finally {
      if (mounted.current) setPending(false)
    }
  }

  const composer = (
    <Composer
      question={question}
      setQuestion={setQuestion}
      pending={pending}
      onSend={() => send()}
      autoFocus={empty}
    />
  )

  if (historyLoading || historyError) return <p className="p-6 text-sm text-muted-foreground">
    {historyError ? 'Could not load this chat. Select it again to retry.' : 'Loading chat…'}
  </p>

  if (empty) {
    return (
      <div className="flex h-full min-h-0 flex-col items-center overflow-y-auto px-4 pb-16" style={{ justifyContent: 'safe center' }}>
        <div className="w-full min-w-0 max-w-2xl">
          <h2 className="mb-6 text-center font-display text-[1.65rem] leading-tight tracking-tight text-foreground sm:mb-8 sm:text-3xl lg:text-4xl">
            {subjectName ? `What can I help with in ${subjectName}?` : 'What can I help with?'}
          </h2>
          {composer}
          <div className="mt-5 flex flex-wrap justify-center gap-2">
            {SUGGESTIONS.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => send(s)}
                className="rounded-full border border-border bg-background px-3 py-1.5 text-[12px] text-muted-foreground transition-colors hover:border-foreground/30 hover:text-foreground"
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div ref={scrollerRef} className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-2xl space-y-6 px-4 py-6 sm:px-6 sm:py-8">
          <AnimatePresence initial={false}>
            {messages.map((m) => {
              const meta = m.query_type ? QUERY_TYPE[m.query_type] : null
              const Icon = meta?.icon

              return (
                <motion.div
                  key={m.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn('flex', m.role === 'user' ? 'justify-end' : 'justify-start')}
                >
                  {m.role === 'user' ? (
                    <div className="max-w-[80%] rounded-2xl bg-secondary px-4 py-2.5 text-[15px] leading-relaxed text-foreground">
                      {m.content}
                    </div>
                  ) : (
                    <article className="w-full min-w-0">
                      {meta && Icon && (
                        <p className="kicker mb-2 flex items-center gap-1.5">
                          <Icon className="h-3 w-3" />
                          {meta.label}
                        </p>
                      )}

                      <Markdown
                        onCitation={(n) => {
                          const c = m.citations?.find((x) => x.n === n)
                          if (c) openCitation(c)
                        }}
                      >
                        {m.content}
                      </Markdown>

                      {m.studio_item_id && onOpenStudio && (
                        <StudioChatCard itemId={m.studio_item_id} onOpen={onOpenStudio} />
                      )}

                      {m.citations && (
                        <CitationList citations={m.citations} onOpen={openCitation} />
                      )}
                    </article>
                  )}
                </motion.div>
              )
            })}
          </AnimatePresence>

          {pending && (
            <SgcdMachine compact label="Finding your answer" detail="Reading your subject’s sources and preparing a cited response…" />
          )}
        </div>
      </div>

      <div className="shrink-0 px-4 pb-4 pt-1 sm:px-6 sm:pb-5">
        {composer}
      </div>
    </div>
  )
}
