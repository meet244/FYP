'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import {
  ArrowUp,
  BookOpen,
  FileText,
  GitCompare,
  History,
  ListTree,
  MessageCircle,
  PieChart,
  Plus,
  Search,
  Sparkles,
  Trash2,
} from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import { toast } from 'sonner'
import { mutate } from 'swr'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Textarea } from '@/components/ui/textarea'
import { Markdown } from '@/components/markdown'
import { ask, deleteSession, getSession } from '@/lib/api/client'
import { keys, useSessions } from '@/lib/api/hooks'
import { relativeTime } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { ChatMessage, Citation, QueryType } from '@/lib/api/types'

/**
 * The router classifies every question before retrieving, and the class changes
 * what the answer is built from. Showing it is honest about that: a `coverage`
 * answer came from a database group-by with no retrieval at all, and a reader
 * should be able to tell that apart from a cited transcript lookup.
 */
const QUERY_TYPE: Record<QueryType, { label: string; icon: React.ElementType; className: string }> = {
  lookup: { label: 'Lookup', icon: Search, className: 'text-blue-500' },
  explain: { label: 'Explain', icon: Sparkles, className: 'text-violet-500' },
  summary: { label: 'Summary', icon: FileText, className: 'text-emerald-500' },
  compare: { label: 'Compare', icon: GitCompare, className: 'text-amber-500' },
  quiz: { label: 'Quiz', icon: BookOpen, className: 'text-pink-500' },
  outline: { label: 'Outline', icon: ListTree, className: 'text-cyan-500' },
  coverage: { label: 'Coverage · no retrieval', icon: PieChart, className: 'text-primary' },
  smalltalk: { label: 'Chat', icon: MessageCircle, className: 'text-muted-foreground' },
}

const SUGGESTIONS = [
  'Summarise the last lecture',
  'What has been covered from the syllabus so far?',
  'Give me five practice questions on this topic',
  'Compare the two approaches the lecturer contrasted',
]

function CitationList({
  citations,
  onOpen,
}: {
  citations: Citation[]
  onOpen: (c: Citation) => void
}) {
  if (citations.length === 0) return null
  return (
    <div className="mt-3 border-t border-border/50 pt-2.5">
      <p className="mb-1.5 text-[10px] font-bold uppercase tracking-widest text-muted-foreground">
        Sources
      </p>
      <div className="flex flex-wrap gap-1.5">
        {citations.map((c) => (
          <button
            key={c.n}
            type="button"
            onClick={() => onOpen(c)}
            className="flex items-center gap-1.5 rounded-lg border border-border/60 bg-background px-2 py-1 text-[11px] transition-colors hover:border-primary/50 hover:bg-primary/5"
          >
            <span className="flex h-4 w-4 items-center justify-center rounded bg-primary/15 font-bold text-primary">
              {c.n}
            </span>
            <span className="max-w-[10rem] truncate text-foreground">
              {c.lecture_title ?? 'Lecture'}
            </span>
            <span className="font-mono text-muted-foreground">{c.timestamp}</span>
            {c.kind === 'note' && (
              <Badge variant="secondary" className="h-3.5 px-1 text-[9px]">
                notes
              </Badge>
            )}
          </button>
        ))}
      </div>
    </div>
  )
}

export function ChatPanel({ subjectId }: { subjectId: string }) {
  const router = useRouter()
  const { data: sessions } = useSessions(subjectId)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [question, setQuestion] = useState('')
  const [pending, setPending] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, pending])

  const openSession = async (id: string) => {
    try {
      const history = await getSession(id)
      setSessionId(id)
      setMessages(history)
    } catch (err) {
      toast.error('Could not open that conversation', {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const newSession = () => {
    setSessionId(null)
    setMessages([])
  }

  const removeSession = async (id: string) => {
    try {
      await deleteSession(id)
      await mutate(keys.sessions(subjectId))
      if (id === sessionId) newSession()
    } catch (err) {
      toast.error('Could not delete conversation', {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  const openCitation = (c: Citation) => {
    if (!c.lecture_id) return
    const t = c.start_s != null ? `?t=${Math.floor(c.start_s)}` : ''
    router.push(`/lectures/${c.lecture_id}${t}`)
  }

  const send = async (text?: string) => {
    const q = (text ?? question).trim()
    if (!q || pending) return

    // Optimistic user turn — the round trip includes routing, retrieval and a
    // full generation, so the question must not sit in the box looking ignored.
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
      setSessionId(res.session_id)
      setMessages((m) => [
        ...m,
        {
          id: `a-${Date.now()}`,
          role: 'assistant',
          content: res.answer,
          query_type: res.query_type,
          citations: res.citations,
          created_at: new Date().toISOString(),
        },
      ])
      await mutate(keys.sessions(subjectId))
    } catch (err) {
      setMessages((m) => m.filter((x) => x.id !== optimistic.id))
      setQuestion(q)
      toast.error('Question failed', {
        description: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setPending(false)
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-2 border-b border-border/60 px-4 py-2.5">
        <MessageCircle className="h-4 w-4 shrink-0 text-primary" />
        <span className="min-w-0 flex-1 truncate text-sm font-semibold text-foreground">
          {sessionId
            ? (sessions?.find((s) => s.id === sessionId)?.title ?? 'Conversation')
            : 'New conversation'}
        </span>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="h-7 w-7" title="Past conversations">
              <History className="h-3.5 w-3.5" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-72">
            {sessions && sessions.length > 0 ? (
              sessions.map((s) => (
                <DropdownMenuItem
                  key={s.id}
                  onSelect={() => openSession(s.id)}
                  className="flex items-start gap-2"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-foreground">{s.title}</p>
                    <p className="text-[10px] text-muted-foreground">
                      {s.message_count} messages · {relativeTime(s.updated_at)}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation()
                      e.preventDefault()
                      removeSession(s.id)
                    }}
                    className="mt-0.5 shrink-0 text-muted-foreground hover:text-destructive"
                    aria-label="Delete conversation"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </DropdownMenuItem>
              ))
            ) : (
              <div className="px-2 py-4 text-center text-xs text-muted-foreground">
                No past conversations.
              </div>
            )}
          </DropdownMenuContent>
        </DropdownMenu>

        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={newSession}
          title="New conversation"
        >
          <Plus className="h-3.5 w-3.5" />
        </Button>
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-4 p-4">
          {messages.length === 0 && !pending && (
            <div className="py-10 text-center">
              <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10">
                <Sparkles className="h-5 w-5 text-primary" />
              </div>
              <h4 className="mb-1 text-sm font-semibold text-foreground">
                Ask across this subject
              </h4>
              <p className="mx-auto mb-5 max-w-xs text-xs leading-relaxed text-muted-foreground">
                Answers are built from your own recordings and cite the moment each claim was made.
              </p>
              <div className="mx-auto flex max-w-sm flex-col gap-1.5">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => send(s)}
                    className="rounded-lg border border-border/60 px-3 py-2 text-left text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-foreground"
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          <AnimatePresence initial={false}>
            {messages.map((m) => {
              const meta = m.query_type ? QUERY_TYPE[m.query_type] : null
              const Icon = meta?.icon

              return (
                <motion.div
                  key={m.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn('flex', m.role === 'user' ? 'justify-end' : 'justify-start')}
                >
                  {m.role === 'user' ? (
                    <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-3.5 py-2 text-sm text-primary-foreground">
                      {m.content}
                    </div>
                  ) : (
                    <div className="w-full rounded-2xl rounded-bl-md border border-border/60 bg-card px-3.5 py-3">
                      {meta && Icon && (
                        <div className="mb-2 flex items-center gap-1.5">
                          <Icon className={cn('h-3 w-3', meta.className)} />
                          <span
                            className={cn(
                              'text-[10px] font-bold uppercase tracking-widest',
                              meta.className
                            )}
                          >
                            {meta.label}
                          </span>
                        </div>
                      )}

                      <Markdown
                        onCitation={(n) => {
                          const c = m.citations?.find((x) => x.n === n)
                          if (c) openCitation(c)
                        }}
                      >
                        {m.content}
                      </Markdown>

                      {m.citations && (
                        <CitationList citations={m.citations} onOpen={openCitation} />
                      )}
                    </div>
                  )}
                </motion.div>
              )
            })}
          </AnimatePresence>

          {pending && (
            <div className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
              <span className="flex gap-1">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="h-1.5 w-1.5 animate-bounce rounded-full bg-primary"
                    style={{ animationDelay: `${i * 120}ms` }}
                  />
                ))}
              </span>
              Routing, retrieving, answering…
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <div className="border-t border-border/60 p-3">
        <div className="relative">
          <Textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
            placeholder="Ask about these lectures…"
            rows={1}
            className="max-h-40 resize-none py-2.5 pr-11 text-sm"
          />
          <Button
            size="icon"
            className="absolute bottom-1.5 right-1.5 h-7 w-7"
            onClick={() => send()}
            disabled={pending || !question.trim()}
            aria-label="Send"
          >
            <ArrowUp className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>
    </div>
  )
}
