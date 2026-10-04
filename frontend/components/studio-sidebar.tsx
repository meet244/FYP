'use client'

import { useState } from 'react'
import Link from 'next/link'
import { AudioLines, ChevronLeft, Folder, MessageSquare, PanelLeftClose, Plus, SquarePen } from 'lucide-react'

import { CreateSubjectDialog } from '@/components/create-subject-dialog'
import { HealthBadge } from '@/components/health-badge'
import { ThemeToggle } from '@/components/app-header'
import { StudioLibrary } from '@/components/studio-library'
import type { SourceSelection } from '@/components/source-viewer'
import { Button } from '@/components/ui/button'
import type { SubjectUploads } from '@/hooks/use-subject-uploads'
import { useSessions, useSubjects } from '@/lib/api/hooks'
import { cn } from '@/lib/utils'
import type { Job } from '@/lib/api/types'

type RailTab = 'chats' | 'files' | 'audio'

function Brand({ onCollapse }: { onCollapse?: () => void }) {
  return (
    <div className="flex h-12 shrink-0 items-center gap-1 border-b border-border px-2">
      <Link href="/" className="flex min-w-0 flex-1 items-center gap-2.5 px-1">
        <span className="flex h-7 w-7 shrink-0 items-center justify-center border border-foreground/20 font-display text-[13px] leading-none text-foreground">
          CS
        </span>
          <span className="hidden truncate font-display text-[17px] leading-none tracking-tight text-foreground sm:inline">
          ClassScribe
        </span>
      </Link>
      {onCollapse && (
        <Button
          variant="ghost"
          size="icon"
          className="hidden h-8 w-8 md:inline-flex"
          onClick={onCollapse}
          aria-label="Collapse sidebar"
        >
          <PanelLeftClose className="h-4 w-4" />
        </Button>
      )}
    </div>
  )
}

function Footer() {
  return (
    <div className="flex shrink-0 items-center justify-between border-t border-border px-1 py-1">
      <HealthBadge />
      <ThemeToggle />
    </div>
  )
}

export function StudioSidebar({
  subjectId,
  jobs = [],
  onQueued,
  onNewChat,
  onCollapse,
  onSelectChat,
  selectedSessionId,
  uploads,
  onOpenSource,
}: {
  subjectId?: string | null
  jobs?: Job[]
  onQueued?: (job: Job) => void
  onNewChat?: () => void
  onCollapse?: () => void
  onSelectChat?: (id: string) => void
  selectedSessionId?: string | null
  uploads?: SubjectUploads
  onOpenSource?: (source: SourceSelection) => void
}) {
  const { data: subjects } = useSubjects()
  const { data: sessions, error: sessionsError } = useSessions(subjectId ?? null)
  const [tab, setTab] = useState<RailTab>('chats')
  const current = subjects?.find((s) => s.id === subjectId)

  return (
    <div className="flex h-full min-h-0 flex-col bg-muted/40">
      <Brand onCollapse={onCollapse} />

      {!subjectId ? (
        <>
          <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
            <p className="mb-1.5 px-1.5 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
              Subjects
            </p>
            <nav className="space-y-0.5">
              {(subjects ?? []).map((s) => (
                <Link
                  key={s.id}
                  href={`/subjects/${s.id}`}
                  className="block truncate px-2 py-1.5 text-[13px] text-muted-foreground hover:bg-background/70 hover:text-foreground"
                >
                  {s.name}
                </Link>
              ))}
            </nav>
            <CreateSubjectDialog
              trigger={
                <button
                  type="button"
                  className="mt-1 flex w-full items-center gap-1.5 px-2 py-1.5 text-[12px] text-muted-foreground hover:text-foreground"
                >
                  <Plus className="h-3 w-3" /> New subject
                </button>
              }
            />
          </div>
          <Footer />
        </>
      ) : (
        <>
          <div className="shrink-0 border-b border-border px-2 py-2">
            <Link
              href="/"
              className="mb-1 flex items-center gap-1 px-1.5 py-1 text-[12px] text-muted-foreground hover:text-foreground"
            >
              <ChevronLeft className="h-3.5 w-3.5" />
              All subjects
            </Link>
            <p className="truncate px-1.5 font-display text-[17px] leading-tight text-foreground">
              {current?.name ?? 'Subject'}
            </p>
          </div>

          <div className="grid shrink-0 grid-cols-3 border-b border-border">
            <button
              type="button"
              onClick={() => setTab('chats')}
              className={cn(
                'flex items-center justify-center gap-1.5 py-2 text-[12px]',
                tab === 'chats'
                  ? 'border-b-2 border-foreground text-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <MessageSquare className="h-3.5 w-3.5" />
              Chats
            </button>
            <button
              type="button"
              onClick={() => setTab('files')}
              className={cn(
                'flex items-center justify-center gap-1.5 py-2 text-[12px]',
                tab === 'files'
                  ? 'border-b-2 border-foreground text-foreground'
                  : 'text-muted-foreground hover:text-foreground'
              )}
            >
              <Folder className="h-3.5 w-3.5" />
              Files
            </button>
            <button type="button" onClick={() => setTab('audio')}
              className={cn('flex items-center justify-center gap-1 px-1 py-2 text-[11px] leading-tight',
                tab === 'audio' ? 'border-b-2 border-foreground text-foreground' : 'text-muted-foreground hover:text-foreground')}>
              <AudioLines className="h-3.5 w-3.5 shrink-0" /> <span>Audio recordings</span>
            </button>
          </div>

          <div className="min-h-0 flex-1 overflow-hidden">
            {tab === 'chats' ? (
              <div className="flex h-full flex-col p-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="mb-2 w-full justify-start gap-1.5"
                  onClick={onNewChat}
                >
                  <SquarePen className="h-3.5 w-3.5" /> New chat
                </Button>
                <div className="min-h-0 flex-1 overflow-y-auto">
                  {(sessions ?? []).map(s => <button key={s.id} type="button"
                    onClick={() => onSelectChat?.(s.id)}
                    className={cn('mb-1 w-full truncate px-2.5 py-2 text-left text-[13px]',
                      selectedSessionId === s.id ? 'bg-background text-foreground' : 'text-muted-foreground hover:bg-background')}>
                    {s.title ?? 'Untitled chat'}
                  </button>)}
                  {!sessions?.length && <p className="px-2 py-4 text-xs text-muted-foreground">{sessionsError ? 'Cannot load saved chats.' : 'Your saved chats will appear here.'}</p>}
                </div>
              </div>
            ) : uploads ? (
              <StudioLibrary
                key={tab}
                subjectId={subjectId}
                kind={tab}
                hasSyllabus={!!current?.has_syllabus}
                jobs={jobs}
                uploads={uploads}
                onQueued={onQueued}
                onOpenSource={onOpenSource}
              />
            ) : (
              <p className="px-4 py-8 text-center text-[12px] text-muted-foreground">
                Files will show up here.
              </p>
            )}
          </div>
          <Footer />
        </>
      )}
    </div>
  )
}
