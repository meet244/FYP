'use client'

import { useEffect, useRef, useState } from 'react'
import { PanelLeft, Upload, WandSparkles, X } from 'lucide-react'

import { StudioSidebar } from '@/components/studio-sidebar'
import { Button } from '@/components/ui/button'
import { useSubjectUploads } from '@/hooks/use-subject-uploads'
import type { SourceSelection } from '@/components/source-viewer'
import { cn } from '@/lib/utils'
import type { Job } from '@/lib/api/types'

const SIDEBAR_KEY = 'cs-sidebar'

export function StudioShell({
  subjectId,
  title,
  subtitle,
  jobs,
  onQueued,
  onNewChat,
  onSelectChat,
  selectedSessionId,
  onOpenSource,
  right,
  rightLabel = 'Studio',
  children,
}: {
  subjectId?: string | null
  title?: React.ReactNode
  subtitle?: string
  jobs?: Job[]
  onQueued?: (job: Job) => void
  onNewChat?: () => void
  onSelectChat?: (id: string) => void
  selectedSessionId?: string | null
  onOpenSource?: (source: SourceSelection) => void
  right?: React.ReactNode
  /** Names the button that opens `right` as a slide-over below the xl breakpoint. */
  rightLabel?: string
  children: React.ReactNode
}) {
  const [mobileOpen, setMobileOpen] = useState(false)
  const [rightOpen, setRightOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const [dragging, setDragging] = useState(false)
  const dragDepth = useRef(0)
  const uploads = useSubjectUploads(subjectId, onQueued)
  const canUpload = !!subjectId && !!onQueued

  const resetDrag = () => {
    dragDepth.current = 0
    setDragging(false)
  }

  useEffect(() => {
    try {
      if (localStorage.getItem(SIDEBAR_KEY) === 'collapsed') setCollapsed(true)
    } catch {
      /* private mode */
    }
  }, [])

  useEffect(() => {
    setMobileOpen(false)
  }, [subjectId])

  const showSidebar = !!subjectId

  const persistCollapsed = (next: boolean) => {
    setCollapsed(next)
    try {
      localStorage.setItem(SIDEBAR_KEY, next ? 'collapsed' : 'open')
    } catch {
      /* private mode */
    }
  }

  const rail = (
    <StudioSidebar
      subjectId={subjectId}
      jobs={jobs}
      onQueued={onQueued}
      onNewChat={onNewChat}
      onSelectChat={onSelectChat}
      selectedSessionId={selectedSessionId}
      onOpenSource={onOpenSource}
      uploads={canUpload ? uploads : undefined}
      onCollapse={() => persistCollapsed(true)}
    />
  )

  return (
    <div
      className="relative flex h-dvh overflow-hidden bg-background"
      onDragEnterCapture={(event) => {
        if (!canUpload || !Array.from(event.dataTransfer.types).includes('Files')) return
        dragDepth.current += 1
        setDragging(true)
      }}
      onDragLeaveCapture={() => {
        dragDepth.current = Math.max(0, dragDepth.current - 1)
        if (dragDepth.current === 0) setDragging(false)
      }}
      onDragOver={(event) => {
        if (!canUpload || !Array.from(event.dataTransfer.types).includes('Files')) return
        // Dedicated drop areas, such as the syllabus uploader, handle their own files.
        if (event.defaultPrevented) {
          setDragging(false)
          return
        }
        event.preventDefault()
        event.dataTransfer.dropEffect = uploads.busy ? 'none' : 'copy'
        setDragging(true)
      }}
      onDropCapture={resetDrag}
      onDragEnd={resetDrag}
      onDrop={(event) => {
        if (!canUpload || event.defaultPrevented || !Array.from(event.dataTransfer.types).includes('Files')) return
        event.preventDefault()
        void uploads.ingest(Array.from(event.dataTransfer.files))
      }}
    >
      {showSidebar && (
        <aside
          className={cn(
            'hidden h-full shrink-0 overflow-hidden border-r border-border transition-[width] duration-200 ease-out md:flex md:flex-col',
            collapsed ? 'w-0 border-r-0' : 'w-[260px]'
          )}
        >
          <div className="flex h-full w-[260px] flex-col">{rail}</div>
        </aside>
      )}

      {showSidebar && mobileOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <button
            type="button"
            className="absolute inset-0 bg-foreground/20"
            aria-label="Close sidebar"
            onClick={() => setMobileOpen(false)}
          />
            <aside className="relative z-10 flex h-full w-[min(260px,88vw)] flex-col border-r border-border bg-background">
            <button
              type="button"
              className="absolute right-2 top-2 z-10 p-1 text-muted-foreground"
              onClick={() => setMobileOpen(false)}
              aria-label="Close"
            >
              <X className="h-4 w-4" />
            </button>
            {rail}
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex h-12 shrink-0 items-center gap-2 border-b border-border px-2 sm:px-3">
          {showSidebar && (
            <Button
              variant="ghost"
              size="icon"
              className={cn('h-8 w-8', !collapsed && 'md:hidden')}
              onClick={() => {
                if (window.matchMedia('(min-width: 768px)').matches) {
                  persistCollapsed(false)
                } else {
                  setMobileOpen(true)
                }
              }}
              aria-label="Open sidebar"
            >
              <PanelLeft className="h-4 w-4" />
            </Button>
          )}
          <div className="min-w-0 flex-1">
            {title ? (
              <>
                <div className="truncate font-display text-[17px] leading-none text-foreground">
                  {title}
                </div>
                {subtitle && (
                  <p className="mt-0.5 truncate text-[11px] text-muted-foreground">{subtitle}</p>
                )}
              </>
            ) : null}
          </div>
          {right && (
            <Button
              variant="outline"
              size="sm"
              className="h-8 gap-1.5 xl:hidden"
              onClick={() => setRightOpen(true)}
            >
              <WandSparkles className="h-3.5 w-3.5" /> {rightLabel}
            </Button>
          )}
        </div>

        <div
          className={cn(
            'grid min-h-0 flex-1',
            right ? 'grid-cols-1 xl:grid-cols-[minmax(0,1fr)_300px]' : 'grid-cols-1'
          )}
        >
          <section className="relative flex min-h-0 min-w-0 flex-col overflow-hidden bg-card">
            {children}
          </section>
          {right && (
            <aside className="hidden min-h-0 border-l border-border bg-muted/30 xl:flex xl:flex-col">
              {right}
            </aside>
          )}
        </div>
      </div>
      {right && rightOpen && (
        <div className="fixed inset-0 z-50 xl:hidden">
          <button
            type="button"
            className="absolute inset-0 bg-foreground/20"
            aria-label={`Close ${rightLabel}`}
            onClick={() => setRightOpen(false)}
          />
          <aside className="absolute inset-y-0 right-0 flex w-[min(320px,92vw)] flex-col border-l border-border bg-background">
            <button
              type="button"
              className="absolute right-2 top-2 z-10 p-1 text-muted-foreground"
              onClick={() => setRightOpen(false)}
              aria-label="Close"
            >
              <X className="h-4 w-4" />
            </button>
            {right}
          </aside>
        </div>
      )}
      {dragging && (
        <div className="pointer-events-none fixed inset-0 z-[100] flex items-center justify-center border-2 border-dashed border-primary bg-background/90 p-6" role="status" aria-live="polite">
          <div className="max-w-md text-center">
            <Upload className="mx-auto mb-4 h-10 w-10 text-primary" />
            <p className="font-display text-3xl text-foreground">
              {uploads.busy ? 'An upload is in progress' : 'Drop files to add to this subject'}
            </p>
            <p className="mt-3 text-sm text-muted-foreground">
              {uploads.busy ? 'Wait for it to finish, then drop your files.' : 'Documents go to Files. Audio goes to Audio recordings.'}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
