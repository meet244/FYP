'use client'

import { use } from 'react'
import Link from 'next/link'
import { ArrowLeft, BookOpen, Mic, PieChart, ServerCrash } from 'lucide-react'

import { AppHeader } from '@/components/app-header'
import { ChatPanel } from '@/components/chat-panel'
import { CoveragePanel } from '@/components/coverage-panel'
import { JobList } from '@/components/job-progress'
import { LectureList } from '@/components/lecture-list'
import { SyllabusPanel } from '@/components/syllabus-panel'
import { UploadLectureDialog } from '@/components/upload-lecture-dialog'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { revalidateAfterJob, useJobTracker, useSubject } from '@/lib/api/hooks'

export default function SubjectPage({
  params,
}: {
  params: Promise<{ subjectId: string }>
}) {
  const { subjectId } = use(params)
  const { data: subject, error, isLoading } = useSubject(subjectId)
  const { jobs, track, dismiss } = useJobTracker(revalidateAfterJob)

  if (error) {
    return (
      <div className="min-h-screen bg-background">
        <AppHeader />
        <div className="mx-auto max-w-2xl px-6 py-24 text-center">
          <ServerCrash className="mx-auto mb-3 h-8 w-8 text-destructive" />
          <h2 className="mb-1 font-semibold text-foreground">Subject unavailable</h2>
          <p className="mb-6 text-sm text-muted-foreground">{error.message}</p>
          <Button asChild variant="outline">
            <Link href="/">
              <ArrowLeft className="mr-2 h-4 w-4" /> Back to subjects
            </Link>
          </Button>
        </div>
      </div>
    )
  }

  return (
    // Full-height split on desktop; on mobile the page scrolls and chat sits below.
    <div className="flex min-h-screen flex-col bg-background lg:h-screen">
      <AppHeader>
        <div className="flex min-w-0 items-center gap-2">
          <Button asChild variant="ghost" size="icon" className="h-7 w-7 shrink-0">
            <Link href="/" aria-label="Back to subjects">
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          {isLoading ? (
            <Skeleton className="h-4 w-40" />
          ) : (
            <>
              <span className="truncate text-sm font-semibold text-foreground">
                {subject?.name}
              </span>
              {subject?.code && (
                <Badge variant="secondary" className="shrink-0 font-mono text-[10px]">
                  {subject.code}
                </Badge>
              )}
            </>
          )}
        </div>
      </AppHeader>

      <div className="grid min-h-0 flex-1 lg:grid-cols-[1fr_400px]">
        {/* ── Left: sources ───────────────────────────────────── */}
        <div className="min-h-0 overflow-y-auto border-border/60 lg:border-r">
          <div className="mx-auto max-w-3xl p-5">
            <JobList jobs={jobs} onDismiss={dismiss} />

            <Tabs defaultValue="lectures">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                <TabsList>
                  <TabsTrigger value="lectures" className="gap-1.5 text-xs">
                    <Mic className="h-3.5 w-3.5" /> Lectures
                    {subject && subject.lecture_count > 0 && (
                      <span className="ml-0.5 text-muted-foreground">{subject.lecture_count}</span>
                    )}
                  </TabsTrigger>
                  <TabsTrigger value="syllabus" className="gap-1.5 text-xs">
                    <BookOpen className="h-3.5 w-3.5" /> Syllabus
                  </TabsTrigger>
                  <TabsTrigger value="coverage" className="gap-1.5 text-xs">
                    <PieChart className="h-3.5 w-3.5" /> Coverage
                  </TabsTrigger>
                </TabsList>

                <UploadLectureDialog
                  subjectId={subjectId}
                  hasSyllabus={!!subject?.has_syllabus}
                  onQueued={track}
                />
              </div>

              <TabsContent value="lectures" className="mt-0">
                <LectureList
                  subjectId={subjectId}
                  hasSyllabus={!!subject?.has_syllabus}
                  onQueued={track}
                />
              </TabsContent>

              <TabsContent value="syllabus" className="mt-0">
                <SyllabusPanel subjectId={subjectId} onQueued={track} />
              </TabsContent>

              <TabsContent value="coverage" className="mt-0">
                <CoveragePanel subjectId={subjectId} />
              </TabsContent>
            </Tabs>
          </div>
        </div>

        {/* ── Right: chat ─────────────────────────────────────── */}
        <div className="hidden min-h-0 lg:block">
          <ChatPanel subjectId={subjectId} />
        </div>
      </div>

      {/* On narrow screens chat gets its own full-height section below. */}
      <div className="h-[70vh] border-t border-border/60 lg:hidden">
        <ChatPanel subjectId={subjectId} />
      </div>
    </div>
  )
}
