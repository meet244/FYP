'use client'

import Link from 'next/link'
import { FileText, Mic, Plus, ServerCrash, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import { mutate } from 'swr'

import { CreateSubjectDialog } from '@/components/create-subject-dialog'
import { AppHeader } from '@/components/app-header'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import { deleteSubject } from '@/lib/api/client'
import { keys, useSubjects } from '@/lib/api/hooks'
import { relativeTime } from '@/lib/format'
import type { Subject } from '@/lib/api/types'

function SubjectCard({ subject }: { subject: Subject }) {
  const remove = async () => {
    try {
      await deleteSubject(subject.id)
      await mutate(keys.subjects)
      toast.success(`Deleted “${subject.name}”`)
    } catch (err) {
      toast.error('Could not delete subject', {
        description: err instanceof Error ? err.message : String(err),
      })
    }
  }

  return (
    <article className="group relative flex flex-col border border-border bg-card">
      <Link href={`/subjects/${subject.id}`} className="flex flex-1 flex-col p-5">
        <div className="mb-6 flex items-start justify-between gap-3">
          {subject.code ? (
            <span className="font-mono text-[11px] tracking-wide text-muted-foreground">
              {subject.code}
            </span>
          ) : (
            <span className="kicker">Course</span>
          )}
          <span className="text-[11px] text-muted-foreground">
            {relativeTime(subject.created_at)}
          </span>
        </div>

        <h3 className="font-display text-[1.65rem] leading-tight tracking-tight text-foreground">
          {subject.name}
        </h3>
        <p className="mt-2 line-clamp-2 min-h-[2.5rem] text-sm leading-relaxed text-muted-foreground">
          {subject.description ?? 'No description yet.'}
        </p>

        <div className="mt-auto flex items-center gap-4 pt-6 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <Mic className="h-3 w-3" />
            {subject.lecture_count}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <FileText className="h-3 w-3" />
            {subject.material_count}
          </span>
          <span className={subject.has_syllabus ? 'text-foreground' : 'text-muted-foreground'}>
            {subject.has_syllabus ? 'Syllabus on file' : 'No syllabus'}
          </span>
        </div>
      </Link>

      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-1.5 top-1.5 h-7 w-7 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
            aria-label={`Delete ${subject.name}`}
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete “{subject.name}”?</AlertDialogTitle>
            <AlertDialogDescription>
              This removes the syllabus, every lecture, their transcripts and notes, and the vector
              index and stored source files. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={remove}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </article>
  )
}

export default function HomePage() {
  const { data: subjects, error, isLoading } = useSubjects()

  return (
    <div className="min-h-dvh bg-background">
      <AppHeader />
      <main className="mx-auto w-full max-w-5xl px-4 py-8 sm:px-8 sm:py-10">
        <header className="mb-8 flex flex-col gap-5 border-b border-border pb-8 sm:mb-10 sm:flex-row sm:items-end sm:justify-between sm:gap-6">
            <div className="max-w-xl">
              <p className="kicker mb-3">Library</p>
              <h1 className="font-display text-[2.15rem] leading-[1.1] tracking-tight text-foreground sm:text-5xl">
                Your subjects
              </h1>
              <p className="mt-3 max-w-md text-[15px] leading-relaxed text-muted-foreground">
                Open a course to chat, upload files, and ask questions grounded in that subject.
              </p>
            </div>
            <CreateSubjectDialog />
          </header>

          {error ? (
            <div className="border border-destructive/25 bg-destructive/5 px-8 py-14 text-center">
              <ServerCrash className="mx-auto mb-3 h-7 w-7 text-destructive" />
              <h3 className="mb-1 font-medium text-foreground">Cannot reach the backend</h3>
              <p className="mx-auto max-w-md text-sm text-muted-foreground">{error.message}</p>
            </div>
          ) : isLoading ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {[0, 1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-56" />
              ))}
            </div>
          ) : subjects && subjects.length > 0 ? (
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {subjects.map((s) => (
                <SubjectCard key={s.id} subject={s} />
              ))}
            </div>
          ) : (
            <div className="border border-dashed border-border px-8 py-20 text-center">
              <p className="kicker mb-4">Get started</p>
              <h3 className="font-display text-2xl text-foreground">No subjects yet</h3>
              <p className="mx-auto mt-2 mb-7 max-w-sm text-sm leading-relaxed text-muted-foreground">
                Create one course. Upload its syllabus, then drop a recording.
              </p>
              <CreateSubjectDialog
                trigger={
                  <Button className="gap-1.5">
                    <Plus className="h-4 w-4" /> Create a subject
                  </Button>
                }
              />
            </div>
          )}
        </main>
    </div>
  )
}
