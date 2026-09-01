'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'
import { BookMarked, FileWarning, Mic, Plus, ServerCrash, Trash2 } from 'lucide-react'
import { toast } from 'sonner'
import { mutate } from 'swr'

import { AppHeader } from '@/components/app-header'
import { CreateSubjectDialog } from '@/components/create-subject-dialog'
import { Badge } from '@/components/ui/badge'
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

const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.05 } } }
const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] } },
}

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
    <motion.div variants={fadeUp} className="group relative">
      <Link
        href={`/subjects/${subject.id}`}
        className="card-hover block rounded-2xl border border-border/60 bg-card p-5"
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-primary/10">
            <BookMarked className="h-5 w-5 text-primary" />
          </div>
          {subject.code && (
            <Badge variant="secondary" className="font-mono text-[10px]">
              {subject.code}
            </Badge>
          )}
        </div>

        <h3 className="mb-1 truncate text-base font-bold text-foreground">{subject.name}</h3>
        <p className="mb-4 line-clamp-2 min-h-[2.5rem] text-sm leading-relaxed text-muted-foreground">
          {subject.description ?? 'No description.'}
        </p>

        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <Mic className="h-3.5 w-3.5" />
            {subject.lecture_count} {subject.lecture_count === 1 ? 'lecture' : 'lectures'}
          </span>
          {subject.has_syllabus ? (
            <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400">
              <BookMarked className="h-3.5 w-3.5" /> Syllabus
            </span>
          ) : (
            <span
              className="flex items-center gap-1.5 text-amber-600 dark:text-amber-400"
              title="Without a syllabus the pipeline runs a single unconditioned pass"
            >
              <FileWarning className="h-3.5 w-3.5" /> No syllabus
            </span>
          )}
          <span className="ml-auto">{relativeTime(subject.created_at)}</span>
        </div>
      </Link>

      <AlertDialog>
        <AlertDialogTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-2 top-2 h-7 w-7 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
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
              index. The original uploads stay on disk. This cannot be undone.
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
    </motion.div>
  )
}

export default function HomePage() {
  const { data: subjects, error, isLoading } = useSubjects()

  return (
    <div className="min-h-screen bg-background">
      <AppHeader />

      {/* Ambient orbs */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="animate-blob absolute -right-32 -top-32 h-[500px] w-[500px] rounded-full bg-primary/10 blur-[120px]" />
        <div className="animate-blob animation-delay-4000 absolute -left-48 top-1/2 h-[400px] w-[400px] rounded-full bg-blue-500/8 blur-[100px]" />
      </div>

      <main className="relative z-10 mx-auto max-w-6xl px-6 py-14">
        <div className="mb-10 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="mb-2 text-4xl font-black tracking-tight text-foreground">
              Your <span className="text-gradient">subjects</span>
            </h1>
            <p className="max-w-xl text-muted-foreground">
              Record a lecture, transcribe it against the syllabus, and ask questions with citations
              back to the moment it was said.
            </p>
          </div>
          <CreateSubjectDialog />
        </div>

        {error ? (
          <div className="rounded-2xl border border-destructive/30 bg-destructive/5 p-10 text-center">
            <ServerCrash className="mx-auto mb-3 h-8 w-8 text-destructive" />
            <h3 className="mb-1 font-semibold text-foreground">Cannot reach the backend</h3>
            <p className="mx-auto max-w-md text-sm text-muted-foreground">{error.message}</p>
          </div>
        ) : isLoading ? (
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-52 rounded-2xl" />
            ))}
          </div>
        ) : subjects && subjects.length > 0 ? (
          <motion.div
            variants={stagger}
            initial="hidden"
            animate="show"
            className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3"
          >
            {subjects.map((s) => (
              <SubjectCard key={s.id} subject={s} />
            ))}
          </motion.div>
        ) : (
          <div className="rounded-2xl border border-dashed border-border bg-card/40 p-16 text-center">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
              <BookMarked className="h-7 w-7 text-primary" />
            </div>
            <h3 className="mb-1.5 text-lg font-bold text-foreground">No subjects yet</h3>
            <p className="mx-auto mb-6 max-w-sm text-sm leading-relaxed text-muted-foreground">
              Create one for each course. Upload its syllabus first — it is what the decoder
              conditions on, and lectures transcribed without it get none of the method&rsquo;s
              benefit.
            </p>
            <CreateSubjectDialog
              trigger={
                <Button className="gap-1.5">
                  <Plus className="h-4 w-4" /> Create your first subject
                </Button>
              }
            />
          </div>
        )}
      </main>
    </div>
  )
}
