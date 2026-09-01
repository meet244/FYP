'use client'

/** SWR bindings over `client.ts`, plus the job-polling loop the pipeline needs. */
import useSWR, { mutate as globalMutate } from 'swr'
import { useCallback, useEffect, useRef, useState } from 'react'

import * as api from './client'
import type { Job, JobKind } from './types'

// Keys are the API paths, so a mutation can invalidate by path prefix.
export const keys = {
  health: '/health',
  subjects: '/subjects',
  subject: (id: string) => `/subjects/${id}`,
  syllabus: (id: string) => `/subjects/${id}/syllabus`,
  coverage: (id: string) => `/subjects/${id}/coverage`,
  lectures: (id: string) => `/subjects/${id}/lectures`,
  lecture: (id: string) => `/lectures/${id}`,
  transcript: (id: string) => `/lectures/${id}/transcript`,
  notes: (id: string) => `/lectures/${id}/notes`,
  sessions: (id: string) => `/subjects/${id}/chat/sessions`,
  session: (id: string) => `/chat/sessions/${id}`,
}

export const useHealth = () =>
  useSWR(keys.health, api.getHealth, { refreshInterval: 30_000, shouldRetryOnError: false })

export const useSubjects = () => useSWR(keys.subjects, api.listSubjects)
export const useSubject = (id: string | null) =>
  useSWR(id ? keys.subject(id) : null, () => api.getSubject(id!))

/** 404 is the normal "no syllabus yet" state, so it must not retry forever. */
export const useSyllabus = (id: string | null) =>
  useSWR(id ? keys.syllabus(id) : null, () => api.getSyllabus(id!), {
    shouldRetryOnError: false,
  })

export const useCoverage = (id: string | null) =>
  useSWR(id ? keys.coverage(id) : null, () => api.getCoverage(id!), {
    shouldRetryOnError: false,
  })

export const useLectures = (id: string | null) =>
  useSWR(id ? keys.lectures(id) : null, () => api.listLectures(id!))

export const useLecture = (id: string | null) =>
  useSWR(id ? keys.lecture(id) : null, () => api.getLecture(id!))

export const useTranscript = (id: string | null) =>
  useSWR(id ? keys.transcript(id) : null, () => api.getTranscript(id!))

export const useNotes = (id: string | null) =>
  useSWR(id ? keys.notes(id) : null, () => api.getNotes(id!), { shouldRetryOnError: false })

export const useSessions = (id: string | null) =>
  useSWR(id ? keys.sessions(id) : null, () => api.listSessions(id!))

export const useSessionMessages = (id: string | null) =>
  useSWR(id ? keys.session(id) : null, () => api.getSession(id!))

/**
 * Track background jobs to completion.
 *
 * Polling rather than a socket: the backend queue is a worker thread with a
 * `jobs` table and no push channel, and a transcription run is minutes long, so
 * a 1.5 s poll is both sufficient and cheap.
 */
export function useJobTracker(onSettled?: (job: Job) => void) {
  const [jobs, setJobs] = useState<Job[]>([])
  const settled = useRef(onSettled)
  settled.current = onSettled

  const track = useCallback((job: Job) => {
    setJobs((prev) => (prev.some((j) => j.id === job.id) ? prev : [...prev, job]))
  }, [])

  const dismiss = useCallback((jobId: string) => {
    setJobs((prev) => prev.filter((j) => j.id !== jobId))
  }, [])

  const active = jobs.filter((j) => j.status === 'queued' || j.status === 'running')

  useEffect(() => {
    if (active.length === 0) return
    const ids = active.map((j) => j.id)

    const tick = async () => {
      const fresh = await Promise.all(
        ids.map((id) => api.getJob(id).catch(() => null))
      )
      setJobs((prev) =>
        prev.map((j) => {
          const next = fresh.find((f) => f?.id === j.id)
          if (!next) return j
          if (
            (j.status === 'queued' || j.status === 'running') &&
            (next.status === 'succeeded' || next.status === 'failed')
          ) {
            settled.current?.(next)
          }
          return next
        })
      )
    }

    const timer = setInterval(tick, 1_500)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active.map((j) => j.id).join(',')])

  return { jobs, active, track, dismiss }
}

/** Invalidate everything a finished job could have changed. */
export function revalidateAfterJob(job: Job) {
  const kind: JobKind = job.kind
  if (job.subject_id) {
    globalMutate(keys.subject(job.subject_id))
    globalMutate(keys.subjects)
    globalMutate(keys.coverage(job.subject_id))
    globalMutate(keys.lectures(job.subject_id))
    if (kind === 'ingest_syllabus') globalMutate(keys.syllabus(job.subject_id))
  }
  if (job.lecture_id) {
    globalMutate(keys.lecture(job.lecture_id))
    globalMutate(keys.transcript(job.lecture_id))
    globalMutate(keys.notes(job.lecture_id))
  }
}
