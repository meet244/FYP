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
  materials: (id: string) => `/subjects/${id}/materials`,
  material: (id: string) => `/materials/${id}`,
  runs: (id: string) => `/lectures/${id}/runs`,
}

export const useASRModels = () => useSWR('/asr/models', api.getASRModels)
export const useRuns = (id: string | null) => useSWR(id ? keys.runs(id) : null, () => api.listRuns(id!))

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

export const useMaterials = (id: string | null) =>
  useSWR(id ? keys.materials(id) : null, () => api.listMaterials(id!))

export const useSessionMessages = (id: string | null) =>
  useSWR(id ? keys.session(id) : null, () => api.getSession(id!))

/**
 * Track background jobs to completion.
 *
 * Polling rather than a socket: the backend queue is a worker thread with a
 * `jobs` table and no push channel, and a transcription run is minutes long, so
 * a 1.5 s poll is both sufficient and cheap.
 */
export function useJobTracker(onSettled?: (job: Job) => void, scope?: { subject_id?: string; lecture_id?: string }) {
  const [jobs, setJobs] = useState<Job[]>([])
  const settled = useRef(onSettled)
  settled.current = onSettled
  const { data: recovered } = useSWR(
    scope ? ['/jobs', scope.subject_id, scope.lecture_id] : null,
    () => api.listJobs(scope),
    { refreshInterval: 1500 }
  )
  const completed = useRef(new Set<string>())
  const dismissed = useRef(new Set<string>())

  useEffect(() => {
    if (!recovered) return
    for (const job of recovered) {
      if (['succeeded', 'failed', 'cancelled'].includes(job.status) && !completed.current.has(job.id)) {
        completed.current.add(job.id)
        settled.current?.(job)
      }
    }
    setJobs((previous) => {
      const updated = previous.map(j => recovered.find(r => r.id === j.id) ?? j)
      const missing = recovered.filter(r => !dismissed.current.has(r.id) && !updated.some(j => j.id === r.id) &&
        ['queued', 'running', 'cancelling', 'failed'].includes(r.status))
      return [...updated, ...missing]
    })
  }, [recovered])

  const track = useCallback((job: Job) => {
    setJobs((prev) => (prev.some((j) => j.id === job.id) ? prev : [...prev, job]))
  }, [])

  const dismiss = useCallback((jobId: string) => {
    dismissed.current.add(jobId)
    setJobs((prev) => prev.filter((j) => j.id !== jobId))
  }, [])

  const active = jobs.filter((j) => ['queued', 'running', 'cancelling'].includes(j.status))

  useEffect(() => {
    if (scope) return // Scoped trackers recover and poll through SWR above.
    if (active.length === 0) return
    const ids = active.map((j) => j.id)

    const tick = async () => {
      const fresh = await Promise.all(
        ids.map((id) => api.getJob(id).catch(() => null))
      )
      for (const next of fresh) {
        if (next && ['succeeded', 'failed', 'cancelled'].includes(next.status) && !completed.current.has(next.id)) {
          completed.current.add(next.id)
          settled.current?.(next)
        }
      }
      setJobs((prev) =>
        prev.map((j) => {
          const next = fresh.find((f) => f?.id === j.id)
          if (!next) return j
          return next
        })
      )
    }

    const timer = setInterval(tick, 1_500)
    return () => clearInterval(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active.map((j) => j.id).join(','), !!scope])

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
    if (kind === 'ingest_material') {
      globalMutate(keys.materials(job.subject_id))
      if (job.material_id) globalMutate(keys.material(job.material_id))
    }
  }
  if (job.lecture_id) {
    globalMutate(keys.lecture(job.lecture_id))
    globalMutate(keys.transcript(job.lecture_id))
    globalMutate(keys.notes(job.lecture_id))
    globalMutate(keys.runs(job.lecture_id))
  }
}
