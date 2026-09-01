/**
 * Thin fetch wrapper over the ClassScribe FastAPI backend.
 *
 * Calls go straight from the browser to uvicorn — the backend sets
 * `allow_origins=["*"]` because the whole design point is that audio never
 * leaves the machine, so there is no proxy layer to add here.
 */
import type {
  ChatResponse,
  ChatMessage,
  ChatSessionSummary,
  Coverage,
  Health,
  Job,
  JobStatus,
  Lecture,
  Note,
  Subject,
  SubjectCreate,
  Syllabus,
  Transcript,
  Unit,
  UnitUpdate,
} from './types'

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:8000'

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData
          ? {}
          : { 'Content-Type': 'application/json' }),
        ...init?.headers,
      },
    })
  } catch {
    // A dead backend is the single most likely failure in a local deployment,
    // so it gets a message that says what to do rather than "Failed to fetch".
    throw new ApiError(0, `Cannot reach the backend at ${API_BASE}. Is uvicorn running?`)
  }

  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = (await res.json()) as { detail?: string | { msg: string }[] }
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join('; ')
    } catch {
      /* non-JSON error body — keep statusText */
    }
    throw new ApiError(res.status, detail)
  }

  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

// --- meta ---
export const getHealth = () => req<Health>('/health')

// --- subjects ---
export const listSubjects = () => req<Subject[]>('/subjects')
export const getSubject = (id: string) => req<Subject>(`/subjects/${id}`)
export const createSubject = (body: SubjectCreate) =>
  req<Subject>('/subjects', { method: 'POST', body: JSON.stringify(body) })
export const deleteSubject = (id: string) =>
  req<void>(`/subjects/${id}`, { method: 'DELETE' })

// --- syllabus ---
export const getSyllabus = (subjectId: string) =>
  req<Syllabus>(`/subjects/${subjectId}/syllabus`)

export function uploadSyllabus(subjectId: string, file: File) {
  const fd = new FormData()
  fd.append('file', file)
  return req<Job>(`/subjects/${subjectId}/syllabus`, { method: 'POST', body: fd })
}

export const updateUnit = (subjectId: string, unitId: string, body: UnitUpdate) =>
  req<Unit>(`/subjects/${subjectId}/syllabus/units/${unitId}`, {
    method: 'PATCH',
    body: JSON.stringify(body),
  })

export const getCoverage = (subjectId: string) =>
  req<Coverage>(`/subjects/${subjectId}/coverage`)

// --- lectures ---
export const listLectures = (subjectId: string) =>
  req<Lecture[]>(`/subjects/${subjectId}/lectures`)
export const getLecture = (id: string) => req<Lecture>(`/lectures/${id}`)
export const getTranscript = (id: string) => req<Transcript>(`/lectures/${id}/transcript`)
export const getNotes = (id: string) => req<Note[]>(`/lectures/${id}/notes`)
export const reprocessLecture = (id: string) =>
  req<Job>(`/lectures/${id}/reprocess`, { method: 'POST' })
export const deleteLecture = (id: string) =>
  req<void>(`/lectures/${id}`, { method: 'DELETE' })

/** The normalised 16 kHz WAV — feed straight to an <audio> element for seeking. */
export const audioUrl = (lectureId: string) => `${API_BASE}/lectures/${lectureId}/audio`

export function uploadLecture(
  subjectId: string,
  file: File,
  opts: { title?: string; recordedAt?: string } = {}
) {
  const fd = new FormData()
  fd.append('file', file)
  if (opts.title) fd.append('title', opts.title)
  if (opts.recordedAt) fd.append('recorded_at', opts.recordedAt)
  return req<Job>(`/subjects/${subjectId}/lectures`, { method: 'POST', body: fd })
}

// --- chat ---
export const ask = (subjectId: string, question: string, sessionId?: string | null) =>
  req<ChatResponse>(`/subjects/${subjectId}/chat`, {
    method: 'POST',
    body: JSON.stringify({ question, session_id: sessionId ?? null }),
  })

export const listSessions = (subjectId: string) =>
  req<ChatSessionSummary[]>(`/subjects/${subjectId}/chat/sessions`)
export const getSession = (sessionId: string) =>
  req<ChatMessage[]>(`/chat/sessions/${sessionId}`)
export const deleteSession = (sessionId: string) =>
  req<void>(`/chat/sessions/${sessionId}`, { method: 'DELETE' })

// --- jobs ---
export const getJob = (id: string) => req<Job>(`/jobs/${id}`)

export function listJobs(params: {
  subject_id?: string
  lecture_id?: string
  status?: JobStatus
  limit?: number
} = {}) {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v != null) qs.set(k, String(v))
  const q = qs.toString()
  return req<Job[]>(`/jobs${q ? `?${q}` : ''}`)
}
