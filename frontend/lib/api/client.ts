/**
 * Thin fetch wrapper over the ClassScribe FastAPI backend.
 *
 * Calls go straight from the browser to uvicorn. The backend permits configured
 * frontend origins; audio recognition runs locally.
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
  Material,
  ASRCatalog,
  ASROptions,
  TranscriptionRun,
  Span,
  StudioCreate,
  StudioItem,
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
export const getASRModels = () => req<ASRCatalog>('/asr/models')

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
export const reprocessLecture = (id: string, options?: ASROptions) =>
  req<Job>(`/lectures/${id}/reprocess`, {
    method: 'POST',
    body: JSON.stringify({ model_id: options?.model_id, method: options?.method, language: options?.language }),
  })
export const regenerateNotes = (id: string) => req<Job>(`/lectures/${id}/notes/regenerate`, { method: 'POST' })
export const listRuns = (id: string) => req<TranscriptionRun[]>(`/lectures/${id}/runs`)
export const getRun = (id: string, runId: string) =>
  req<TranscriptionRun & { spans: Span[] }>(`/lectures/${id}/runs/${runId}`)
export const deleteLecture = (id: string) =>
  req<void>(`/lectures/${id}`, { method: 'DELETE' })

/** The normalised 16 kHz WAV — feed straight to an <audio> element for seeking. */
export const audioUrl = (lectureId: string) => `${API_BASE}/lectures/${lectureId}/audio`
export const materialFileUrl = (materialId: string) => `${API_BASE}/materials/${materialId}/file`
export const syllabusFileUrl = (subjectId: string) => `${API_BASE}/subjects/${subjectId}/syllabus/file`
export const getMaterialPreview = (id: string) => req<Material & { text: string | null }>(`/materials/${id}`)

export function uploadLecture(
  subjectId: string,
  file: File,
  opts: { title?: string; recordedAt?: string } & ASROptions = {}
) {
  const fd = new FormData()
  fd.append('file', file)
  if (opts.title) fd.append('title', opts.title)
  if (opts.recordedAt) fd.append('recorded_at', opts.recordedAt)
  if (opts.model_id) fd.append('model_id', opts.model_id)
  if (opts.method) fd.append('method', opts.method)
  if (opts.language) fd.append('language', opts.language)
  return req<Job>(`/subjects/${subjectId}/lectures`, { method: 'POST', body: fd })
}

export function uploadMaterials(subjectId: string, files: File[]) {
  const fd = new FormData()
  for (const file of files) fd.append('files', file)
  return req<Job[]>(`/subjects/${subjectId}/materials`, { method: 'POST', body: fd })
}

export const listMaterials = (subjectId: string) =>
  req<Material[]>(`/subjects/${subjectId}/materials`)
export const deleteMaterial = (id: string) =>
  req<void>(`/materials/${id}`, { method: 'DELETE' })

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

// --- studio ---
export const listStudioItems = (subjectId: string) =>
  req<StudioItem[]>(`/subjects/${subjectId}/studio`)
export const getStudioItem = (id: string) => req<StudioItem>(`/studio/${id}`)
export const createStudioItem = (subjectId: string, body: StudioCreate) =>
  req<StudioItem>(`/subjects/${subjectId}/studio`, { method: 'POST', body: JSON.stringify(body) })
export const retryStudioItem = (id: string) =>
  req<StudioItem>(`/studio/${id}/retry`, { method: 'POST' })
export const deleteStudioItem = (id: string) =>
  req<void>(`/studio/${id}`, { method: 'DELETE' })

// --- jobs ---
export const getJob = (id: string) => req<Job>(`/jobs/${id}`)
export const cancelJob = (id: string) => req<Job>(`/jobs/${id}/cancel`, { method: 'POST' })
export const retryJob = (id: string) => req<Job>(`/jobs/${id}/retry`, { method: 'POST' })

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
