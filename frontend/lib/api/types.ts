/**
 * TypeScript mirror of the backend contract in `backend/app/schemas.py`.
 *
 * Field names are snake_case because they come straight off the wire — FastAPI
 * serialises the Pydantic models as-is, and renaming here would only create a
 * second place to keep in sync.
 */

// --- subjects ---
export interface Subject {
  id: string
  name: string
  code: string | null
  description: string | null
  created_at: string
  has_syllabus: boolean
  lecture_count: number
}

export interface SubjectCreate {
  name: string
  code?: string | null
  description?: string | null
}

// --- syllabus ---
export interface Unit {
  id: string
  unit_key: string
  order_index: number
  title: string
  /** The code-mixed narration SGCD actually conditions on. Editable by hand. */
  prose: string
  keywords: string[]
}

export interface Syllabus {
  id: string
  source_filename: string | null
  provenance: string
  created_at: string
  units: Unit[]
}

export interface UnitUpdate {
  title?: string
  prose?: string
  keywords?: string[]
}

// --- lectures ---
/** Set in `jobs/handlers.py` as the pipeline advances. */
export type LectureStatus =
  | 'uploaded'
  | 'transcribing'
  | 'transcribed'
  | 'summarising'
  | 'ready'

export interface AsrStats {
  n_spans: number
  mean_span_s: number
  /** False when the lecture was decoded without a syllabus — one pass, no SGCD. */
  conditioned: boolean
  safeguard_fallbacks: number
  safeguard_fallback_rate: number
  elapsed_s: number
  realtime_factor: number | null
  /** Descriptive check that the dual-script convention survived. Not a WER proxy. */
  script_mix: Record<string, number>
}

export interface Lecture {
  id: string
  subject_id: string
  title: string
  status: LectureStatus
  duration_s: number | null
  recorded_at: string | null
  created_at: string
  error: string | null
  asr_stats: AsrStats | null
}

export interface Span {
  index: number
  start_s: number
  end_s: number
  text: string
  retrieved_unit_ids: string[]
  /** True when the safeguard reverted this span to the unconditioned hypothesis. */
  safeguard_fallback: boolean
}

export interface Transcript {
  lecture_id: string
  title: string
  status: LectureStatus
  duration_s: number | null
  text: string
  spans: Span[]
  asr_stats: AsrStats | null
}

// --- notes ---
export interface Term {
  term: string
  definition: string
}

export interface Outcome {
  text: string
  bloom_level: string | null
}

export interface Note {
  id: string
  topic: string
  markdown: string
  order_index: number
  start_s: number | null
  end_s: number | null
  unit_id: string | null
  terms: Term[]
  outcomes: Outcome[]
}

// --- coverage ---
export interface CoverageNote {
  lecture_id: string
  lecture_title: string
  note_id: string
  topic: string
  start_s: number | null
}

export interface CoveredUnit {
  unit_id: string
  unit_key: string
  title: string
  note_count: number
  notes: CoverageNote[]
}

export interface OutstandingUnit {
  unit_id: string
  unit_key: string
  title: string
}

export interface Coverage {
  subject_id: string
  subject_name: string
  total_units: number
  covered_units: number
  coverage_fraction: number | null
  covered: CoveredUnit[]
  outstanding: OutstandingUnit[]
  unattributed_notes: number
}

// --- chat ---
/** `app/rag/router.py` QUERY_TYPES. */
export type QueryType =
  | 'lookup'
  | 'explain'
  | 'summary'
  | 'compare'
  | 'quiz'
  | 'outline'
  | 'coverage'
  | 'smalltalk'

export interface Citation {
  n: number
  kind: string | null
  lecture_id: string | null
  lecture_title: string | null
  start_s: number | null
  end_s: number | null
  timestamp: string | null
  note_id: string | null
}

export interface ChatRequest {
  question: string
  session_id?: string | null
}

export interface ChatResponse {
  session_id: string
  answer: string
  query_type: QueryType
  citations: Citation[]
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  query_type: QueryType | null
  citations: Citation[] | null
  created_at: string
}

export interface ChatSessionSummary {
  id: string
  title: string
  message_count: number
  updated_at: string
}

// --- jobs ---
export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'
export type JobKind = 'process_lecture' | 'ingest_syllabus'

export interface Job {
  id: string
  kind: JobKind
  status: JobStatus
  /** process_lecture: audio|transcribing|notes|indexing|done · ingest_syllabus: parsing|writing */
  stage: string | null
  progress: number
  message: string | null
  error: string | null
  lecture_id: string | null
  subject_id: string | null
  created_at: string
  updated_at: string
}

// --- meta ---
export interface Health {
  status: string
  ffmpeg: boolean
  asr: { backend: string; model: string; language: string | null }
  sgcd: {
    span_target_s: number
    retrieval_k: number
    prompt_max_tokens: number
    safeguard_enabled: boolean
  }
  llm_model: string
}
