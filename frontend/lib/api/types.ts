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
  material_count: number
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
  | 'failed'

export interface ASROptions {
  model_id?: string
  method?: 'baseline' | 'sgcd' | 's5'
  language?: 'hi' | 'en' | 'auto'
}

export interface ASRConfig extends ASROptions {
  model_id: string
  method: 'baseline' | 'sgcd' | 's5'
  language: 'hi' | 'en' | 'auto'
  backend: string
  model: string
  safeguard_enabled: boolean
}

export interface ASRModel {
  id: string
  name: string
  family: string
  supports_context: boolean
  languages: string[]
  default_language: string | null
  warning: string | null
  available: boolean
  unavailable_reason: string | null
}

export interface ASRCatalog {
  default: ASRConfig
  models: ASRModel[]
  methods: { id: string; name: string; families: string[]; model_ids?: string[]; experimental: boolean; warning?: string }[]
}

export interface TranscriptionRun {
  id: string
  lecture_id: string
  job_id: string
  config: ASRConfig
  stats: AsrStats
  created_at: string
}

export interface AsrStats {
  rescored?: boolean
  lm_sha256?: string | null
  model_id?: string
  model?: string
  backend?: string
  method?: string
  language?: string
  calibrated_safeguard?: boolean
  rtf?: number
  resumed_spans?: number
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
  asr_config?: ASRConfig | null
  summary?: string | null
}

export interface Span {
  baseline_text?: string | null
  prompt_tokens?: number
  avg_logprob?: number | null
  compression_ratio?: number | null
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

// --- materials ---
export type MaterialKind = 'pdf' | 'image' | 'doc'
export type MaterialStatus = 'uploaded' | 'processing' | 'ready' | 'failed'

export interface Material {
  id: string
  subject_id: string
  kind: MaterialKind
  title: string
  original_filename: string | null
  mime: string | null
  status: MaterialStatus
  n_chunks: number
  error: string | null
  created_at: string
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
  | 'studio'

export interface Citation {
  n: number
  kind: string | null
  lecture_id: string | null
  lecture_title: string | null
  start_s: number | null
  end_s: number | null
  timestamp: string | null
  note_id: string | null
  unit_id: string | null
  unit_title: string | null
  material_id: string | null
  material_title: string | null
  excerpt?: string | null
  page?: number | null
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
  studio_item_id?: string | null
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  query_type: QueryType | null
  citations: Citation[] | null
  studio_item_id?: string | null
  created_at: string
}

export interface ChatSessionSummary {
  id: string
  title: string
  message_count: number
  updated_at: string
}

// --- jobs ---
export type JobStatus = 'queued' | 'running' | 'cancelling' | 'cancelled' | 'succeeded' | 'failed'
export type JobKind = 'process_lecture' | 'generate_notes' | 'ingest_syllabus' | 'ingest_material'

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
  material_id: string | null
  asr_config?: ASRConfig | null
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
  llm_configured: boolean
  asr_default: ASRConfig
}

// --- studio ---
/** `app/studio/generators.py` KINDS. */
export type StudioKind = 'quiz' | 'flashcards' | 'mindmap' | 'report' | 'slides' | 'infographic'
export type StudioStatus = 'pending' | 'running' | 'ready' | 'failed'
export type ReportFormat = 'study_guide' | 'briefing' | 'faq' | 'glossary'
export type Difficulty = 'easy' | 'medium' | 'hard' | 'mixed'

export interface StudioScope {
  type: 'subject' | 'unit' | 'lecture'
  id: string | null
  label?: string
}

export interface StudioOptions {
  count?: number | null
  difficulty?: Difficulty | null
  format?: ReportFormat | null
  focus?: string | null
}

export interface QuizContent {
  title: string
  questions: {
    question: string
    options: string[]
    answer_index: number
    explanation: string
    hint: string | null
    difficulty: 'easy' | 'medium' | 'hard' | null
    sources: number[]
  }[]
}

export interface FlashcardsContent {
  title: string
  cards: { front: string; back: string; sources: number[] }[]
}

export interface MindNode {
  id: string
  label: string
  summary: string | null
  sources: number[]
  children: MindNode[]
}

export interface MindmapContent {
  title: string
  root: MindNode
}

export interface ReportContent {
  title: string
  markdown: string
}

export interface SlidesContent {
  title: string
  subtitle: string | null
  slides: { title: string; bullets: string[]; speaker_notes: string | null; sources: number[] }[]
}

export type InfographicIcon =
  | 'lightbulb' | 'gear' | 'chart' | 'book' | 'target' | 'layers'
  | 'network' | 'warning' | 'check' | 'clock' | 'cpu' | 'function'

export interface InfographicContent {
  title: string
  subtitle: string | null
  stats: { value: string; label: string; sources: number[] }[]
  sections: { heading: string; icon: InfographicIcon; points: string[]; sources: number[] }[]
  process: { title: string; steps: string[] } | null
  takeaway: string | null
}

export interface StudioContentMap {
  quiz: QuizContent
  flashcards: FlashcardsContent
  mindmap: MindmapContent
  report: ReportContent
  slides: SlidesContent
  infographic: InfographicContent
}

export interface StudioItem {
  id: string
  subject_id: string
  kind: StudioKind
  title: string
  scope: StudioScope
  options: StudioOptions
  status: StudioStatus
  content: StudioContentMap[StudioKind] | null
  citations: Citation[] | null
  model: string | null
  error: string | null
  created_at: string
  updated_at: string
}

export interface StudioCreate {
  kind: StudioKind
  scope: { type: StudioScope['type']; id: string | null }
  options: StudioOptions
}
