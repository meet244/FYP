import type { Job } from '@/lib/api/types'

/** Segment counts describe transcription work, not an estimate of time remaining. */
export function jobProgress(job: Job) {
  const overall = { value: Math.min(1, Math.max(0, job.progress)), label: 'Overall progress', suffix: 'overall' }
  if (job.stage !== 'transcribing' || !['running', 'cancelling'].includes(job.status)) return overall
  const pass = job.message?.match(/^(first|second) pass (\d+)\/(\d+)$/i)
  const span = job.message?.match(/^span (\d+)\/(\d+)$/i)
  const match = pass ?? span
  if (!match) return overall
  const current = Number(match[pass ? 2 : 1])
  const total = Number(match[pass ? 3 : 2])
  if (!total || current < 1 || current > total) return overall
  const contextual = job.asr_config?.method === 'sgcd'
  const contextPass = Boolean(span || pass?.[1].toLowerCase() === 'second')
  // The reported segment is about to decode; count only earlier segments as complete.
  return {
    value: !contextual && span ? 1 : (current - 1) / total,
    label: contextual ? contextPass ? 'Context pass progress' : 'First pass progress' : 'Transcript progress',
    suffix: contextual ? contextPass ? 'context pass' : 'first pass' : 'transcript',
  }
}
