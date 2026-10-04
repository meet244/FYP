'use client'

import { useASRModels } from '@/lib/api/hooks'
import type { ASROptions } from '@/lib/api/types'

export function ASRSelector({ value, onChange, disabled }: {
  value: ASROptions
  onChange: (next: ASROptions) => void
  disabled?: boolean
}) {
  const { data, error } = useASRModels()
  const selected = data?.models.find(m => m.id === (value.model_id ?? data.default.model_id))
  if (!data) return <p className="text-xs text-muted-foreground">{error ? 'Cannot load transcription models.' : 'Loading transcription models…'}</p>
  const method = value.method ?? data.default.method
  const methods = data.methods.filter(m => selected && m.families.includes(selected.family)
    && (!m.model_ids || m.model_ids.includes(selected.id)))
  const methodWarning = methods.find(m => m.id === method)?.warning
  return (
    <div className="space-y-2">
      <label className="block text-xs font-medium">
        Transcription model
        <select
          className="mt-1 h-8 w-full border border-border bg-background px-2 text-xs"
          aria-label="Transcription model"
          disabled={disabled}
          value={selected?.id ?? ''}
          onChange={e => {
            const model = data.models.find(m => m.id === e.target.value)!
            onChange({ model_id: model.id, method: model.id === data.default.model_id ? data.default.method : 'baseline',
              language: (model.default_language ?? 'auto') as ASROptions['language'] })
          }}
        >
          {data.models.map(m => <option key={m.id} value={m.id} disabled={!m.available}>
            {m.name}{m.id === 'qwen-0.6b' ? ' · recommended' : ''}{!m.available ? ' · unavailable' : ''}
          </option>)}
        </select>
      </label>
      <div className="grid grid-cols-2 gap-2">
        <label className="text-xs">
          Language
          <select aria-label="Transcription language" disabled={disabled} className="mt-1 h-8 w-full border border-border bg-background px-2"
            value={value.language ?? data.default.language}
            onChange={e => onChange({ ...value, language: e.target.value as ASROptions['language'] })}>
            <option value="auto">Auto detect</option>
            {selected?.languages.includes('hi') && <option value="hi">Hindi + English</option>}
            <option value="en">English</option>
          </select>
        </label>
        <label className="text-xs">
          Decoding method
          <select aria-label="Transcription method" disabled={disabled} className="mt-1 h-8 w-full border border-border bg-background px-2"
            value={method}
            onChange={e => onChange({ ...value, method: e.target.value as ASROptions['method'] })}>
            {methods.map(m => <option key={m.id} value={m.id}>{m.id === 'baseline' ? 'Fast · single pass' : m.id === 's5' ? 'Domain LM · 5 beams' : m.name}</option>)}
          </select>
        </label>
      </div>
      {method === 's5' && <div className="space-y-1 text-[11px] leading-relaxed text-muted-foreground">
        <p>S5 searches five transcript candidates per segment. Long recordings take more work than single-pass decoding.</p>
        <button type="button" disabled={disabled} className="text-foreground underline underline-offset-2 disabled:opacity-50"
          onClick={() => onChange({ ...value, method: 'baseline' })}>Use faster decoding for new recordings</button>
      </div>}
      {method === 'baseline' && <p className="text-[11px] leading-relaxed text-muted-foreground">Single pass without domain-LM rescoring. Transcript accuracy may differ from S5. This setting applies to new uploads; queued recordings keep their saved settings.</p>}
      {methodWarning && <p className="text-[11px] leading-relaxed text-muted-foreground">{methodWarning}</p>}
      {selected?.warning && method === 'sgcd' && <p className="text-[11px] leading-relaxed text-muted-foreground">{selected.warning}</p>}
      {selected?.family === 'parakeet' && <p className="text-[11px] text-muted-foreground">English only.</p>}
    </div>
  )
}
