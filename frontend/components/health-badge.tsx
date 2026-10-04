'use client'

import { AlertTriangle, CircleCheck, CircleX, Cpu } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Separator } from '@/components/ui/separator'
import { useHealth } from '@/lib/api/hooks'
import { API_BASE } from '@/lib/api/client'
import { cn } from '@/lib/utils'

/**
 * Span length below ~20 s puts SGCD back in the regime where conditioning
 * *regresses* WER (paper §V-D). It is a silent failure, so the UI says so.
 */
const SPAN_FLOOR_S = 20

function Row({ label, value, warn }: { label: string; value: React.ReactNode; warn?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1 text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={cn(
          'truncate text-right font-mono',
          warn ? 'font-semibold text-amber-500' : 'text-foreground'
        )}
      >
        {value}
      </span>
    </div>
  )
}

export function HealthBadge() {
  const { data, error, isLoading } = useHealth()

  const down = !!error
  const spanLow = !!data && data.asr_default.method === 'sgcd' && data.sgcd.span_target_s < SPAN_FLOOR_S
  const warn = !!data && (!data.ffmpeg || spanLow || !data.llm_configured)

  const dot = down
    ? 'bg-destructive'
    : warn
      ? 'bg-amber-500'
      : isLoading
        ? 'bg-muted-foreground animate-pulse'
        : 'bg-emerald-500'

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 gap-2 px-2 text-muted-foreground">
          <span className={cn('h-1.5 w-1.5 rounded-full', dot)} />
          <span className="hidden text-xs font-medium sm:inline">
            {down ? 'Backend offline' : data ? data.asr.backend : 'Connecting'}
          </span>
        </Button>
      </PopoverTrigger>

      <PopoverContent align="end" className="w-80">
        {down ? (
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm font-semibold text-destructive">
              <CircleX className="h-4 w-4" /> Backend unreachable
            </div>
            <p className="text-xs leading-relaxed text-muted-foreground">
              Nothing is listening on{' '}
              <code className="rounded bg-muted px-1 py-0.5 font-mono">{API_BASE}</code>. Start it
              from the <code className="rounded bg-muted px-1 py-0.5 font-mono">backend/</code>{' '}
              directory:
            </p>
            <pre className="overflow-x-auto rounded-lg border border-border/60 bg-muted/40 p-2 font-mono text-[11px]">
              .venv/bin/uvicorn app.main:app --reload
            </pre>
          </div>
        ) : !data ? (
          <p className="text-xs text-muted-foreground">Checking backend…</p>
        ) : (
          <div className="space-y-1">
            <div className="flex items-center gap-2 text-sm font-medium text-foreground">
              <Cpu className="h-4 w-4 text-muted-foreground" /> Active configuration
            </div>

            <Separator className="my-2" />

            <p className="kicker pb-1">ASR</p>
            <Row label="backend" value={data.asr.backend} />
            <Row label="model" value={data.asr.model.split('/').pop()} />
            <Row label="language" value={data.asr.language ?? 'auto'} />

            <Separator className="my-2" />

            <p className="kicker pb-1">SGCD</p>
            <Row label="span target" value={`${data.sgcd.span_target_s}s`} warn={spanLow} />
            <Row label="retrieval k" value={data.sgcd.retrieval_k} />
            <Row label="prompt budget" value={`${data.sgcd.prompt_max_tokens} tok`} />
            <Row
              label="safeguard"
              value={data.asr_default.safeguard_enabled ? 'on' : 'off'}
            />

            <Separator className="my-2" />
            <Row label="llm" value={data.llm_model} />
            <Row label="LLM credential" value={data.llm_configured ? 'configured' : 'not configured'} warn={!data.llm_configured} />
            <Row label="ffmpeg" value={data.ffmpeg ? 'found' : 'MISSING'} warn={!data.ffmpeg} />

            {spanLow && (
              <p className="mt-3 flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 text-[11px] leading-relaxed text-amber-600 dark:text-amber-400">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>
                  Span target is below {SPAN_FLOOR_S}s. Conditioning regresses WER at short spans —
                  The project's Whisper experiments found worse results at short spans.
                </span>
              </p>
            )}
            {!data.ffmpeg && (
              <p className="mt-3 flex gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-2 text-[11px] leading-relaxed text-amber-600 dark:text-amber-400">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>
                  ffmpeg is not on PATH. Lecture uploads will fail until you install it.
                </span>
              </p>
            )}
            {!data.llm_configured && <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
              Transcription works locally. Configure Gemini for generated notes, syllabus parsing and image extraction. Chat can quote indexed sources without it.
            </p>}
            {!warn && (
              <p className="mt-3 flex items-center gap-2 text-[11px] text-emerald-600 dark:text-emerald-400">
                <CircleCheck className="h-3.5 w-3.5" /> Pipeline ready.
              </p>
            )}
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}
