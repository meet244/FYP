'use client'

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { Download, Pause, Play, RotateCcw, RotateCw } from 'lucide-react'

import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { audioUrl } from '@/lib/api/client'
import { timestamp } from '@/lib/format'
import { cn } from '@/lib/utils'

export interface AudioPlayerHandle {
  /** Seek and play — used by transcript spans, note headers and citations. */
  seek: (seconds: number) => void
}

const RATES = [1, 1.25, 1.5, 2]

export const AudioPlayer = forwardRef<
  AudioPlayerHandle,
  { lectureId: string; title: string; onTime?: (t: number) => void; className?: string }
>(function AudioPlayer({ lectureId, title, onTime, className }, ref) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [total, setTotal] = useState(0)
  const [rate, setRate] = useState(1)
  const [missing, setMissing] = useState(false)

  useImperativeHandle(ref, () => ({
    seek(seconds: number) {
      const el = audioRef.current
      if (!el) return
      el.currentTime = seconds
      void el.play().catch(() => {
        /* autoplay blocked before any user gesture — the seek still lands */
      })
    },
  }))

  useEffect(() => {
    const el = audioRef.current
    if (!el) return
    el.playbackRate = rate
  }, [rate])

  const skip = (delta: number) => {
    const el = audioRef.current
    if (!el) return
    el.currentTime = Math.max(0, Math.min(el.duration || 0, el.currentTime + delta))
  }

  if (missing) {
    return (
      <div
        className={cn(
          'rounded-xl border border-dashed border-border/60 px-4 py-3 text-xs text-muted-foreground',
          className
        )}
      >
        Normalised audio is not on disk for this lecture, so timestamps cannot be played back.
      </div>
    )
  }

  return (
    <div className={cn('rounded-xl border border-border/60 bg-card p-3', className)}>
      <audio
        ref={audioRef}
        src={audioUrl(lectureId)}
        preload="metadata"
        onLoadedMetadata={(e) => setTotal(e.currentTarget.duration)}
        onTimeUpdate={(e) => {
          setCurrent(e.currentTarget.currentTime)
          onTime?.(e.currentTarget.currentTime)
        }}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onError={() => setMissing(true)}
      />

      <div className="flex items-center gap-2">
        <Button
          size="icon"
          className="h-9 w-9 shrink-0 rounded-full"
          onClick={() => {
            const el = audioRef.current
            if (!el) return
            playing ? el.pause() : void el.play()
          }}
          aria-label={playing ? 'Pause' : 'Play'}
        >
          {playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
        </Button>

        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0 text-muted-foreground"
          onClick={() => skip(-10)}
          aria-label="Back 10 seconds"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0 text-muted-foreground"
          onClick={() => skip(10)}
          aria-label="Forward 10 seconds"
        >
          <RotateCw className="h-3.5 w-3.5" />
        </Button>

        <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
          {timestamp(current)}
        </span>

        <Slider
          value={[current]}
          max={total || 1}
          step={0.5}
          onValueChange={([v]) => {
            const el = audioRef.current
            if (el) el.currentTime = v
          }}
          className="mx-1 min-w-0 flex-1"
          aria-label="Seek"
        />

        <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
          {timestamp(total)}
        </span>

        <Button
          variant="ghost"
          size="sm"
          className="h-7 w-11 shrink-0 font-mono text-xs text-muted-foreground"
          onClick={() => setRate(RATES[(RATES.indexOf(rate) + 1) % RATES.length])}
          title="Playback speed"
        >
          {rate}×
        </Button>

        <Button
          asChild
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0 text-muted-foreground"
          title="Download normalised WAV"
        >
          <a href={audioUrl(lectureId)} download={`${title}.wav`}>
            <Download className="h-3.5 w-3.5" />
          </a>
        </Button>
      </div>
    </div>
  )
})
