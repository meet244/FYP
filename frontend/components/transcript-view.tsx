'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { Copy, Search, ShieldAlert, X } from 'lucide-react'
import { toast } from 'sonner'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { timestamp } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { Span, Unit } from '@/lib/api/types'

function highlight(text: string, needle: string) {
  if (!needle.trim()) return text
  const parts = text.split(new RegExp(`(${needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'))
  return parts.map((p, i) =>
    p.toLowerCase() === needle.toLowerCase() ? (
      <mark key={i} className="rounded bg-primary/25 px-0.5 text-foreground">
        {p}
      </mark>
    ) : (
      p
    )
  )
}

export function TranscriptView({
  spans,
  units,
  currentTime,
  onSeek,
}: {
  spans: Span[]
  units: Unit[]
  currentTime: number
  onSeek: (seconds: number) => void
}) {
  const [query, setQuery] = useState('')
  const [onlyFallbacks, setOnlyFallbacks] = useState(false)
  const [follow, setFollow] = useState(true)
  const activeRef = useRef<HTMLDivElement>(null)

  const unitTitle = useMemo(
    () => new Map(units.map((u) => [u.id, `${u.unit_key} · ${u.title}`])),
    [units]
  )

  const activeIndex = spans.findIndex((s) => currentTime >= s.start_s && currentTime < s.end_s)

  useEffect(() => {
    if (follow && activeIndex >= 0) {
      activeRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [activeIndex, follow])

  const visible = spans.filter((s) => {
    if (onlyFallbacks && !s.safeguard_fallback) return false
    if (query.trim() && !s.text.toLowerCase().includes(query.toLowerCase())) return false
    return true
  })

  const fallbackCount = spans.filter((s) => s.safeguard_fallback).length

  const copyAll = async () => {
    await navigator.clipboard.writeText(spans.map((s) => s.text).join(' '))
    toast.success('Transcript copied')
  }

  return (
    <TooltipProvider delayDuration={200}>
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <div className="relative min-w-48 flex-1">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search the transcript…"
              className="h-8 pl-8 pr-8 text-sm"
            />
            {query && (
              <button
                type="button"
                onClick={() => setQuery('')}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                aria-label="Clear search"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Switch id="follow" checked={follow} onCheckedChange={setFollow} />
            <Label htmlFor="follow" className="text-xs text-muted-foreground">
              Follow audio
            </Label>
          </div>

          {fallbackCount > 0 && (
            <div className="flex items-center gap-2">
              <Switch
                id="fallbacks"
                checked={onlyFallbacks}
                onCheckedChange={setOnlyFallbacks}
              />
              <Label htmlFor="fallbacks" className="text-xs text-muted-foreground">
                Reverted only ({fallbackCount})
              </Label>
            </div>
          )}

          <Button variant="ghost" size="sm" className="h-8 gap-1.5 text-xs" onClick={copyAll}>
            <Copy className="h-3 w-3" /> Copy
          </Button>
        </div>

        {visible.length === 0 ? (
          <p className="border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
            {spans.length === 0 ? 'No transcript spans yet.' : 'No spans match this filter.'}
          </p>
        ) : (
          <div className="space-y-0.5">
            {visible.map((s) => {
              const active = s.index === spans[activeIndex]?.index
              return (
                <div
                  key={s.index}
                  ref={active ? activeRef : undefined}
                  onClick={() => onSeek(s.start_s)}
                  className={cn(
                    'group flex cursor-pointer gap-3 border-l-2 px-2.5 py-2.5 transition-colors',
                    active
                      ? 'border-foreground bg-muted/40'
                      : 'border-transparent hover:bg-muted/30'
                  )}
                >
                  <button
                    type="button"
                    className={cn(
                      'shrink-0 pt-0.5 font-mono text-xs tabular-nums transition-colors',
                      active ? 'font-medium text-foreground' : 'text-muted-foreground group-hover:text-foreground'
                    )}
                  >
                    {timestamp(s.start_s)}
                  </button>

                  <div className="min-w-0 flex-1">
                    <p
                      className={cn(
                        'text-sm leading-relaxed',
                        active ? 'text-foreground' : 'text-muted-foreground'
                      )}
                    >
                      {highlight(s.text, query)}
                    </p>

                    {(s.safeguard_fallback || s.retrieved_unit_ids.length > 0) && (
                      <div className="mt-1 flex flex-wrap items-center gap-1.5 opacity-0 transition-opacity group-hover:opacity-100">
                        {s.safeguard_fallback && (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <Badge
                                variant="outline"
                                className="h-4 gap-1 border-amber-500/30 px-1.5 text-[9px] text-amber-600 dark:text-amber-400"
                              >
                                <ShieldAlert className="h-2.5 w-2.5" /> reverted
                              </Badge>
                            </TooltipTrigger>
                            <TooltipContent side="bottom" className="max-w-60 text-xs">
                              The safeguard rejected the conditioned decode for this span and kept
                              the unconditioned hypothesis.
                            </TooltipContent>
                          </Tooltip>
                        )}

                        {s.retrieved_unit_ids.map((id) => (
                          <Badge
                            key={id}
                            variant="secondary"
                            className="h-4 px-1.5 text-[9px] font-normal"
                            title="Syllabus unit retrieved for this span"
                          >
                            {unitTitle.get(id)?.split(' · ')[0] ?? id.slice(0, 8)}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </TooltipProvider>
  )
}
