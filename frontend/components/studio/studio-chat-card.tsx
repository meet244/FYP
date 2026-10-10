'use client'

import { useEffect, useRef } from 'react'
import { Loader2 } from 'lucide-react'
import { mutate } from 'swr'

import { KIND_META } from '@/components/studio/kinds'
import { keys, useStudioItem } from '@/lib/api/hooks'
import { cn } from '@/lib/utils'

/** A Studio item requested from chat; live while generating, opens the viewer when ready. */
export function StudioChatCard({ itemId, onOpen }: { itemId: string; onOpen: (id: string) => void }) {
  const { data: item, error } = useStudioItem(itemId)
  const lastStatus = useRef(item?.status)

  // Keep the Studio list in step when generation finishes from the chat side.
  useEffect(() => {
    if (item && lastStatus.current && lastStatus.current !== item.status) void mutate(keys.studio(item.subject_id))
    lastStatus.current = item?.status
  }, [item])

  if (error) return <p className="mt-3 text-xs text-muted-foreground">This item was deleted.</p>
  if (!item) return null

  const meta = KIND_META[item.kind]
  const Icon = meta.icon
  const busy = item.status === 'pending' || item.status === 'running'
  const failed = item.status === 'failed'

  return (
    <button
      type="button"
      onClick={() => onOpen(item.id)}
      className="mt-3 flex w-full max-w-md items-center gap-3 border border-border bg-background px-4 py-3 text-left transition-colors hover:border-foreground/30"
    >
      <span className="flex h-9 w-9 shrink-0 items-center justify-center border border-border">
        {busy ? <Loader2 className={cn('h-4 w-4 animate-spin', meta.tint)} /> : <Icon className={cn('h-4 w-4', meta.tint)} />}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block truncate text-sm text-foreground">{item.title}</span>
        <span className={cn('block truncate text-xs', failed ? 'text-destructive' : 'text-muted-foreground')}>
          {busy ? 'Generating…' : failed ? (item.error ?? 'Generation failed. Open to retry.') : `Open ${meta.label.toLowerCase()}`}
        </span>
      </span>
    </button>
  )
}
