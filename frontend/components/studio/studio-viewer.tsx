'use client'

import { Loader2, RotateCw } from 'lucide-react'
import { mutate } from 'swr'
import { toast } from 'sonner'

import { KIND_META } from '@/components/studio/kinds'
import {
  FlashcardsView,
  InfographicView,
  MindmapView,
  QuizView,
  ReportView,
  SlidesView,
} from '@/components/studio/studio-views'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { retryStudioItem } from '@/lib/api/client'
import { keys, useStudioItem } from '@/lib/api/hooks'
import { cn } from '@/lib/utils'
import type {
  Citation,
  FlashcardsContent,
  InfographicContent,
  MindmapContent,
  QuizContent,
  ReportContent,
  SlidesContent,
  StudioItem,
} from '@/lib/api/types'

function Body({ item, onOpenSource }: { item: StudioItem; onOpenSource: (c: Citation) => void }) {
  const citations = item.citations ?? []
  const props = { citations, onOpenSource }

  if (item.status === 'failed') {
    const retry = async () => {
      try {
        await retryStudioItem(item.id)
        await mutate(keys.studioItem(item.id))
        await mutate(keys.studio(item.subject_id))
      } catch (err) {
        toast.error('Could not retry', { description: err instanceof Error ? err.message : String(err) })
      }
    }
    return (
      <div className="mx-auto max-w-md px-6 py-16 text-center">
        <p className="text-sm text-destructive">{item.error ?? 'Generation failed.'}</p>
        <Button className="mt-5 gap-1.5" onClick={retry}><RotateCw className="h-3.5 w-3.5" /> Try again</Button>
      </div>
    )
  }
  if (item.status !== 'ready' || !item.content) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 py-24 text-sm text-muted-foreground">
        <Loader2 className={cn('h-6 w-6 animate-spin', KIND_META[item.kind].tint)} />
        Reading {item.scope.label ?? 'your sources'} and generating…
      </div>
    )
  }
  // `key` resets interactive state (quiz answers, card position) per item.
  switch (item.kind) {
    case 'quiz':
      return <QuizView key={item.id} content={item.content as QuizContent} {...props} />
    case 'flashcards':
      return <FlashcardsView key={item.id} content={item.content as FlashcardsContent} {...props} />
    case 'mindmap':
      return <MindmapView key={item.id} content={item.content as MindmapContent} {...props} />
    case 'report':
      return <ReportView key={item.id} content={item.content as ReportContent} {...props} />
    case 'slides':
      return <SlidesView key={item.id} content={item.content as SlidesContent} {...props} />
    case 'infographic':
      return <InfographicView key={item.id} content={item.content as InfographicContent} {...props} />
  }
}

export function StudioViewer({ itemId, onClose, onOpenSource }: {
  itemId: string | null
  onClose: () => void
  onOpenSource: (c: Citation) => void
}) {
  const { data: item, error } = useStudioItem(itemId)
  const meta = item ? KIND_META[item.kind] : null
  const Icon = meta?.icon

  return (
    <Dialog open={!!itemId} onOpenChange={(open) => { if (!open) onClose() }}>
      <DialogContent className="flex h-[92dvh] w-[calc(100%-1.5rem)] max-w-6xl flex-col gap-0 overflow-hidden p-0 sm:max-w-6xl">
        <DialogHeader className="shrink-0 border-b border-border px-5 py-3.5 pr-12 text-left">
          <DialogTitle className="flex items-center gap-2 font-display text-xl font-normal">
            {Icon && <Icon className={cn('h-4.5 w-4.5 shrink-0', meta!.tint)} />}
            <span className="truncate">{item?.title ?? 'Studio'}</span>
          </DialogTitle>
          <DialogDescription className="truncate">
            {item ? `${meta!.label} from ${item.scope.label ?? 'this subject'}. Numbered chips open the source.` : 'Loading…'}
          </DialogDescription>
        </DialogHeader>
        <div className="flex min-h-0 flex-1 flex-col overflow-y-auto">
          {error ? (
            <p className="p-6 text-sm text-destructive">Could not open this item: {error.message}</p>
          ) : item ? (
            <Body item={item} onOpenSource={onOpenSource} />
          ) : null}
        </div>
      </DialogContent>
    </Dialog>
  )
}
