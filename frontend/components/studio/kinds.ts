import {
  BarChart3,
  FileText,
  GalleryVerticalEnd,
  Network,
  Presentation,
  SquareCheckBig,
} from 'lucide-react'

import type { StudioKind } from '@/lib/api/types'

export interface KindMeta {
  label: string
  icon: React.ElementType
  /** Icon tint; one hue per kind so a generated item is recognisable in a list. */
  tint: string
  blurb: string
  count?: { label: string; default: number; choices: number[] }
}

export const STUDIO_KINDS: StudioKind[] = ['quiz', 'flashcards', 'mindmap', 'report', 'slides', 'infographic']

export const KIND_META: Record<StudioKind, KindMeta> = {
  quiz: {
    label: 'Quiz',
    icon: SquareCheckBig,
    tint: 'text-sky-700 dark:text-sky-300',
    blurb: 'Multiple-choice questions with hints and explanations',
    count: { label: 'Questions', default: 10, choices: [5, 10, 15, 20] },
  },
  flashcards: {
    label: 'Flashcards',
    icon: GalleryVerticalEnd,
    tint: 'text-amber-700 dark:text-amber-300',
    blurb: 'Term and answer cards to test recall',
    count: { label: 'Cards', default: 15, choices: [10, 15, 20, 30] },
  },
  mindmap: {
    label: 'Mind map',
    icon: Network,
    tint: 'text-violet-700 dark:text-violet-300',
    blurb: 'Topics and how they connect, expandable branch by branch',
  },
  report: {
    label: 'Reports',
    icon: FileText,
    tint: 'text-rose-700 dark:text-rose-300',
    blurb: 'Study guide, briefing, FAQ or glossary',
  },
  slides: {
    label: 'Slide deck',
    icon: Presentation,
    tint: 'text-fuchsia-700 dark:text-fuchsia-300',
    blurb: 'A presentable lesson with speaker notes',
    count: { label: 'Slides', default: 8, choices: [5, 8, 12, 16] },
  },
  infographic: {
    label: 'Infographic',
    icon: BarChart3,
    tint: 'text-emerald-700 dark:text-emerald-300',
    blurb: 'A one-page visual summary to revise from',
  },
}

export const REPORT_FORMATS = [
  { value: 'study_guide', label: 'Study guide' },
  { value: 'briefing', label: 'Briefing doc' },
  { value: 'faq', label: 'FAQ' },
  { value: 'glossary', label: 'Glossary' },
] as const
