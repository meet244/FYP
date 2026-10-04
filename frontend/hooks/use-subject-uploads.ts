'use client'

import { useRef, useState } from 'react'
import { toast } from 'sonner'
import { mutate } from 'swr'

import { uploadLecture, uploadMaterials } from '@/lib/api/client'
import { keys } from '@/lib/api/hooks'
import type { ASROptions, Job } from '@/lib/api/types'

const AUDIO_EXT = new Set([
  '.m4a', '.mp3', '.wav', '.aac', '.ogg', '.opus', '.flac', '.mp4', '.3gp', '.amr', '.webm',
])
const MATERIAL_EXT = new Set(['.pdf', '.png', '.jpg', '.jpeg', '.webp', '.gif', '.txt', '.md', '.csv', '.docx', '.pptx'])
export const AUDIO_FILE_ACCEPT = [...AUDIO_EXT].join(',')
export const DOCUMENT_FILE_ACCEPT = [...MATERIAL_EXT].join(',')
export const SUBJECT_FILE_ACCEPT = [...AUDIO_EXT, ...MATERIAL_EXT].join(',')

function extOf(file: File) {
  const name = file.name.toLowerCase()
  const index = name.lastIndexOf('.')
  return index >= 0 ? name.slice(index) : ''
}

export type SubjectUploads = ReturnType<typeof useSubjectUploads>

export function useSubjectUploads(subjectId?: string | null, onQueued?: (job: Job) => void) {
  const [asr, setAsr] = useState<ASROptions>({})
  const [busy, setBusy] = useState(false)
  const uploading = useRef(false)

  const ingest = async (files: File[]) => {
    if (!files.length || !subjectId || !onQueued) return
    if (uploading.current) {
      toast.info('Please wait for the current upload to finish')
      return
    }
    const unsupported = files.filter(file => !AUDIO_EXT.has(extOf(file)) && !MATERIAL_EXT.has(extOf(file)))
    if (unsupported.length) {
      toast.error('Unsupported file type', {
        description: `${unsupported.map(file => file.name).join(', ')}. Choose recordings, PDFs, images, TXT, Markdown, CSV, DOCX or PPTX files.`,
      })
      return
    }

    uploading.current = true
    setBusy(true)
    const audio = files.filter(file => AUDIO_EXT.has(extOf(file)))
    const materials = files.filter(file => MATERIAL_EXT.has(extOf(file)))
    const options = { ...asr }
    try {
      for (const file of audio) {
        const job = await uploadLecture(subjectId, file, {
          ...options,
          title: file.name.replace(/\.[^.]+$/, ''),
        })
        onQueued(job)
        void mutate(keys.lectures(subjectId))
      }
      if (materials.length) {
        const queued = await uploadMaterials(subjectId, materials)
        queued.forEach(onQueued)
        void mutate(keys.materials(subjectId))
      }
      toast.success('Queued', {
        description: [
          audio.length ? `${audio.length} audio` : null,
          materials.length ? `${materials.length} file${materials.length === 1 ? '' : 's'}` : null,
        ].filter(Boolean).join(' · '),
      })
    } catch (err) {
      toast.error('Upload failed', {
        description: err instanceof Error ? err.message : String(err),
      })
    } finally {
      uploading.current = false
      setBusy(false)
    }
  }

  return { asr, setAsr, busy, ingest }
}
