'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Plus } from 'lucide-react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Spinner } from '@/components/ui/spinner'
import { createSubject } from '@/lib/api/client'
import { keys } from '@/lib/api/hooks'
import { mutate } from 'swr'

export function CreateSubjectDialog({ trigger }: { trigger?: React.ReactNode }) {
  const router = useRouter()
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [name, setName] = useState('')
  const [code, setCode] = useState('')
  const [description, setDescription] = useState('')

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) return
    setBusy(true)
    try {
      const subject = await createSubject({
        name: name.trim(),
        code: code.trim() || null,
        description: description.trim() || null,
      })
      await mutate(keys.subjects)
      setOpen(false)
      setName('')
      setCode('')
      setDescription('')
      router.push(`/subjects/${subject.id}`)
    } catch (err) {
      toast.error('Could not create subject', {
        description: err instanceof Error ? err.message : String(err),
      })
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {trigger ?? (
          <Button size="sm" className="gap-1.5 shadow-md shadow-primary/20">
            <Plus className="h-3.5 w-3.5" /> New subject
          </Button>
        )}
      </DialogTrigger>

      <DialogContent className="sm:max-w-md">
        <form onSubmit={submit}>
          <DialogHeader>
            <DialogTitle>New subject</DialogTitle>
            <DialogDescription>
              One subject per course. Chat is scoped to a subject so an Operating Systems question
              never retrieves from your Networks lectures.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-5">
            <div className="space-y-1.5">
              <Label htmlFor="subject-name">Name</Label>
              <Input
                id="subject-name"
                autoFocus
                placeholder="Operating Systems"
                value={name}
                onChange={(e) => setName(e.target.value)}
                maxLength={256}
                required
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="subject-code">
                Course code <span className="text-muted-foreground">(optional)</span>
              </Label>
              <Input
                id="subject-code"
                placeholder="ITC501"
                value={code}
                onChange={(e) => setCode(e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="subject-desc">
                Description <span className="text-muted-foreground">(optional)</span>
              </Label>
              <Textarea
                id="subject-desc"
                rows={2}
                placeholder="Semester 5 core — process scheduling, memory, file systems."
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>
          </div>

          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => setOpen(false)} disabled={busy}>
              Cancel
            </Button>
            <Button type="submit" disabled={busy || !name.trim()}>
              {busy && <Spinner className="mr-2 h-3.5 w-3.5" />}
              Create subject
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
