import Link from 'next/link'
import { ArrowLeft, FileQuestion } from 'lucide-react'
import { Button } from '@/components/ui/button'

export default function NotFound() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-4 bg-background px-6 text-center">
      <FileQuestion className="h-9 w-9 text-muted-foreground" />
      <h1 className="text-2xl font-bold text-foreground">Page not found</h1>
      <p className="max-w-sm text-sm text-muted-foreground">
        That subject or lecture does not exist — it may have been deleted.
      </p>
      <Button asChild variant="outline" className="mt-2">
        <Link href="/">
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to subjects
        </Link>
      </Button>
    </main>
  )
}
