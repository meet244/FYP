'use client'

import { use, useState } from 'react'
import { ServerCrash } from 'lucide-react'

import { ChatPanel } from '@/components/chat-panel'
import { JobList } from '@/components/job-progress'
import { ProcessingStatus } from '@/components/processing-stage'
import { StudioShell } from '@/components/studio-shell'
import { SourceViewer, type SourceSelection } from '@/components/source-viewer'
import { Skeleton } from '@/components/ui/skeleton'
import { revalidateAfterJob, useJobTracker, useSubject } from '@/lib/api/hooks'

export default function SubjectPage({
  params,
}: {
  params: Promise<{ subjectId: string }>
}) {
  const { subjectId } = use(params)
  return <SubjectView key={subjectId} subjectId={subjectId} />
}

function SubjectView({ subjectId }: { subjectId: string }) {
  const { data: subject, error, isLoading } = useSubject(subjectId)
  const { jobs, track, dismiss } = useJobTracker(revalidateAfterJob, { subject_id: subjectId })
  const [chatReset, setChatReset] = useState(0)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [source, setSource] = useState<SourceSelection | null>(null)

  if (error) {
    return (
      <StudioShell title="Subject unavailable">
        <div className="mx-auto max-w-lg px-6 py-24 text-center">
          <ServerCrash className="mx-auto mb-3 h-7 w-7 text-destructive" />
          <h2 className="font-display text-2xl text-foreground">Subject unavailable</h2>
          <p className="mt-2 text-sm text-muted-foreground">{error.message}</p>
        </div>
      </StudioShell>
    )
  }

  return (
    <StudioShell
      subjectId={subjectId}
      title={isLoading ? <Skeleton className="h-4 w-40" /> : subject?.name}
      jobs={jobs}
      onQueued={track}
      onNewChat={() => { setSessionId(null); setChatReset((n) => n + 1) }}
      selectedSessionId={sessionId}
      onSelectChat={setSessionId}
      onOpenSource={setSource}
    >
      <JobList jobs={jobs} onDismiss={dismiss} onQueued={track} />
      <ProcessingStatus jobs={jobs} />
      <ChatPanel
        key={`${subjectId}-${sessionId ?? 'new'}-${chatReset}`}
        subjectId={subjectId}
        subjectName={subject?.name}
        resetNonce={chatReset}
        selectedSessionId={sessionId}
        onSessionChange={setSessionId}
        onOpenSource={setSource}
      />
      <SourceViewer source={source} subjectId={subjectId} onClose={() => setSource(null)} />
    </StudioShell>
  )
}
