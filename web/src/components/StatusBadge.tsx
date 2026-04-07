import type { PipelineStep } from '../api/types'

const STATUS_CLASS: Record<string, string> = {
  pending: 'badge-pending',
  completed: 'badge-completed',
  succeeded: 'badge-succeeded',
  failed: 'badge-failed',
  error: 'badge-error',
  running: 'badge-running',
  ready: 'badge-ready',
  planned: 'badge-planned',
}

interface Props {
  status: string
  step?: PipelineStep
}

export default function StatusBadge({ status, step }: Props) {
  const cls = STATUS_CLASS[status] || 'badge-pending'
  return (
    <span className={`badge ${cls}`}>
      {step ? `${step}: ` : ''}{status}
    </span>
  )
}
