import { PIPELINE_STEPS } from '../api/types'
import type { StepRecord, PipelineStep } from '../api/types'
import StatusBadge from './StatusBadge'

const STEP_LABELS: Record<PipelineStep, string> = {
  merge: 'Merge',
  train: 'Train',
  abliterate: 'Abliterate',
  quantize: 'Quantize',
}

const STEP_ICONS: Record<PipelineStep, string> = {
  merge: '🔀',
  train: '🏋️',
  abliterate: '✂️',
  quantize: '📦',
}

interface Props {
  steps: Record<string, StepRecord>
  onStepClick?: (step: PipelineStep) => void
}

export default function PipelineStepper({ steps, onStepClick }: Props) {
  const stepKeys = PIPELINE_STEPS as unknown as PipelineStep[]
  const completedCount = stepKeys.filter((s) => {
    const rec = steps[s]
    return rec && (rec.status === 'completed' || rec.status === 'succeeded')
  }).length

  return (
    <div className="card">
      <h3>Pipeline Progress</h3>
      <div style={{ display: 'flex', alignItems: 'center', gap: 0, padding: '8px 0' }}>
        {stepKeys.map((step, i) => {
          const record = steps[step]
          const isComplete = record && (record.status === 'completed' || record.status === 'succeeded')
          const isPending = !record || record.status === 'pending'
          const isRunning = record && ['running', 'ready', 'planned'].includes(record.status)
          return (
            <div key={step} style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 8,
                  flex: 1,
                  cursor: onStepClick ? 'pointer' : 'default',
                  opacity: isPending ? 0.4 : 1,
                }}
                onClick={() => onStepClick?.(step)}
              >
                <div
                  style={{
                    width: 48,
                    height: 48,
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: 20,
                    background: isComplete ? 'var(--success)' : isRunning ? 'var(--info)' : 'var(--bg-hover)',
                    border: `2px solid ${isComplete ? 'var(--success)' : isRunning ? 'var(--info)' : 'var(--border)'}`,
                  }}
                >
                  {isComplete ? '✓' : STEP_ICONS[step]}
                </div>
                <span style={{ fontSize: 13, fontWeight: 600 }}>{STEP_LABELS[step]}</span>
                {record && <StatusBadge status={record.status} />}
                {record?.remote_job?.duration_seconds != null && (
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    {record.remote_job.duration_seconds.toFixed(1)}s
                  </span>
                )}
              </div>
              {i < stepKeys.length - 1 && (
                <div
                  style={{
                    flex: 1,
                    height: 2,
                    background: isComplete ? 'var(--success)' : 'var(--border)',
                    minWidth: 40,
                    marginBottom: 32,
                  }}
                />
              )}
            </div>
          )
        })}
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: 'var(--text-muted)' }}>
        {completedCount}/{stepKeys.length} steps complete
      </div>
    </div>
  )
}
