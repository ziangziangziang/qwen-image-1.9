import { useParams, Link } from 'react-router-dom'
import { useRun } from '../hooks/useRun'
import PipelineStepper from '../components/PipelineStepper'
import StatusBadge from '../components/StatusBadge'
import { PIPELINE_STEPS } from '../api/types'
import type { PipelineStep } from '../api/types'

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>()
  const { manifest, loading, error } = useRun(runId)

  if (loading) return <div className="loading">Loading run...</div>
  if (error) return <div className="error">Error: {error}</div>
  if (!manifest) return <div className="empty"><h3>Run not found</h3></div>

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1>{manifest.run_id}</h1>
            <p>Created {new Date(manifest.created_at).toLocaleString()} · Updated {new Date(manifest.updated_at).toLocaleString()}</p>
          </div>
          <Link to="/app" style={{ fontSize: 14 }}>← Back to runs</Link>
        </div>
      </div>

      <PipelineStepper
        steps={manifest.steps}
        onStepClick={(step) => {
          const record = manifest.steps[step]
          if (record && record.status !== 'pending') {
            window.location.href = `/app/runs/${runId}/steps/${step}`
          }
        }}
      />

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Source Models</h3>
        <table>
          <thead>
            <tr>
              <th>Alias</th>
              <th>Model ID</th>
              <th>Role</th>
              <th>Architecture</th>
            </tr>
          </thead>
          <tbody>
            {Object.values(manifest.source_models).map((model) => (
              <tr key={model.alias}>
                <td><code>{model.alias}</code></td>
                <td style={{ fontSize: 13 }}>{model.model_id}</td>
                <td><StatusBadge status={model.role} /></td>
                <td style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  {model.architecture?.backbone || '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3>Steps</h3>
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>Status</th>
              <th>Input Checkpoint</th>
              <th>Output Checkpoint</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            {(PIPELINE_STEPS as unknown as PipelineStep[]).map((step) => {
              const record = manifest.steps[step]
              if (!record) return null
              const isComplete = record.status === 'completed' || record.status === 'succeeded'
              return (
                <tr key={step} className={isComplete ? 'clickable' : ''} onClick={() => isComplete && (window.location.href = `/app/runs/${runId}/steps/${step}`)}>
                  <td><strong>{step}</strong></td>
                  <td><StatusBadge status={record.status} /></td>
                  <td style={{ fontSize: 12, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {record.input_checkpoint || '—'}
                  </td>
                  <td style={{ fontSize: 12, maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {record.output_checkpoint || '—'}
                  </td>
                  <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    {record.updated_at ? new Date(record.updated_at).toLocaleString() : '—'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {manifest.notes && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Notes</h3>
          <p style={{ color: 'var(--text-secondary)' }}>{manifest.notes}</p>
        </div>
      )}
    </div>
  )
}
