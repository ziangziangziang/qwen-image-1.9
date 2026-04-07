import { Link } from 'react-router-dom'
import { useRuns } from '../hooks/useRuns'
import StatusBadge from '../components/StatusBadge'
import { PIPELINE_STEPS } from '../api/types'
import type { PipelineStep } from '../api/types'

export default function RunList() {
  const { runs, loading, error } = useRuns()

  if (loading) return <div className="loading">Loading runs...</div>
  if (error) return <div className="error">Error: {error}</div>
  if (!runs?.length) {
    return (
      <div className="empty">
        <h3>No runs found</h3>
        <p>Run <code>q19 merge</code> to create your first run.</p>
      </div>
    )
  }

  return (
    <div>
      <div className="page-header">
        <h1>Qwen-Image 1.9 Dashboard</h1>
        <p>{runs.length} run{runs.length !== 1 ? 's' : ''} found</p>
      </div>
      <table>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Updated</th>
            {PIPELINE_STEPS.map((s) => <th key={s}>{s}</th>)}
            <th>Tags</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.run_id} className="clickable">
              <td>
                <Link to={`/app/runs/${run.run_id}`} style={{ fontWeight: 600 }}>
                  {run.run_id}
                </Link>
              </td>
              <td style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
                {new Date(run.updated_at).toLocaleString()}
              </td>
              {(PIPELINE_STEPS as unknown as PipelineStep[]).map((step) => {
                const record = run.steps[step]
                return (
                  <td key={step}>
                    {record ? <StatusBadge status={record.status} /> : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                  </td>
                )
              })}
              <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                {run.tags?.join(', ') || '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
