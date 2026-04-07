import { useParams, Link } from 'react-router-dom'
import { useRun } from '../hooks/useRun'
import { useStepEval } from '../hooks/useStepEval'
import { useStepMetrics } from '../hooks/useStepMetrics'
import { useStepSamples } from '../hooks/useStepSamples'
import PipelineStepper from '../components/PipelineStepper'
import StatusBadge from '../components/StatusBadge'
import LossCurveChart from '../components/LossCurveChart'
import ImageCompare from '../components/ImageCompare'
import MetricCards from '../components/MetricCards'
import ArtifactList from '../components/ArtifactList'

export default function StepDetail() {
  const { runId, step } = useParams<{ runId: string; step: string }>()
  const { manifest, loading: manifestLoading, error: manifestError } = useRun(runId)
  const { evalSummary } = useStepEval(runId!, step!)
  const { metrics, loading: metricsLoading } = useStepMetrics(runId!, step!)
  const { samples, loading: samplesLoading } = useStepSamples(runId!, step!)

  if (manifestLoading) return <div className="loading">Loading...</div>
  if (manifestError) return <div className="error">Error: {manifestError}</div>
  if (!manifest) return <div className="empty"><h3>Run not found</h3></div>

  const stepRecord = manifest.steps[step!]
  if (!stepRecord) return <div className="empty"><h3>Step not found</h3></div>

  const isComplete = stepRecord.status === 'completed' || stepRecord.status === 'succeeded'

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <h1>{manifest.run_id} / {step}</h1>
            <p><StatusBadge status={stepRecord.status} /></p>
          </div>
          <Link to={`/app/runs/${runId}`} style={{ fontSize: 14 }}>← Back to run</Link>
        </div>
      </div>

      <PipelineStepper steps={manifest.steps} />

      {!isComplete && (
        <div className="card" style={{ marginTop: 16 }}>
          <h3>Step Not Complete</h3>
          <p style={{ color: 'var(--text-secondary)' }}>
            This step is in <code>{stepRecord.status}</code> state. Detailed metrics and samples are available after execution completes.
          </p>
        </div>
      )}

      <MetricCards metrics={stepRecord.metrics} />

      {isComplete && (
        <>
          <div style={{ marginTop: 16 }}>
            <LossCurveChart metrics={metricsLoading || !metrics ? [] : metrics.metrics} />
          </div>

          <div style={{ marginTop: 16 }}>
            <ImageCompare
              runId={runId!}
              step={step!}
              samples={samplesLoading || !samples ? [] : samples.samples}
            />
          </div>

          <div style={{ marginTop: 16 }}>
            <ArtifactList artifacts={stepRecord.artifacts} />
          </div>

          {evalSummary && evalSummary.suites.length > 0 && (
            <div className="card" style={{ marginTop: 16 }}>
              <h3>Evaluation Suites</h3>
              <table>
                <thead>
                  <tr>
                    <th>Suite</th>
                    <th>Task</th>
                    <th>Samples</th>
                    <th>Metrics</th>
                  </tr>
                </thead>
                <tbody>
                  {evalSummary.suites.map((suite) => (
                    <tr key={suite.eval_suite_id}>
                      <td><code style={{ fontSize: 12 }}>{suite.eval_suite_id}</code></td>
                      <td><StatusBadge status={suite.task_type} /></td>
                      <td>{suite.samples.length}</td>
                      <td style={{ fontSize: 12 }}>
                        {Object.entries(suite.aggregate_metrics).map(([k, v]) => (
                          <span key={k} style={{ marginRight: 12 }}>
                            <strong>{k}:</strong> {typeof v === 'number' ? v.toFixed(4) : String(v)}
                          </span>
                        ))}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <div className="card" style={{ marginTop: 16 }}>
            <h3>Execution Details</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, fontSize: 13 }}>
              <div>
                <div style={{ color: 'var(--text-muted)', fontSize: 11, textTransform: 'uppercase', marginBottom: 4 }}>Command</div>
                <code style={{ fontSize: 12 }}>{stepRecord.command.join(' ')}</code>
              </div>
              <div>
                <div style={{ color: 'var(--text-muted)', fontSize: 11, textTransform: 'uppercase', marginBottom: 4 }}>Remote Job</div>
                <div>{stepRecord.remote_job.name || '—'}</div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  {stepRecord.remote_job.workdir || '—'}
                </div>
              </div>
              <div>
                <div style={{ color: 'var(--text-muted)', fontSize: 11, textTransform: 'uppercase', marginBottom: 4 }}>Input Checkpoint</div>
                <code style={{ fontSize: 11, wordBreak: 'break-all' }}>{stepRecord.input_checkpoint || '—'}</code>
              </div>
              <div>
                <div style={{ color: 'var(--text-muted)', fontSize: 11, textTransform: 'uppercase', marginBottom: 4 }}>Output Checkpoint</div>
                <code style={{ fontSize: 11, wordBreak: 'break-all' }}>{stepRecord.output_checkpoint || '—'}</code>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
