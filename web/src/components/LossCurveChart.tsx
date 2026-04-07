import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'
import type { MetricEntry } from '../api/types'

const COLORS = ['#6c5ce7', '#00b894', '#e17055', '#74b9ff', '#fdcb6e', '#a29bfe']

interface Props {
  metrics: MetricEntry[]
}

export default function LossCurveChart({ metrics }: Props) {
  if (!metrics.length) {
    return (
      <div className="card">
        <h3>Training Loss</h3>
        <div className="empty">
          <h3>No training metrics available</h3>
          <p>Run the pipeline with --execute to generate training data.</p>
        </div>
      </div>
    )
  }

  const datasets = metrics.filter((m) => m.loss_curve && m.loss_curve.length > 0)

  if (!datasets.length) {
    return (
      <div className="card">
        <h3>Training Loss</h3>
        <div className="empty">
          <h3>No loss curves found</h3>
          <p>Metrics files exist but contain no loss curve data.</p>
        </div>
      </div>
    )
  }

  const maxSteps = Math.max(...datasets.map((d) => d.loss_curve!.length))
  const data = Array.from({ length: maxSteps }, (_, i) => {
    const point: Record<string, number | string> = { step: i }
    datasets.forEach((d) => {
      if (d.loss_curve && i < d.loss_curve.length) {
        point[d.name] = d.loss_curve[i]
      }
    })
    return point
  })

  return (
    <div className="card">
      <h3>Training Loss</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="step"
            stroke="var(--text-muted)"
            tick={{ fontSize: 12 }}
            label={{ value: 'Step', position: 'insideBottom', offset: -4, fill: 'var(--text-muted)', fontSize: 12 }}
          />
          <YAxis
            stroke="var(--text-muted)"
            tick={{ fontSize: 12 }}
            label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 'var(--radius)',
              color: 'var(--text-primary)',
            }}
          />
          <Legend />
          {datasets.map((d, i) => (
            <Line
              key={d.name}
              type="monotone"
              dataKey={d.name}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              strokeWidth={2}
              name={d.name.replace(/-/g, ' ')}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', gap: 24, marginTop: 12, fontSize: 12, color: 'var(--text-secondary)' }}>
        {datasets.map((d) => (
          <span key={d.name}>
            <strong>{d.name}</strong>:{' '}
            final: {d.final_loss?.toFixed(6) ?? 'n/a'},{' '}
            min: {d.min_loss?.toFixed(6) ?? 'n/a'},{' '}
            steps: {d.max_steps ?? d.loss_curve?.length ?? 'n/a'}
          </span>
        ))}
      </div>
    </div>
  )
}
