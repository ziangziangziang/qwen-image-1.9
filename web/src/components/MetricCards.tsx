interface Props {
  metrics: Record<string, unknown>
}

const METRIC_COLORS: Record<string, string> = {
  generation_score: '#00b894',
  edit_score: '#74b9ff',
  donor_regression_delta: '#e17055',
  refusal_rate_delta: '#e17055',
  capability_retention_score: '#00b894',
  merged_regression_delta: '#fdcb6e',
  run_profile: '#a29bfe',
  candidate_count: '#74b9ff',
}

export default function MetricCards({ metrics }: Props) {
  const entries = Object.entries(metrics).filter(([, v]) => v !== null && v !== undefined)

  if (!entries.length) {
    return (
      <div className="card">
        <h3>Metrics</h3>
        <div className="empty">
          <h3>No metrics available</h3>
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <h3>Metrics</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 12 }}>
        {entries.map(([key, value]) => {
          const color = METRIC_COLORS[key] || 'var(--text-primary)'
          const displayValue = typeof value === 'number'
            ? Number.isInteger(value) ? value : value.toFixed(4)
            : String(value)
          return (
            <div
              key={key}
              style={{
                background: 'var(--bg-secondary)',
                borderRadius: 'var(--radius)',
                padding: 16,
                borderLeft: `3px solid ${color}`,
              }}
            >
              <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                {key.replace(/_/g, ' ')}
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, marginTop: 4, color }}>
                {displayValue}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
