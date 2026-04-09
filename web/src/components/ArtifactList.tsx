import type { ArtifactRef } from '../api/types'

interface Props {
  artifacts: ArtifactRef[]
}

export default function ArtifactList({ artifacts }: Props) {
  if (!artifacts.length) {
    return null
  }

  return (
    <div className="card">
      <h3>Artifacts</h3>
      <table>
        <thead>
          <tr>
            <th>Kind</th>
            <th>Path</th>
            <th>Type</th>
          </tr>
        </thead>
        <tbody>
          {artifacts.map((a, i) => (
            <tr key={i}>
              <td><code>{a.kind}</code></td>
              <td><code style={{ fontSize: 12 }}>{a.path_or_uri}</code></td>
              <td style={{ fontSize: 12, color: 'var(--text-muted)' }}>{a.content_type}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
