import { useState } from 'react'
import { sampleUrl } from '../api/client'

interface Props {
  runId: string
  step: string
  samples: string[]
}

export default function ImageCompare({ runId, step, samples }: Props) {
  const imageFiles = samples.filter((s) => /\.(png|jpg|jpeg|gif|webp)$/i.test(s))
  const [selected, setSelected] = useState(0)

  if (!imageFiles.length) {
    return (
      <div className="card">
        <h3>Sample Images</h3>
        <div className="empty">
          <h3>No samples available</h3>
          <p>Run the pipeline with --execute to generate sample images.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="card">
      <h3>Sample Images ({imageFiles.length})</h3>
      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ width: 160, display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 400, overflowY: 'auto' }}>
          {imageFiles.map((file, i) => (
            <div
              key={file}
              onClick={() => setSelected(i)}
              style={{
                cursor: 'pointer',
                border: i === selected ? '2px solid var(--accent)' : '2px solid transparent',
                borderRadius: 'var(--radius)',
                overflow: 'hidden',
                opacity: i === selected ? 1 : 0.6,
              }}
            >
              <img
                src={sampleUrl(runId, step, file)}
                alt={file}
                style={{ width: '100%', display: 'block' }}
              />
              <div style={{ fontSize: 10, padding: '4px 6px', color: 'var(--text-muted)', background: 'var(--bg-secondary)' }}>
                {file}
              </div>
            </div>
          ))}
        </div>
        <div style={{ flex: 1 }}>
          <img
            src={sampleUrl(runId, step, imageFiles[selected])}
            alt={imageFiles[selected]}
            style={{ width: '100%', borderRadius: 'var(--radius)' }}
          />
          <div style={{ marginTop: 8, fontSize: 13, color: 'var(--text-secondary)' }}>
            {imageFiles[selected]}
          </div>
        </div>
      </div>
    </div>
  )
}
