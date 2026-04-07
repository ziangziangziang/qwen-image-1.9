import type {
  EvalSummary,
  MetricsResponse,
  RunListItem,
  RunManifest,
  SamplesResponse,
} from './types'

const BASE = '/api'

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export async function fetchRuns(): Promise<RunListItem[]> {
  return fetchJson<RunListItem[]>('/runs')
}

export async function fetchRun(runId: string): Promise<RunManifest> {
  return fetchJson<RunManifest>(`/runs/${runId}`)
}

export async function fetchStepResult(runId: string, step: string): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(`/runs/${runId}/steps/${step}`)
}

export async function fetchEvalSummary(runId: string, step: string): Promise<EvalSummary> {
  return fetchJson<EvalSummary>(`/runs/${runId}/steps/${step}/eval-summary`)
}

export async function fetchMetrics(runId: string, step: string): Promise<MetricsResponse> {
  return fetchJson<MetricsResponse>(`/runs/${runId}/steps/${step}/metrics`)
}

export async function fetchSamples(runId: string, step: string): Promise<SamplesResponse> {
  return fetchJson<SamplesResponse>(`/runs/${runId}/steps/${step}/samples`)
}

export function sampleUrl(runId: string, step: string, filename: string): string {
  return `${BASE}/runs/${runId}/steps/${step}/samples/${filename}`
}

export async function fetchTrainingConfig(runId: string, step: string): Promise<Record<string, unknown>> {
  return fetchJson<Record<string, unknown>>(`/runs/${runId}/steps/${step}/training-config`)
}
