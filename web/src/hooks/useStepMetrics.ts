import { useAsync } from '../hooks/useAsync'
import { fetchMetrics } from '../api/client'
import type { MetricsResponse } from '../api/types'

export function useStepMetrics(runId: string, step: string) {
  const { data, loading, error } = useAsync<MetricsResponse>(
    () => fetchMetrics(runId, step),
    [runId, step],
  )
  return { metrics: data, loading, error }
}
