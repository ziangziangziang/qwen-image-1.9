import { useAsync } from '../hooks/useAsync'
import { fetchSamples } from '../api/client'
import type { SamplesResponse } from '../api/types'

export function useStepSamples(runId: string, step: string) {
  const { data, loading, error } = useAsync<SamplesResponse>(
    () => fetchSamples(runId, step),
    [runId, step],
  )
  return { samples: data, loading, error }
}
