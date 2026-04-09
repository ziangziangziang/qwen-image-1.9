import { useAsync } from '../hooks/useAsync'
import { fetchRun } from '../api/client'
import type { RunManifest } from '../api/types'

export function useRun(runId: string | undefined) {
  const { data, loading, error } = useAsync<RunManifest>(
    () => fetchRun(runId!),
    [runId],
  )
  return { manifest: data, loading, error }
}
