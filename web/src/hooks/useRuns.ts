import { useAsync } from '../hooks/useAsync'
import { fetchRuns } from '../api/client'
import type { RunListItem } from '../api/types'

export function useRuns() {
  const { data, loading, error } = useAsync<RunListItem[]>(() => fetchRuns(), [])
  return { runs: data, loading, error }
}
