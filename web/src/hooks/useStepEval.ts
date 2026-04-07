import { useAsync } from '../hooks/useAsync'
import { fetchEvalSummary } from '../api/client'
import type { EvalSummary } from '../api/types'

export function useStepEval(runId: string, step: string) {
  const { data, loading, error } = useAsync<EvalSummary>(
    () => fetchEvalSummary(runId, step),
    [runId, step],
  )
  return { evalSummary: data, loading, error }
}
