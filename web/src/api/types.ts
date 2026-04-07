export interface SourceModel {
  alias: string
  model_id: string
  role: string
  architecture?: {
    backbone?: string
    text_encoder?: string
    parameters?: string
  }
}

export interface ArtifactRef {
  kind: string
  path_or_uri: string
  content_type: string
  size_bytes?: number | null
  label?: string | null
}

export interface RemoteJob {
  name: string
  workdir: string
  artifact_dir: string
  status: string
  duration_seconds?: number
  started_at?: string
  ended_at?: string
  log_path?: string
  execution_manifest?: string
}

export interface StepRecord {
  step: string
  status: string
  input_checkpoint: string | null
  output_checkpoint: string | null
  command: string[]
  remote_job: RemoteJob
  artifacts: ArtifactRef[]
  metrics: Record<string, unknown>
  eval_summary: string | null
  step_result: string | null
  report: string | null
  updated_at: string | null
}

export interface RunManifest {
  run_id: string
  created_at: string
  updated_at: string
  artifact_root: string
  source_models: Record<string, SourceModel>
  steps: Record<string, StepRecord>
  report_index: string
  tags: string[]
  notes: string
}

export interface RunListItem {
  run_id: string
  updated_at: string
  report_index: string
  steps: Record<string, StepRecord>
  tags: string[]
}

export interface EvalSuite {
  eval_suite_id: string
  checkpoint_ref: string
  task_type: string
  samples: ArtifactRef[]
  aggregate_metrics: Record<string, unknown>
  failures: string[]
  judge: {
    framework: string
    version: string
  }
}

export interface EvalSummary {
  run_id: string
  step: string
  checkpoint_ref: string
  suites: EvalSuite[]
  aggregate_metrics: Record<string, unknown>
  sample_root: string
}

export interface MetricEntry {
  name: string
  loss_curve?: number[]
  final_loss?: number
  min_loss?: number
  max_loss?: number
  max_steps?: number
  batch_size?: number
  learning_rate?: number
  seed?: number
  run_started_at?: string
  run_ended_at?: string
  elapsed_seconds?: number
  status?: string
  training_method?: Record<string, unknown>
  hardware?: Record<string, unknown>
  structure?: {
    layers?: string[]
  }
  [key: string]: unknown
}

export interface MetricsResponse {
  run_id: string
  step: string
  metrics: MetricEntry[]
}

export interface SamplesResponse {
  run_id: string
  step: string
  samples: string[]
}

export const PIPELINE_STEPS = ['merge', 'train', 'abliterate', 'quantize'] as const
export type PipelineStep = (typeof PIPELINE_STEPS)[number]
