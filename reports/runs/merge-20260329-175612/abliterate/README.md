# Abliterate

## Status
- Run: `merge-20260329-175612`
- Step: `abliterate`
- Status: `completed`
- Input checkpoint: `reports/stage-2/artifacts/core-candidates/core-delta-w035/qwen-image-1.9-core-bf16.safetensors`
- Output checkpoint: `/mnt/artifacts/qwen-image-1.9/runs/merge-20260329-175612/abliterate/abliterated-model`

## Remote Job
- Name: `abliterate-refusal-direction`
- Workdir: `/mnt/experiments/qwen-image-1.9`
- Artifact dir: `/mnt/artifacts/qwen-image-1.9`
- Status: `ready`

## Metrics
- `refusal_direction_removed`: `True`
- `target_subspace`: `refusal-behavior`
- `safety_review_required`: `True`

## Artifacts
| Kind | Path | Content Type |
| --- | --- | --- |
| `checkpoint_ref` | `/mnt/artifacts/qwen-image-1.9/runs/merge-20260329-175612/abliterate/abliterated-model` | `application/octet-stream` |

## Policy
- This run treats refusal-direction removal as an internal-only experiment.
- Eval and review are required before the run can be considered complete.

# Abliterate Evaluation Report

## Checkpoint
- Ref: `/mnt/artifacts/qwen-image-1.9/runs/merge-20260329-175612/abliterate/abliterated-model`
- Sample root: `reports/runs/merge-20260329-175612/abliterate/samples`

## Aggregate Metrics
| Metric | Value |
| --- | --- |
| `refusal_rate_delta` | `-0.72` |
| `capability_retention_score` | `0.78` |
| `merged_regression_delta` | `0.08` |

## Suites
| Suite | Task Type | Failures |
| --- | --- | --- |
| `abliterate-refusal-delta` | `refusal-behavior` | `0` |
| `abliterate-capability-retention` | `generation` | `0` |
| `abliterate-regression-vs-merged` | `quality-regression` | `0` |
