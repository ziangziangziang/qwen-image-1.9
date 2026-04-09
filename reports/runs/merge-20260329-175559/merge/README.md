# Merge

## Status
- Run: `merge-20260329-175559`
- Step: `merge`
- Status: `completed`
- Input checkpoint: `Qwen/Qwen-Image-2512`
- Output checkpoint: `reports/stage-2/artifacts/core-candidates/core-delta-w035/qwen-image-1.9-core-bf16.safetensors`

## Remote Job
- Name: `merge-fusion`
- Workdir: `/mnt/experiments/qwen-image-1.9`
- Artifact dir: `/mnt/artifacts/qwen-image-1.9`
- Status: `ready`

## Metrics
- `run_profile`: `full`
- `execution_enabled`: `False`
- `candidate_count`: `4`

## Artifacts
| Kind | Path | Content Type |
| --- | --- | --- |
| `manifest` | `reports/runs/merge-20260329-175559/merge/artifacts/merge-manifest.json` | `application/json` |
| `dataset_manifest` | `reports/runs/merge-20260329-175559/merge/artifacts/dataset-manifest.json` | `application/json` |
| `report` | `reports/runs/merge-20260329-175559/merge/artifacts/README.md` | `text/markdown` |

## Fusion Plan
- Run profile: `full`
- Execution enabled: `False`
- Selected checkpoint: `reports/stage-2/artifacts/core-candidates/core-delta-w035/qwen-image-1.9-core-bf16.safetensors`

## Dataset
- Output root: `reports/stage-2/datasets/teacher-db`
- Splits: `generation_teacher, edit_teacher, layered_teacher`

# Merge Evaluation Report

## Checkpoint
- Ref: `reports/stage-2/artifacts/core-candidates/core-delta-w035/qwen-image-1.9-core-bf16.safetensors`
- Sample root: `reports/runs/merge-20260329-175559/merge/samples`

## Aggregate Metrics
| Metric | Value |
| --- | --- |
| `generation_score` | `0.84` |
| `edit_score` | `0.81` |
| `donor_regression_delta` | `0.06` |

## Suites
| Suite | Task Type | Failures |
| --- | --- | --- |
| `merge-generation-fidelity` | `generation` | `0` |
| `merge-edit-fidelity` | `edit` | `0` |
| `merge-regression-vs-donors` | `quality-regression` | `0` |
