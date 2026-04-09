# Stage 2 Fusion

## Mission
Stage 2 now builds two tracks from the Stage 1 evidence: a stable BF16 core based on `2512 + transformer-only edit delta`, and an experimental Layered bridge branch that learns RGB behavior from synthetic teacher data instead of pretending `vae` and `rope` mismatches will disappear.

## Run Mode
- Run mode: `write`
- Run profile: `full`
- Execution enabled: `True`
- Execution policy: `overwrite`
- Cleanup performed: `True`
- Resource profile: `num_gpus=2`, `vram_target_gb=160`
- Limits: `{"bridge_batch_size": 2, "bridge_train_steps": 1000, "consistency_eval_prompt_count": 16, "core_candidate_id": "core-delta-w035", "dataset_samples_per_split": 16, "eval_edit_prompt_count": 8, "eval_prompt_count": 24, "poc_guidance_scale": 1.0, "poc_negative_prompt": "low resolution, low quality, deformed limbs, deformed fingers, oversaturated image, waxy skin, over-smoothed face, artificial look, chaotic composition, blurry text, distorted text", "poc_side": 1024, "poc_steps": 30, "poc_true_cfg_scale": 4.0}`

## Stage 1 Evidence
- Foundation vs Edit transformer path is the one clean merge lane: `shared=2856`, `exact=0.3246`, `strategy=delta-merge`.
- Edit deltas cluster in late MMDiT blocks: `40, 42, 43, 44, 46, 47, 48, 49, 50, 57`.
- Layered text encoder is a no-op donor in practice: `exact=True`.
- Layered conflicts remain real: VAE `RGB -> RGBA`, rope `2D-or-rotary -> Layer3D`.

## No-Go List
- `text_encoder`: Stage 1 value analysis shows exact text-encoder parity, so Stage 2 does not spend merge budget on a no-op subsystem.
- `vae`: Layered keeps RGBA semantics while the core stack is RGB, so the VAE stays out of Stage 2 fusion.
- `rope`: Layer3D positional behavior does not directly align with the 2D foundation, so Stage 2 keeps rope changes behind the bridge experiment.

## Stable Core Track
- Foundation: `Qwen/Qwen-Image-2512`
- Delta source: `Qwen/Qwen-Image-Edit-2511`
- Delta base candidate: `Qwen/Qwen-Image`
- Target scope: `transformer` / `mmdit_backbone`
- Selection rule: `Prefer edit retention without visible generation regression on the smoke suite.`

| Candidate | Blend weight | Status | Planned checkpoint | Planned smoke report |
| --- | --- | --- | --- | --- |
| `core-delta-w020` | `0.2` | `candidate` | `reports/stage-2/artifacts/core-candidates/core-delta-w020/qwen-image-1.9-core-bf16.safetensors` | `reports/stage-2/evals/core-candidates/core-delta-w020/smoke-summary.json` |
| `core-delta-w030` | `0.3` | `candidate` | `reports/stage-2/artifacts/core-candidates/core-delta-w030/qwen-image-1.9-core-bf16.safetensors` | `reports/stage-2/evals/core-candidates/core-delta-w030/smoke-summary.json` |
| `core-delta-w035` | `0.35` | `selected` | `reports/stage-2/artifacts/core-candidates/core-delta-w035/qwen-image-1.9-core-bf16.safetensors` | `reports/stage-2/evals/core-candidates/core-delta-w035/smoke-summary.json` |
| `core-delta-w040` | `0.4` | `candidate` | `reports/stage-2/artifacts/core-candidates/core-delta-w040/qwen-image-1.9-core-bf16.safetensors` | `reports/stage-2/evals/core-candidates/core-delta-w040/smoke-summary.json` |

## Experimental Layered Bridge Track
- Donor: `Qwen/Qwen-Image-Layered`
- Strategy: `learnable-bridge`
- Base core candidate: `core-delta-w035`
- Bridge scope: `transformer_blocks.40:60`
- Extra parameter paths: transformer.time_text_embed.addition_t_embedding.weight
- Trainable modules: bridge_adapter, per_block_gates
- Freeze policy: text_encoder, vae, rope, transformer_blocks.0-39
- Distillation target: `rgb-output`
- Output adapter: `reports/stage-2/artifacts/experimental/layered-bridge-adapter.safetensors`
- Output checkpoint: `reports/stage-2/artifacts/experimental/qwen-image-1.9-layered-bridge-bf16.safetensors`

## Teacher Dataset
- Dataset manifest: `reports/stage-2/dataset-manifest.json`
- Output root: `reports/stage-2/datasets/teacher-db`
- Layered flattening: `alpha-composite-to-rgb`

| Split | Teacher model | Task | Planned samples | Asset root |
| --- | --- | --- | --- | --- |
| `generation_teacher` | `Qwen/Qwen-Image-2512` | `text-to-image` | `16` | `reports/stage-2/datasets/teacher-db/generation_teacher` |
| `edit_teacher` | `Qwen/Qwen-Image-Edit-2511` | `generate-then-edit` | `16` | `reports/stage-2/datasets/teacher-db/edit_teacher` |
| `layered_teacher` | `Qwen/Qwen-Image-Layered` | `layer-aware-generation` | `16` | `reports/stage-2/datasets/teacher-db/layered_teacher` |

## Remote Jobs
| Job | Status | Entry point | Workdir | Log |
| --- | --- | --- | --- | --- |
| `core_delta_sweep` | `planned` | `qwen_image_19.stage_2_fusion._worker_edit_delta` | `reports/stage-2/jobs/core-delta-sweep` | `reports/stage-2/logs/core-delta-sweep.log` |
| `core_smoke_eval` | `planned` | `qwen_image_19.stage_2_fusion._worker_bf16_compose` | `reports/stage-2/jobs/core-smoke-eval` | `reports/stage-2/logs/core-smoke-eval.log` |
| `teacher_dataset_generation` | `planned` | `qwen_image_19.stage_2_fusion._worker_teacher_dataset` | `reports/stage-2/jobs/teacher-dataset` | `reports/stage-2/logs/teacher-dataset.log` |
| `layered_bridge_train` | `planned` | `qwen_image_19.stage_2_fusion._worker_layered_bridge` | `reports/stage-2/jobs/layered-bridge-train` | `reports/stage-2/logs/layered-bridge-train.log` |
| `experimental_smoke_eval` | `planned` | `qwen_image_19.stage_2_fusion._worker_bf16_compose` | `reports/stage-2/jobs/experimental-smoke-eval` | `reports/stage-2/logs/experimental-smoke-eval.log` |
| `core_edit_eval` | `planned` | `qwen_image_19.stage_2_fusion._worker_bf16_compose` | `reports/stage-2/jobs/core-edit-eval` | `reports/stage-2/logs/core-edit-eval.log` |
| `consistency_eval` | `planned` | `qwen_image_19.stage_2_fusion._worker_bf16_compose` | `reports/stage-2/jobs/consistency-eval` | `reports/stage-2/logs/consistency-eval.log` |

## Artifacts
| Artifact | Path |
| --- | --- |
| `report_readme` | `reports/stage-2/README.md` |
| `merge_manifest` | `reports/stage-2/merge-manifest.json` |
| `dataset_manifest` | `reports/stage-2/dataset-manifest.json` |
| `training_report` | `reports/stage-2/training-report.md` |
| `training_figures_dir` | `reports/stage-2/figures` |
| `stable_core_checkpoint` | `reports/stage-2/artifacts/core-candidates/core-delta-w035/qwen-image-1.9-core-bf16.safetensors` |
| `experimental_bridge_adapter` | `reports/stage-2/artifacts/experimental/layered-bridge-adapter.safetensors` |
| `experimental_bridge_checkpoint` | `reports/stage-2/artifacts/experimental/qwen-image-1.9-layered-bridge-bf16.safetensors` |
| `artifact_dir` | `reports/stage-2` |
| `run_status` | `reports/stage-2/run-status.json` |

## Limitations
- Stage 2 does not attempt true RGBA decomposition support. Layered supervision is flattened back into RGB composites.
- The stable core winner is provisional until the remote coefficient sweep and smoke suite complete.
- The Layered branch is experimental and should be treated as a bridge adapter, not a drop-in replacement for the core checkpoint.

---

## Run Results

> Profile: `full` · Policy: `overwrite` · Total: `1h 42m 26s`

### Job Execution

| Job | Status | Duration (s) | Exit | Log |
| --- | --- | --- | --- | --- |
| `core_delta_sweep` | ✓ `succeeded` | `1184.1361` | `0` | `reports/stage-2/logs/core-delta-sweep.log` |
| `core_smoke_eval` | ✓ `succeeded` | `338.6642` | `0` | `reports/stage-2/logs/core-smoke-eval.log` |
| `teacher_dataset_generation` | ✓ `succeeded` | `2613.0654` | `0` | `reports/stage-2/logs/teacher-dataset.log` |
| `layered_bridge_train` | ✓ `succeeded` | `25.1272` | `0` | `reports/stage-2/logs/layered-bridge-train.log` |
| `experimental_smoke_eval` | ✓ `succeeded` | `334.4732` | `0` | `reports/stage-2/logs/experimental-smoke-eval.log` |
| `core_edit_eval` | ✓ `succeeded` | `617.5378` | `0` | `reports/stage-2/logs/core-edit-eval.log` |
| `consistency_eval` | ✓ `succeeded` | `1033.6228` | `0` | `reports/stage-2/logs/consistency-eval.log` |

### Training Report

[training-report.md](reports/stage-2/training-report.md)

### Visual Evaluation

#### Generation (core-delta merged model — text-to-image)

_No samples found at `/lustre_scratch/user_scratch/zziang/qwen-image-1.9/stage-2/evals/core-candidates/core-delta-w035/samples`._

#### Edit (before → after pairs)

_No edit samples found at `/lustre_scratch/user_scratch/zziang/qwen-image-1.9/stage-2/evals/core-edit/edit-samples`._

#### Experimental Layered Bridge (smoke eval)

_No samples found at `/lustre_scratch/user_scratch/zziang/qwen-image-1.9/stage-2/evals/experimental/samples`._

### Consistency Eval

- Prompts evaluated: `16`
- Mean pixel-L2 drift: `0.0`  _(lower = merged model stays close to foundation)_
- Min drift: `0.0` / Max drift: `0.0`
- Status: `passed`
