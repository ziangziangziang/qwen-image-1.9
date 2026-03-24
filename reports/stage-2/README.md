# Stage 2 Fusion

## Mission
Stage 2 now builds two tracks from the Stage 1 evidence: a stable BF16 core based on `2512 + transformer-only edit delta`, and an experimental Layered bridge branch that learns RGB behavior from synthetic teacher data instead of pretending `vae` and `rope` mismatches will disappear.

## Run Mode
- Run mode: `write`
- Run profile: `smoke`
- Execution enabled: `True`
- Execution policy: `overwrite`
- Cleanup performed: `False`
- Resource profile: `num_gpus=2`, `vram_target_gb=160`
- Limits: `{"bridge_batch_size": 1, "bridge_train_steps": 64, "consistency_eval_prompt_count": 4, "core_candidate_id": "core-delta-w035", "dataset_samples_per_split": 2, "eval_edit_prompt_count": 3, "eval_prompt_count": 6, "poc_guidance_scale": 1.0, "poc_negative_prompt": "low resolution, low quality, deformed limbs, deformed fingers, oversaturated image, waxy skin, over-smoothed face, artificial look, chaotic composition, blurry text, distorted text", "poc_side": 512, "poc_true_cfg_scale": 4.0}`

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
| `core-delta-w035` | `0.35` | `selected` | `reports/stage-2/artifacts/core-candidates/core-delta-w035/qwen-image-1.9-core-bf16.safetensors` | `reports/stage-2/evals/core-candidates/core-delta-w035/smoke-summary.json` |

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
| `generation_teacher` | `Qwen/Qwen-Image-2512` | `text-to-image` | `2` | `reports/stage-2/datasets/teacher-db/generation_teacher` |
| `edit_teacher` | `Qwen/Qwen-Image-Edit-2511` | `generate-then-edit` | `2` | `reports/stage-2/datasets/teacher-db/edit_teacher` |
| `layered_teacher` | `Qwen/Qwen-Image-Layered` | `layer-aware-generation` | `2` | `reports/stage-2/datasets/teacher-db/layered_teacher` |

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
