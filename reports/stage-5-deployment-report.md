# Stage 5 Deployment Report

## Runtime
- Backend: `vllm-omni`
- Accelerator: `rocm`

## Stage graph
- `thinker`: text-encoder -> remote://bf16/ablated-text-encoder-placeholder
- `generator`: mmdit -> remote://bf16/merged-mmdit-placeholder

## Remote output
- Generated config: `/mnt/artifacts/qwen-image-1.9/stage-5/generated-stage-config.yaml`

## Known limitations
- This scaffold validates structure and orchestration only.
- Benchmark fields should be appended after remote execution.
