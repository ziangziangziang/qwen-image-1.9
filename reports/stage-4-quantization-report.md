# Stage 4 Quantization Report

## Purpose
Compress the BF16 artifact without casually deleting vision quality.

## GGUF path
- Targets: Q4_K_M, IQ4_XS
- IMatrix dataset: `stage-4-diverse-image-set`
- Minimum images: `2000`

## EXL2/GPTQ path
- Targets: 4.0bpw
- Runtime: `vllm-omni` on `ROCm`

## Remote execution
- Artifact dir: `/mnt/artifacts/qwen-image-1.9`
- Cache dir: `/mnt/cache/qwen-image`
