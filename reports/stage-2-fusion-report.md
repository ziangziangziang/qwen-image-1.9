# Stage 2 Fusion Report

## Mission
Produce a BF16 research artifact that keeps 2512 as the realism anchor, injects edit behavior from 2511, and borrows layered prompt logic without pretending RGBA is just a mood.

## Edit delta recipe
- Foundation: `Qwen/Qwen-Image-2512`
- Delta source: `Qwen/Qwen-Image-Edit-2511`
- Delta base hint: `Qwen/Qwen-Image-Edit-2509`
- Blend weight: `0.35`
- Target subsystems: mmdit_backbone, edit_heads, conditioning_blocks

## Text encoder merge recipe
- Foundation: `Qwen/Qwen-Image-2512`
- Donor: `Qwen/Qwen-Image-Layered`
- Strategy: `ties`
- Conflict policy: `keep-foundation-on-shape-mismatch`

## Remote artifact target
- Output checkpoint: `/mnt/artifacts/qwen-image-1.9/stage-2/qwen-image-1.9-bf16.safetensors`

## Unresolved conflicts
- RGBA-VAE semantics remain out of line with the RGB foundation.
- Layer-specific RoPE must stay behind a compatibility gate until Stage 1 evidence improves.
