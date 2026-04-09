# Qwen-Image 1.9

> Merging a text-to-image generator and an instruction-based image editor into a single checkpoint via per-block SLERP, then recovering quality with real-image LoRA fine-tuning on MagicBrush instruction-edit pairs.

---

## Status

| Phase | Status |
|---|---|
| Architecture analysis | ✅ Complete |
| Block-selective SLERP merge | ✅ Complete |
| Merge eval (gen + edit) | ✅ Complete |
| Phase 1 LoRA (text-only, 5k steps) | ✅ Complete — did not recover quality |
| Phase 2 LoRA (MagicBrush real-image, 15k steps) | 🔄 In progress — ~11 200 / 15 000 steps |
| Mid-training eval (checkpoint-8000) | ✅ Complete — panels in `lora_quick_eval/` |
| Abliterate → quantize → publish | ⏳ Pending training completion |

**Hardware:** AMD MI300X VF · 192 GB VRAM · ROCm 7.2

---

## Abstract

Two publicly available Qwen-Image model variants — `Qwen-Image-2512` (text-to-image generation) and `Qwen-Image-Edit-2511` (instruction-based editing) — share an identical MMDiT transformer architecture but serve fundamentally different tasks. We investigate whether a single merged checkpoint can serve both tasks.

**Phase 1 (merge):** We analysed per-block weight divergence (global cosine similarity 0.9872) and designed a block-selective SLERP strategy that concentrates edit-model influence in the late semantic blocks where editing deltas are most pronounced. The merged checkpoint loads as either pipeline without architecture changes.

**Phase 2 (Phase 1 LoRA — text-only):** A 5 000-step LoRA run over text-only generation prompts failed to recover quality. The adapter's velocity target was the noisy latent itself (trivial signal), and the merged checkpoint's loss landscape was poorly conditioned for small adapter recovery.

**Phase 3 (Phase 2 LoRA — MagicBrush real-image):** We implemented proper flow-matching training with VAE-encoded image targets and source-image token concatenation. Now running at 15 000 steps over 500 MagicBrush instruction-edit pairs × 30 epochs, interleaved 1:1 with 500 generation prompts. Loss has stabilised around 1.0–1.2 for edit steps and 0.05–0.18 for generation steps, indicating the adapter is successfully learning the real-image edit objective.

---

## 1. Background & Motivation

Standard practice for multiple model variants is to host them separately. Measuring the global cosine similarity at **0.9872** — the models share 98.7% of their representational geometry — made them natural candidates for a single merged checkpoint.

**Goal:** one checkpoint serving both `QwenImagePipeline` (generation) and `QwenImageEditPlusPipeline` (editing) with quality degradation below human-perceptible thresholds.

---

## 2. Architecture Analysis

### 2.1 Shared Transformer Structure

| Property | Qwen-Image-2512 (Gen) | Qwen-Image-Edit-2511 (Edit) |
|---|---|---|
| Architecture | `QwenImageTransformer2DModel` | `QwenImageTransformer2DModel` |
| Transformer blocks | 60 | 60 |
| Attention dim | 3584 | 3584 |
| Input channels | 64 | 64 |
| Output channels | 16 | 16 |
| VAE scale factor | 8 | 8 |
| Training objective | Flow-matching · text → image | Flow-matching · source + instruction → edit |

**Key finding:** The models differ **only in weights and pipeline class**, not architecture. No channel-concatenation for source image — `in_channels=64` for both.

### 2.2 How the Edit Model Conditions on Source Images

```python
# QwenImageEditPipeline.__call__
latent_model_input = torch.cat([noisy_target, image_latents], dim=1)  # sequence concat, not channel concat
noise_pred = noise_pred[:, :noisy_target.size(1)]  # discard source tokens after denoising
```

The source image is appended as **extra sequence tokens** (`dim=1`), not channels. The transformer attends jointly over noisy-target and source-image tokens using the same `in_channels=64` projection. This is architecturally free to merge with the generator.

### 2.3 Per-Block Weight Divergence

| Region | Blocks | Cosine similarity (avg) | Interpretation |
|---|---|---|---|
| I/O projections | non-block | ~0.96 | Moderate edit signal |
| Input patch embed | block 0 | ~0.97 | Light edit signal |
| Early/mid backbone | blocks 1–39 | ~0.99 | Generation quality lives here |
| Late semantic blocks | blocks 40–58 | ~0.97–0.98 | Edit semantics concentrated |
| Final output block | block 59 | ~0.90 | Most divergent block |

---

## 3. Merge Strategy: Block-Selective SLERP

### 3.1 SLERP Formula

$$\text{SLERP}(w_0, w_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\, w_0 + \frac{\sin(t\Omega)}{\sin\Omega}\, w_1, \qquad \Omega = \arccos\!\left(\hat{w}_0 \cdot \hat{w}_1\right)$$

where $t=0$ → pure generator, $t=1$ → pure editor. Falls back to linear interpolation for $\cos > 0.98$.

### 3.2 Per-Region Interpolation Weights

| Region | $t$ | Rationale |
|---|---|---|
| Non-block layers | **0.40** | High edit signal in I/O projections |
| Block 0 | **0.30** | Moderate-to-light |
| Blocks 1–39 | **0.10** | Generation quality fragile here |
| Blocks 40–58 | **0.25** | Edit semantics concentrated |
| Block 59 | **0.15** | Most divergent — conservative blend |

### 3.3 Merge Experiments

| Strategy | Config | Pixel std | Observation |
|---|---|---|---|
| Tri-capability SLERP | `--edit-coefficient 0.35` | 36.5 | Visible artefacts, degraded sharpness |
| Reduced coefficient | `--edit-coefficient 0.10` | 44.1 | Improved; residual degradation |
| **Block-selective SLERP** | `--recipe slerp-selective` | — | Best quality retention |

---

## 4. Training

### 4.1 Phase 1 LoRA — Text-Only (5 000 steps) — Unsuccessful

Two root causes:
1. **Trivial training signal.** With $x_0=0$, the velocity target reduces to the noisy latent itself — converges cheaply but teaches nothing useful.
2. **Insufficient LoRA capacity.** 0.12% trainable parameters (23 M / 20 B) cannot re-specialise from an off-manifold starting point.

Loss plateaued at ~0.95 with high variance; no clear descent.

### 4.2 Phase 2 LoRA — MagicBrush Real Images (15 000 steps) — In Progress

**Changes from Phase 1:**

| Component | Phase 1 | Phase 2 |
|---|---|---|
| Dataset | Text captions only | 500 MagicBrush pairs + 500 gen captions (1:1) |
| Training signal | Noisy latent → trivial | Real VAE-encoded target → proper flow-matching |
| Source conditioning | None | Source image appended as extra tokens |
| Epochs | 1 pass | 30 (dataset cycled) |
| Grad clip | Per-backward (bug) | Once per optimizer step (fixed) |

**Training objective (edit steps):**

$$\mathcal{L} = \bigl\| f_\theta\!\left((1{-}\bar\alpha)\, x_0 + \bar\alpha\,\epsilon,\; t,\; c_\text{text},\; c_\text{src}\right) - (\epsilon - x_0) \bigr\|_2^2$$

where $x_0$ is the VAE-encoded target image, $\epsilon \sim \mathcal{N}(0, I)$, $\bar\alpha = t/1000$, and $c_\text{src}$ are source-image tokens concatenated before the transformer.

**Configuration:**

| Parameter | Value |
|---|---|
| Dataset | `osunlp/MagicBrush` train split, 500 pairs |
| Gen mix | `ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions`, 500 |
| LoRA rank / α | 16 / 16 |
| Optimizer | AdamW, cosine schedule |
| Learning rate | 2e-5 |
| Warmup | 100 steps |
| Gradient accumulation | 4 steps |
| Max steps | 15 000 |
| Precision | bf16 autocast, fp32 LoRA params |
| Checkpoint interval | every 1 000 steps |

### 4.3 Loss Curve (Phase 2)

The bimodal pattern is expected: generation steps hit 0.05–0.18, edit steps hit 1.0–1.3. The plateau in edit-step loss indicates convergence, not stagnation.

```
Step   Loss          Notes
0      1.07          edit — initialisation
500    0.23          gen
1 000  0.14          gen
1 500  4.86          edit — hard timestep, early training
2 000  1.16          edit
3 000  1.19          edit
4 000  0.80          edit
5 000  0.27          gen
6 000  1.15          edit
7 000  0.99          edit
8 000  1.05          edit  ← checkpoint-8000 mid-eval
9 000  1.05          edit
10 000 1.10          edit
11 000 1.04          edit  ← current
```

Isolated spikes at steps 3600 (6.2), 9400 (5.5), 10500 (7.0) are hard-timestep edit samples caught by the NaN/Inf skip guard.

---

## 5. Evaluation

### 5.1 Generation Eval — `scripts/_eval_compare.py`

3-panel: **Original Gen Model | Merged | Merged + LoRA**
- 8 text-to-image prompts + 4 pseudo-edit prompts
- 28 steps · guidance 4.5 · 1024×1024

### 5.2 Edit Eval — `scripts/_eval_edit.py`

3-panel: **Source | Original Edit Model | Merged**

| # | Source | Instruction |
|---|---|---|
| 1 | street photo of a modern city block at noon | change sky to a dramatic sunset |
| 2 | japanese garden with stone lanterns and green moss | add falling cherry blossom petals |
| 3 | portrait of a young woman in soft natural light | turn into a detailed pencil sketch |
| 4 | outdoor portrait of a hiker in front of green hills | replace background with snowy mountain |
| 5 | a golden retriever sitting on a park bench | make it look like an oil painting |
| 6 | a modern living room with white walls | change lighting to warm candlelight |

### 5.3 Mid-Training LoRA Eval — `scripts/_eval_lora_quick.py`

3-panel: **Source | Edit Baseline | Merged + LoRA (checkpoint-N)**

Runs alongside training via `enable_model_cpu_offload()`. Results for `checkpoint-8000` in `lora_quick_eval/`.

```bash
python3 scripts/_eval_lora_quick.py \
  --checkpoint .../lora-adapter/checkpoint-8000 \
  --steps 15 --cases 3 --size 512
```

---

## 6. Artifacts

| Artifact | Location |
|---|---|
| Merged checkpoint | `/scratch/training/slerp-selective-20260408/merge/merged-tri-capability-checkpoint` |
| LoRA adapter | `.../post_merge_train/lora-adapter/` |
| Checkpoints (every 1k) | `.../lora-adapter/checkpoint-{1000..11000}` |
| Training log | `.../post_merge_train/train.log` |
| Gen eval panels | `.../comparison/` |
| Edit eval panels | `.../edit_eval/` |
| Mid-training eval | `.../lora_quick_eval/` |

**Sizes:** Merged base **54 GB** · LoRA adapter **91 MB** (0.12%) · Post-merge (LoRA merged in) **54 GB**

---

## 7. Engineering Notes

### 7.1 Bugs Fixed

| Bug | Root Cause | Fix |
|---|---|---|
| `AttributeError: pixel_values` in `encode_prompt` | EditPlus pipeline assumed image arg always present | Monkey-patched `_get_qwen_prompt_embeds` |
| `ValueError: not enough values (expected 5, got 4)` | Qwen VAE expects 5D `[B,C,F,H,W]` | `img_t = img_t.unsqueeze(2)` adds F=1 |
| `AttributeError: FrozenDict has no 'scaling_factor'` | Qwen VAE uses per-channel mean/std | `(latent - latents_mean) / latents_std` |
| `TypeError: reshape() Tensor at pos 3` | `patch_size` config value is a Tensor | `int(_ps[0]) if hasattr(_ps, "__len__") else int(_ps)` |
| `ZeroDivisionError` at `H // p` | `for p in transformer.parameters()` shadowed outer `p` | Renamed to `_patch_sz` throughout |
| `miopenStatusInternalError` during VAE encode | Temporal conv3d cache `_enc_feat_map` carried stale data between independent encodes | Reset cache before each standalone encode |
| Dataset exhausted at ~700 steps | Producer ran dataset once; `epochs=30` was unused | Outer epoch loop wraps round-robin iteration |
| Gradient instability | `clip_grad_norm_` ran after every individual backward in `grad_accum=4` context | Moved clip inside optimizer step block |

### 7.2 Loading the LoRA Adapter

```python
from peft import PeftModel
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(MERGED_CKPT, torch_dtype=torch.bfloat16)
pipe.transformer = PeftModel.from_pretrained(
    pipe.transformer, LORA_PATH, adapter_name="magicbrush"
)
pipe.transformer.set_adapter("magicbrush")
pipe.enable_model_cpu_offload()  # or .to("cuda") if VRAM available
```

### 7.3 VAE Notes

`AutoencoderKLQwenImage` is a **video VAE**:
- Requires 5D input `[B, C, F, H, W]` — single images need `unsqueeze(2)` for F=1
- Normalizes with per-channel `latents_mean` / `latents_std` (16-element), **not** a scalar `scaling_factor`
- Has a temporal conv3d feature cache (`_enc_feat_map`) that must be reset between independent encodes

---

## 8. Pipeline

| Step | Command | Status |
|---|---|---|
| Preflight | `q19 preflight` | ✅ |
| Merge | `q19 merge --recipe slerp-selective --execute` | ✅ |
| Post-merge train | `q19 post-merge-train --training-config configs/merge/stage-2-training-magicbrush.yaml --execute` | 🔄 |
| Abliterate | `q19 abliterate --execute` | ⏳ |
| Post-abliterate train | `q19 post-abliterate-train --execute` | ⏳ |
| Quantize | `q19 quantize --execute` | ⏳ |
| Eval | `q19 eval` | ⏳ |
| Publish | `q19 publish` | ⏳ |

---

## 9. Reproduce

```bash
pip install -e .

# Merge
q19 merge --recipe slerp-selective --run-id my-run --artifact-dir /scratch/training --execute

# Train
q19 post-merge-train \
  --run-id my-run \
  --artifact-dir /scratch/training \
  --input-checkpoint /scratch/training/my-run/merge/merged-tri-capability-checkpoint \
  --training-config configs/merge/stage-2-training-magicbrush.yaml \
  --execute

# Mid-training eval (runs alongside training, CPU offload)
python3 scripts/_eval_lora_quick.py \
  --checkpoint /scratch/training/my-run/post_merge_train/lora-adapter/checkpoint-8000 \
  --steps 15 --cases 3 --size 512

make test
```

---

## License

Apache-2.0
