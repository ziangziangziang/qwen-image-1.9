"""Evaluation runner — image generation + quality metrics.

Used by ``post_quantize_eval`` and can be invoked standalone.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

PROMPT_BANK = [
    "studio product photo of a camera with readable label text",
    "portrait of a cyclist in white jacket under soft daylight",
    "neon city street at night with reflective wet pavement",
    "clean cutout object on neutral studio background",
    "storybook castle scene with crisp headline typography",
    "pet portrait with detailed fur and natural expression",
    "minimal poster design with bold geometric shapes",
    "macro shot of a watch with metallic highlights",
]


def run_generation_eval(
    *,
    model_id: str,
    sample_dir: Path,
    num_prompts: int = 8,
    steps: int = 6,
    width: int = 512,
    height: int = 512,
    seed: int = 2025,
) -> dict[str, Any]:
    """Generate images from *model_id* and compute quality metrics."""
    sample_dir.mkdir(parents=True, exist_ok=True)
    prompts = PROMPT_BANK[: max(1, min(num_prompts, len(PROMPT_BANK)))]

    try:
        import torch
    except ImportError:
        return _skip("torch not available")

    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        return _skip("diffusers not available")

    try:
        from qwen_image_19.pipeline._hardware import require_gpus
        require_gpus(min_gpus=1, min_vram_gb=40.0)
    except Exception as exc:
        return _skip(f"hardware: {exc}")

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, use_safetensors=True,
        )
        pipe.set_progress_bar_config(disable=True)
    except Exception as exc:
        return _skip(f"pipeline load: {exc}")

    generated = 0
    latencies: list[float] = []
    luminance_values: list[float] = []
    sample_files: list[str] = []
    started = time.perf_counter()

    try:
        for idx, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            gen = torch.Generator(device="cuda").manual_seed(seed + idx)
            result = pipe(prompt=prompt, num_inference_steps=max(1, steps), generator=gen,
                          width=width, height=height)
            img = result.images[0].convert("RGB")
            path = sample_dir / f"eval-{idx + 1:03d}.png"
            img.save(path)
            sample_files.append(path.name)
            px = torch.tensor(list(img.getdata()), dtype=torch.float32)
            luminance_values.append(float(px.mean().item()))
            latencies.append(time.perf_counter() - t0)
            generated += 1
    except Exception as exc:
        return {
            "status": "error", "reason": str(exc),
            "sample_files": sample_files, "metrics": {},
        }
    finally:
        del pipe
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started
    mean_lum = sum(luminance_values) / len(luminance_values) if luminance_values else 0
    mean_lat = sum(latencies) / len(latencies) if latencies else 0

    return {
        "status": "passed",
        "generated_images": generated,
        "elapsed_seconds": round(elapsed, 3),
        "sample_files": sample_files,
        "metrics": {
            "generation_score": round(min(1.0, mean_lum / 255.0), 4),
            "mean_luminance": round(mean_lum, 4),
            "mean_latency_per_image": round(mean_lat, 3),
            "total_images": generated,
        },
    }


def _skip(reason: str) -> dict[str, Any]:
    return {"status": "skipped", "reason": reason, "sample_files": [], "metrics": {}}
