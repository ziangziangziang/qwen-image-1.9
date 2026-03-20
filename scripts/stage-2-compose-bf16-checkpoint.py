#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import time

import torch

from qwen_image_19.stage_2_fusion import fuse
from qwen_image_19.stage_2_fusion.runtime import (
    Stage2HardwareError,
    resolve_stage2_diffusion_runtime,
)

try:
    from diffusers import DiffusionPipeline
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "diffusers is required for true smoke evaluation execution."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--task", default="core-smoke")
    parser.add_argument("--model-ref")
    parser.add_argument("--model-id")
    parser.add_argument("--output", help="Relative eval summary output path.")
    parser.add_argument("--num-prompts", type=int, default=6)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--required-gpus", type=int, default=2)
    parser.add_argument("--required-total-vram-gb", type=float, default=160.0)
    return parser.parse_args()


def add_supported_call_args(pipe, call_kwargs: dict[str, object], optional_kwargs: dict[str, object]) -> None:
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):  # pragma: no cover
        params = {}
    accepts_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params.values())
    for key, value in optional_kwargs.items():
        if value is None:
            continue
        if accepts_var_kwargs or key in params:
            call_kwargs[key] = value


if __name__ == "__main__":
    args = parse_args()
    if not args.execute:
        print(json.dumps(fuse(dry_run=True), indent=2))
        raise SystemExit(0)
    if not args.output:
        raise SystemExit("--output is required with --execute")
    if not args.model_id:
        raise SystemExit("--model-id is required with --execute")
    try:
        runtime = resolve_stage2_diffusion_runtime(
            required_gpus=args.required_gpus,
            required_total_vram_gb=args.required_total_vram_gb,
        )
    except Stage2HardwareError as exc:
        raise SystemExit(str(exc)) from exc

    prompts = [
        "studio product photo of a camera with readable label text",
        "portrait of a cyclist in white jacket under soft daylight",
        "neon city street at night with reflective wet pavement",
        "clean cutout object on neutral studio background",
        "storybook castle scene with crisp headline typography",
        "pet portrait with detailed fur and natural expression",
        "minimal poster design with bold geometric shapes",
        "macro shot of a watch with metallic highlights",
    ]
    prompt_count = max(1, min(args.num_prompts, len(prompts)))
    selected_prompts = prompts[:prompt_count]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_dir = output.parent / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipe = DiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            **runtime.pipeline_load_kwargs,
        )
        pipe.set_progress_bar_config(disable=True)
        generated = 0
        per_prompt_seconds: list[float] = []
        luminance_values: list[float] = []
        started = time.perf_counter()
        for idx, prompt in enumerate(selected_prompts):
            single_start = time.perf_counter()
            generator = torch.Generator(device=runtime.primary_device).manual_seed(args.seed + idx)
            call_kwargs = {
                "prompt": prompt,
                "num_inference_steps": max(1, args.steps),
                "width": args.width,
                "height": args.height,
                "generator": generator,
            }
            add_supported_call_args(
                pipe,
                call_kwargs,
                {
                    "negative_prompt": args.negative_prompt,
                    "true_cfg_scale": args.true_cfg_scale,
                    "guidance_scale": args.guidance_scale,
                },
            )
            result = pipe(
                **call_kwargs,
            )
            image = result.images[0].convert("RGB")
            image_path = sample_dir / f"{args.task}-{idx + 1:03d}.png"
            image.save(image_path)
            pixels = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image.height, image.width, 3)
            luminance_values.append(float(pixels.mean().item()))
            per_prompt_seconds.append(time.perf_counter() - single_start)
            generated += 1

        total_seconds = time.perf_counter() - started
        del pipe
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError as exc:
        raise SystemExit(
            "Stage 2 diffusion OOM during core/experimental smoke eval generation. "
            f"{runtime.summary()}. No fallback/offload retry is configured."
        ) from exc

    model_ref_meta = None
    if args.model_ref:
        meta_path = Path(args.model_ref).with_suffix(Path(args.model_ref).suffix + ".meta.json")
        if meta_path.exists():
            model_ref_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    output = Path(args.output)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "model_ref": args.model_ref,
        "model_id": args.model_id,
        "num_prompts": args.num_prompts,
        "generated_images": generated,
        "steps": args.steps,
        "resolution": f"{args.width}x{args.height}",
        "elapsed_seconds": round(total_seconds, 3),
        "per_prompt_seconds": [round(value, 3) for value in per_prompt_seconds],
        "mean_luminance": sum(luminance_values) / len(luminance_values) if luminance_values else None,
        "model_ref_meta": model_ref_meta,
        "metrics": {
            "edit_retention_score": None,
            "generation_regression_score": None,
        },
        "status": "passed",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": args.output}, indent=2))
