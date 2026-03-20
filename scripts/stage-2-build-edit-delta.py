#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

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
        "diffusers is required for real smoke execution. Install runtime dependencies on the remote machine."
    ) from exc

try:
    from safetensors.torch import save_file
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "safetensors is required for Stage 2 delta artifact writing."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--candidate-id", default="core-delta-w035")
    parser.add_argument("--output-checkpoint", help="Relative output checkpoint path.")
    parser.add_argument("--foundation-model", help="Foundation model id for smoke PoC generation.")
    parser.add_argument("--edit-model", help="Edit model id for smoke PoC generation.")
    parser.add_argument("--blend-weight", type=float, default=0.35)
    parser.add_argument("--prompt", default="studio product photo of a camera on a clean table")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--required-gpus", type=int, default=2)
    parser.add_argument("--required-total-vram-gb", type=float, default=160.0)
    return parser.parse_args()


def to_chw_tensor(image) -> torch.Tensor:
    rgb = image.convert("RGB")
    width, height = rgb.size
    data = torch.tensor(list(rgb.getdata()), dtype=torch.float32).view(height, width, 3)
    return data.permute(2, 0, 1).contiguous() / 255.0


def generate_image(
    model_id: str,
    prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    required_gpus: int,
    required_total_vram_gb: float,
) -> torch.Tensor:
    try:
        runtime = resolve_stage2_diffusion_runtime(
            required_gpus=required_gpus,
            required_total_vram_gb=required_total_vram_gb,
        )
    except Stage2HardwareError as exc:
        raise SystemExit(str(exc)) from exc
    dtype = torch.bfloat16
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            **runtime.pipeline_load_kwargs,
        )
        pipe.set_progress_bar_config(disable=True)
        generator = torch.Generator(device=runtime.primary_device).manual_seed(seed)
        output = pipe(
            prompt=prompt,
            num_inference_steps=max(1, steps),
            width=width,
            height=height,
            generator=generator,
        )
        image = output.images[0]
        tensor = to_chw_tensor(image)
        del output
        del pipe
        torch.cuda.empty_cache()
        return tensor
    except torch.OutOfMemoryError as exc:
        raise SystemExit(
            "Stage 2 diffusion OOM while generating core delta artifact. "
            f"{runtime.summary()}. No fallback/offload retry is configured."
        ) from exc


def write_artifact(
    candidate_id: str,
    output_checkpoint: str,
    foundation_model: str,
    edit_model: str,
    blend_weight: float,
    prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    required_gpus: int,
    required_total_vram_gb: float,
) -> dict[str, object]:
    path = Path(output_checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    foundation_image = generate_image(
        model_id=foundation_model,
        prompt=prompt,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        required_gpus=required_gpus,
        required_total_vram_gb=required_total_vram_gb,
    )
    edit_image = generate_image(
        model_id=edit_model,
        prompt=prompt,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        required_gpus=required_gpus,
        required_total_vram_gb=required_total_vram_gb,
    )
    delta = edit_image - foundation_image
    blended = torch.clamp(foundation_image + (blend_weight * delta), 0.0, 1.0)
    save_file(
        {
            "poc.foundation_image": foundation_image.cpu(),
            "poc.edit_image": edit_image.cpu(),
            "poc.image_delta": delta.cpu(),
            "poc.blended_image": blended.cpu(),
        },
        str(path),
        metadata={
            "candidate_id": candidate_id,
            "foundation_model": foundation_model,
            "edit_model": edit_model,
            "blend_weight": str(blend_weight),
        },
    )
    payload: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "artifact_type": "bf16-core-smoke-poc",
        "foundation_model": foundation_model,
        "edit_model": edit_model,
        "blend_weight": blend_weight,
        "prompt": prompt,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
        "delta_l2_norm": float(torch.norm(delta).item()),
        "delta_mean_abs": float(torch.mean(torch.abs(delta)).item()),
    }
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    args = parse_args()
    if not args.execute:
        print(json.dumps(fuse(dry_run=True)["manifest"]["core_delta_recipe"], indent=2))
        raise SystemExit(0)
    if not args.output_checkpoint:
        raise SystemExit("--output-checkpoint is required with --execute")
    if not args.foundation_model or not args.edit_model:
        raise SystemExit("--foundation-model and --edit-model are required with --execute")
    artifact = write_artifact(
        candidate_id=args.candidate_id,
        output_checkpoint=args.output_checkpoint,
        foundation_model=args.foundation_model,
        edit_model=args.edit_model,
        blend_weight=args.blend_weight,
        prompt=args.prompt,
        steps=args.steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        required_gpus=args.required_gpus,
        required_total_vram_gb=args.required_total_vram_gb,
    )
    print(json.dumps({"status": "ok", "output_checkpoint": args.output_checkpoint, "metadata": artifact}, indent=2))
