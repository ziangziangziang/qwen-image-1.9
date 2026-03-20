#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import time

import torch
from qwen_image_19.stage_2_fusion.runtime import (
    Stage2DiffusionRuntimeConfig,
    Stage2HardwareError,
    resolve_stage2_diffusion_runtime,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", required=True, help="Path to reports/stage-2/dataset-manifest.json")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--required-gpus", type=int, default=2)
    parser.add_argument("--required-total-vram-gb", type=float, default=160.0)
    return parser.parse_args()


try:
    from diffusers import DiffusionPipeline
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "diffusers is required for true smoke dataset generation."
    ) from exc


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def parse_resolution(resolution: str, max_side: int) -> tuple[int, int]:
    parts = resolution.lower().split("x")
    if len(parts) != 2:
        return max_side, max_side
    try:
        width = max(64, min(max_side, int(parts[0])))
        height = max(64, min(max_side, int(parts[1])))
    except ValueError:
        return max_side, max_side
    return width, height


def load_pipeline(model_id: str, runtime: Stage2DiffusionRuntimeConfig) -> DiffusionPipeline:
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        **runtime.pipeline_load_kwargs,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def render_image(
    pipe: DiffusionPipeline,
    runtime: Stage2DiffusionRuntimeConfig,
    prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    input_image=None,
    negative_prompt: str | None = None,
    true_cfg_scale: float | None = None,
    guidance_scale: float | None = None,
):
    generator = torch.Generator(device=runtime.primary_device).manual_seed(seed)
    call_kwargs = {
        "prompt": prompt,
        "num_inference_steps": max(1, steps),
        "width": width,
        "height": height,
        "generator": generator,
    }
    if input_image is not None:
        call_kwargs["image"] = input_image
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):  # pragma: no cover
        params = {}
    accepts_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params.values())
    optional_kwargs = {
        "negative_prompt": negative_prompt,
        "true_cfg_scale": true_cfg_scale,
        "guidance_scale": guidance_scale,
    }
    for key, value in optional_kwargs.items():
        if value is None:
            continue
        if accepts_var_kwargs or key in params:
            call_kwargs[key] = value
    output = pipe(**call_kwargs)
    return output.images[0]


def render_edit_pair(
    source_pipe: DiffusionPipeline,
    edit_pipe: DiffusionPipeline,
    runtime: Stage2DiffusionRuntimeConfig,
    source_prompt: str,
    edit_instruction: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
    negative_prompt: str | None = None,
    true_cfg_scale: float | None = None,
    guidance_scale: float | None = None,
):
    source_image = render_image(
        source_pipe,
        runtime,
        source_prompt,
        steps,
        width,
        height,
        seed,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
    )
    edited_image = render_image(
        edit_pipe,
        runtime,
        edit_instruction,
        steps,
        width,
        height,
        seed + 1,
        input_image=source_image,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        guidance_scale=guidance_scale,
    )
    return source_image, edited_image


if __name__ == "__main__":
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Dataset manifest not found: {args.manifest}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps(payload, indent=2))
        raise SystemExit(0)
    try:
        runtime = resolve_stage2_diffusion_runtime(
            required_gpus=args.required_gpus,
            required_total_vram_gb=args.required_total_vram_gb,
        )
    except Stage2HardwareError as exc:
        raise SystemExit(str(exc)) from exc

    generated = 0
    started = time.perf_counter()
    records = payload.get("planned_records", [])
    splits = payload.get("splits", {})
    try:
        for split_name, split in splits.items():
            split_records = [record for record in records if str(record["sample_id"]).startswith(f"{split_name}-")]
            if not split_records:
                continue
            if split_name == "edit_teacher":
                source_split = splits.get("generation_teacher")
                source_model = source_split.get("teacher_model") if isinstance(source_split, dict) else None
                if not source_model:
                    raise SystemExit(
                        "Edit-teacher split requires `generation_teacher.teacher_model` in dataset manifest."
                    )

                source_pipe = load_pipeline(source_model, runtime=runtime)
                source_images: dict[str, object] = {}
                try:
                    for record in split_records:
                        settings = record.get("generation_settings", {})
                        width, height = parse_resolution(str(settings.get("resolution", "512x512")), args.max_side)
                        steps = min(int(settings.get("steps", args.max_steps)), args.max_steps)
                        seed = int(record.get("seed", 0))
                        source_prompt = str(record.get("source_prompt", ""))
                        asset_paths = record.get("asset_paths", {})
                        source_image = render_image(
                            source_pipe,
                            runtime,
                            source_prompt,
                            steps,
                            width,
                            height,
                            seed,
                            negative_prompt=args.negative_prompt,
                            true_cfg_scale=args.true_cfg_scale,
                            guidance_scale=args.guidance_scale,
                        )
                        source_path = Path(asset_paths["source_image"])
                        source_path.parent.mkdir(parents=True, exist_ok=True)
                        source_image.save(source_path)
                        source_images[str(record["sample_id"])] = source_image
                finally:
                    del source_pipe
                    torch.cuda.empty_cache()

                edit_pipe = load_pipeline(split["teacher_model"], runtime=runtime)
                try:
                    for record in split_records:
                        settings = record.get("generation_settings", {})
                        width, height = parse_resolution(str(settings.get("resolution", "512x512")), args.max_side)
                        steps = min(int(settings.get("steps", args.max_steps)), args.max_steps)
                        seed = int(record.get("seed", 0))
                        edit_instruction = str(record.get("edit_instruction", ""))
                        asset_paths = record.get("asset_paths", {})
                        source_image = source_images[str(record["sample_id"])]
                        edited_image = render_image(
                            edit_pipe,
                            runtime,
                            edit_instruction,
                            steps,
                            width,
                            height,
                            seed + 1,
                            input_image=source_image,
                            negative_prompt=args.negative_prompt,
                            true_cfg_scale=args.true_cfg_scale,
                            guidance_scale=args.guidance_scale,
                        )
                        edited_path = Path(asset_paths["edited_image"])
                        edited_path.parent.mkdir(parents=True, exist_ok=True)
                        edited_image.save(edited_path)

                        metadata_path = Path(asset_paths["metadata"])
                        write_text(
                            metadata_path,
                            json.dumps(
                                {
                                    "sample_id": record["sample_id"],
                                    "teacher_model": record["teacher_model"],
                                    "seed": seed,
                                    "steps": steps,
                                    "resolution": f"{width}x{height}",
                                    "generated_at": datetime.now(timezone.utc).isoformat(),
                                },
                                indent=2,
                            )
                            + "\n",
                        )
                        generated += 1
                finally:
                    del edit_pipe
                    torch.cuda.empty_cache()
                continue

            pipe = load_pipeline(split["teacher_model"], runtime=runtime)
            try:
                for record in split_records:
                    settings = record.get("generation_settings", {})
                    width, height = parse_resolution(str(settings.get("resolution", "512x512")), args.max_side)
                    steps = min(int(settings.get("steps", args.max_steps)), args.max_steps)
                    seed = int(record.get("seed", 0))
                    asset_paths = record.get("asset_paths", {})

                    prompt = str(record.get("prompt", ""))
                    image = render_image(
                        pipe,
                        runtime,
                        prompt,
                        steps,
                        width,
                        height,
                        seed,
                        negative_prompt=args.negative_prompt,
                        true_cfg_scale=args.true_cfg_scale,
                        guidance_scale=args.guidance_scale,
                    )
                    image_path = Path(asset_paths["image"])
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)

                    metadata_path = Path(asset_paths["metadata"])
                    write_text(
                        metadata_path,
                        json.dumps(
                            {
                                "sample_id": record["sample_id"],
                                "teacher_model": record["teacher_model"],
                                "seed": seed,
                                "steps": steps,
                                "resolution": f"{width}x{height}",
                                "generated_at": datetime.now(timezone.utc).isoformat(),
                            },
                            indent=2,
                        )
                        + "\n",
                    )
                    generated += 1
            finally:
                del pipe
                torch.cuda.empty_cache()
    except torch.OutOfMemoryError as exc:
        raise SystemExit(
            "Stage 2 diffusion OOM during synthetic dataset generation. "
            f"{runtime.summary()}. No fallback/offload retry is configured."
        ) from exc

    sentinel = Path(payload["output_root"]) / "dataset-ready.json"
    elapsed = time.perf_counter() - started
    write_text(
        sentinel,
        json.dumps(
            {
                "record_count": len(payload.get("planned_records", [])),
                "splits": list(payload.get("splits", {}).keys()),
                "generated_records": generated,
                "elapsed_seconds": round(elapsed, 3),
            },
            indent=2,
        )
        + "\n",
    )
    print(json.dumps({"status": "ok", "output_root": payload["output_root"]}, indent=2))
