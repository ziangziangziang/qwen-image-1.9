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
    # eval-type controls the evaluation mode:
    #   generation (default): text-to-image pass on --model-id
    #   edit: before/after pairs; requires --edit-prompts-json and --foundation-model-id
    #   consistency: paired run on --model-id vs --consistency-baseline-model-id,
    #                computes per-image pixel-L2 drift
    parser.add_argument("--eval-type", choices=["generation", "edit", "consistency"], default="generation")
    parser.add_argument("--model-ref")
    parser.add_argument("--model-id")
    parser.add_argument("--edit-prompts-json", default=None,
                        help="JSON file with list of {source_prompt, edit_instruction} for edit eval.")
    parser.add_argument("--foundation-model-id", default=None,
                        help="Foundation model id used to generate the 'before' image in edit eval.")
    parser.add_argument("--consistency-baseline-model-id", default=None,
                        help="Baseline model id for consistency eval; "
                             "drift is measured between this and --model-id.")
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

    # ── dispatch to the appropriate eval mode ──────────────────────────────────
    if args.eval_type == "edit":
        _run_edit_eval(args)
    elif args.eval_type == "consistency":
        _run_consistency_eval(args)
    else:
        _run_generation_eval(args)


# ─────────────────────────────────────────────────────────────────────────────
# Generation eval  (default)
# ─────────────────────────────────────────────────────────────────────────────

def _run_generation_eval(args: argparse.Namespace) -> None:
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
                "generator": generator,
            }
            add_supported_call_args(
                pipe,
                call_kwargs,
                {
                    "width": args.width,
                    "height": args.height,
                    "negative_prompt": args.negative_prompt,
                    "true_cfg_scale": args.true_cfg_scale,
                    "guidance_scale": args.guidance_scale,
                },
            )
            result = pipe(**call_kwargs)
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
        "eval_type": "generation",
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
        "sample_dir": str(sample_dir),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": args.output}, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Edit eval  — before/after pairs
# ─────────────────────────────────────────────────────────────────────────────

def _run_edit_eval(args: argparse.Namespace) -> None:
    """Generate before/after image pairs for an edit capability evaluation.

    For each (source_prompt, edit_instruction) pair:
      1. Generate the *source* image from the foundation model.
      2. Pass the source image + edit instruction into the *merged* model.
      3. Save both as  edit-NNN-before.png / edit-NNN-after.png.
    The output JSON lists all pairs with paths, per-pair timing, and luminance delta.
    """
    if not args.foundation_model_id:
        raise SystemExit("--foundation-model-id is required for --eval-type edit")
    if not args.edit_prompts_json:
        raise SystemExit("--edit-prompts-json is required for --eval-type edit")

    edit_prompts_path = Path(args.edit_prompts_json)
    if not edit_prompts_path.exists():
        raise SystemExit(f"edit-prompts-json not found: {edit_prompts_path}")
    edit_pairs: list[dict[str, str]] = json.loads(edit_prompts_path.read_text(encoding="utf-8"))
    edit_pairs = edit_pairs[:max(1, args.num_prompts)]

    try:
        runtime = resolve_stage2_diffusion_runtime(
            required_gpus=args.required_gpus,
            required_total_vram_gb=args.required_total_vram_gb,
        )
    except Stage2HardwareError as exc:
        raise SystemExit(str(exc)) from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_dir = output.parent / "edit-samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total_started = time.perf_counter()

    try:
        # Load foundation model for "before" generation
        foundation_pipe = DiffusionPipeline.from_pretrained(
            args.foundation_model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            **runtime.pipeline_load_kwargs,
        )
        foundation_pipe.set_progress_bar_config(disable=True)

        for idx, pair in enumerate(edit_pairs):
            source_prompt = pair.get("source_prompt", "")
            edit_instruction = pair.get("edit_instruction", "")
            generator = torch.Generator(device=runtime.primary_device).manual_seed(args.seed + idx)
            before_kwargs: dict = {
                "prompt": source_prompt,
                "num_inference_steps": max(1, args.steps),
                "generator": generator,
            }
            add_supported_call_args(
                foundation_pipe,
                before_kwargs,
                {
                    "width": args.width,
                    "height": args.height,
                    "negative_prompt": args.negative_prompt,
                    "true_cfg_scale": args.true_cfg_scale,
                    "guidance_scale": args.guidance_scale,
                },
            )
            before_result = foundation_pipe(**before_kwargs)
            before_image = before_result.images[0].convert("RGB")
            before_path = sample_dir / f"edit-{idx + 1:03d}-before.png"
            before_image.save(before_path)

        del foundation_pipe
        torch.cuda.empty_cache()

        # Load merged model for "after" generation
        merged_pipe = DiffusionPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            **runtime.pipeline_load_kwargs,
        )
        merged_pipe.set_progress_bar_config(disable=True)

        for idx, pair in enumerate(edit_pairs):
            edit_instruction = pair.get("edit_instruction", "")
            before_path = sample_dir / f"edit-{idx + 1:03d}-before.png"
            before_image = _load_rgb_image(before_path)

            pair_started = time.perf_counter()
            generator = torch.Generator(device=runtime.primary_device).manual_seed(args.seed + idx)
            after_kwargs: dict = {
                "prompt": edit_instruction,
                "image": before_image,
                "num_inference_steps": max(1, args.steps),
                "generator": generator,
            }
            add_supported_call_args(
                merged_pipe,
                after_kwargs,
                {
                    "width": args.width,
                    "height": args.height,
                    "negative_prompt": args.negative_prompt,
                    "true_cfg_scale": args.true_cfg_scale,
                    "guidance_scale": args.guidance_scale,
                },
            )
            after_result = merged_pipe(**after_kwargs)
            after_image = after_result.images[0].convert("RGB")
            after_path = sample_dir / f"edit-{idx + 1:03d}-after.png"
            after_image.save(after_path)
            pair_elapsed = time.perf_counter() - pair_started

            before_t = _image_to_float_tensor(before_image)
            after_t = _image_to_float_tensor(after_image)
            luminance_delta = float((after_t.mean() - before_t.mean()).item())
            pixel_l2_delta = float(torch.norm(after_t - before_t).item())

            results.append({
                "idx": idx + 1,
                "source_prompt": pair.get("source_prompt", ""),
                "edit_instruction": edit_instruction,
                "before_path": str(before_path),
                "after_path": str(after_path),
                "elapsed_seconds": round(pair_elapsed, 3),
                "luminance_delta": round(luminance_delta, 4),
                "pixel_l2_delta": round(pixel_l2_delta, 4),
            })

        del merged_pipe
        torch.cuda.empty_cache()

    except torch.OutOfMemoryError as exc:
        raise SystemExit(
            "Stage 2 diffusion OOM during edit eval. "
            f"{runtime.summary()}. No fallback/offload retry is configured."
        ) from exc

    total_seconds = time.perf_counter() - total_started
    mean_pixel_l2 = sum(r["pixel_l2_delta"] for r in results) / len(results) if results else None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_type": "edit",
        "task": args.task,
        "model_ref": args.model_ref,
        "merged_model_id": args.model_id,
        "foundation_model_id": args.foundation_model_id,
        "num_pairs": len(results),
        "steps": args.steps,
        "resolution": f"{args.width}x{args.height}",
        "elapsed_seconds": round(total_seconds, 3),
        "mean_pixel_l2_delta": round(mean_pixel_l2, 4) if mean_pixel_l2 is not None else None,
        "pairs": results,
        "metrics": {
            "edit_retention_score": None,
        },
        "status": "passed",
        "sample_dir": str(sample_dir),
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": args.output, "eval_type": "edit"}, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Consistency eval  — drift between foundation and merged model
# ─────────────────────────────────────────────────────────────────────────────

def _run_consistency_eval(args: argparse.Namespace) -> None:
    """Compare merged model outputs against the foundation baseline for the same prompts+seeds.

    Computes per-image pixel-L2 drift (lower = merged model stays close to foundation).
    Does NOT need external metrics packages; uses in-memory tensor arithmetic.
    """
    if not args.consistency_baseline_model_id:
        raise SystemExit("--consistency-baseline-model-id is required for --eval-type consistency")

    try:
        runtime = resolve_stage2_diffusion_runtime(
            required_gpus=args.required_gpus,
            required_total_vram_gb=args.required_total_vram_gb,
        )
    except Stage2HardwareError as exc:
        raise SystemExit(str(exc)) from exc

    # Consistency eval uses a fixed prompt bank from the dataset config; fall back to defaults
    prompts_raw = [
        "studio product photo of a brushed metal camera on cream paper with sharp label text",
        "cinematic portrait of a botanist in a glass greenhouse with readable name badge text",
        "rainy neon street at night with a taxi sign and reflective pavement",
        "storybook castle on a hill at sunrise with crisp title lettering",
        "macro shot of a mechanical watch on dark velvet with metallic highlights",
        "editorial portrait of a chef plating food under dramatic overhead light",
        "wide landscape of a red-rock desert canyon at golden hour",
        "clean cutout of a retro speaker on a neutral gradient background",
        "low-angle photo of skyscrapers against a cloudy blue sky",
        "illustration of a cozy reading nook with warm lamplight and bookshelf",
        "street photo of a cyclist in motion with shallow depth of field",
        "bold minimal poster: single large sans-serif word centered on solid color",
        "nature macro of a dewy spider web in early morning light",
        "underwater photo of a sea turtle among colorful coral reefs",
        "product flat lay of coffee accessories on white marble surface",
        "nighttime city skyline reflection in still water",
    ]
    prompt_count = max(1, min(args.num_prompts, len(prompts_raw)))
    prompts = prompts_raw[:prompt_count]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline_dir = output.parent / "consistency-baseline"
    merged_dir = output.parent / "consistency-merged"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    def _run_pipeline(model_id: str, out_dir: Path, label: str) -> list[dict]:
        records = []
        try:
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
                **runtime.pipeline_load_kwargs,
            )
            pipe.set_progress_bar_config(disable=True)
            for idx, prompt in enumerate(prompts):
                generator = torch.Generator(device=runtime.primary_device).manual_seed(args.seed + idx)
                call_kwargs: dict = {
                    "prompt": prompt,
                    "num_inference_steps": max(1, args.steps),
                    "generator": generator,
                }
                add_supported_call_args(
                    pipe,
                    call_kwargs,
                    {
                        "width": args.width,
                        "height": args.height,
                        "negative_prompt": args.negative_prompt,
                        "true_cfg_scale": args.true_cfg_scale,
                        "guidance_scale": args.guidance_scale,
                    },
                )
                result = pipe(**call_kwargs)
                img = result.images[0].convert("RGB")
                path = out_dir / f"{label}-{idx + 1:03d}.png"
                img.save(path)
                t = _image_to_float_tensor(img)
                records.append({"idx": idx + 1, "path": str(path), "tensor": t, "luminance": float(t.mean().item())})
            del pipe
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError as exc:
            raise SystemExit(
                f"OOM while running consistency eval on {model_id}. "
                f"{runtime.summary()}."
            ) from exc
        return records

    total_started = time.perf_counter()
    baseline_records = _run_pipeline(args.consistency_baseline_model_id, baseline_dir, "baseline")
    merged_records = _run_pipeline(args.model_id, merged_dir, "merged")
    total_seconds = time.perf_counter() - total_started

    pairs = []
    drift_values: list[float] = []
    for b, m in zip(baseline_records, merged_records):
        pixel_l2 = float(torch.norm(m["tensor"] - b["tensor"]).item())
        drift_values.append(pixel_l2)
        pairs.append({
            "idx": b["idx"],
            "prompt": prompts[b["idx"] - 1],
            "baseline_path": b["path"],
            "merged_path": m["path"],
            "baseline_luminance": round(b["luminance"], 4),
            "merged_luminance": round(m["luminance"], 4),
            "pixel_l2_drift": round(pixel_l2, 4),
        })

    mean_drift = sum(drift_values) / len(drift_values) if drift_values else None
    max_drift = max(drift_values) if drift_values else None
    min_drift = min(drift_values) if drift_values else None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_type": "consistency",
        "task": args.task,
        "merged_model_id": args.model_id,
        "baseline_model_id": args.consistency_baseline_model_id,
        "num_prompts": prompt_count,
        "steps": args.steps,
        "resolution": f"{args.width}x{args.height}",
        "elapsed_seconds": round(total_seconds, 3),
        "consistency": {
            "mean_pixel_l2_drift": round(mean_drift, 4) if mean_drift is not None else None,
            "max_pixel_l2_drift": round(max_drift, 4) if max_drift is not None else None,
            "min_pixel_l2_drift": round(min_drift, 4) if min_drift is not None else None,
            "interpretation": (
                "pixel_l2_drift measures per-image tensor distance between merged and baseline outputs "
                "for identical prompts and seeds. Lower values indicate the merge preserved generation behavior."
            ),
        },
        "pairs": pairs,
        "status": "passed",
    }
    output = Path(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "output": args.output,
        "eval_type": "consistency",
        "mean_pixel_l2_drift": payload["consistency"]["mean_pixel_l2_drift"],
    }, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_float_tensor(image) -> "torch.Tensor":
    rgb = image.convert("RGB")
    w, h = rgb.size
    data = torch.tensor(list(rgb.getdata()), dtype=torch.float32).view(h, w, 3)
    return data / 255.0


def _load_rgb_image(path: Path):
    from PIL import Image
    return Image.open(path).convert("RGB")
