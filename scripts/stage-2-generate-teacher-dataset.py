#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", required=True, help="Path to reports/stage-2/dataset-manifest.json")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
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


def load_pipeline(model_id: str) -> DiffusionPipeline:
    if not torch.cuda.is_available():
        raise SystemExit("GPU is required for true Stage 2 smoke execution.")
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def render_image(
    pipe: DiffusionPipeline,
    prompt: str,
    steps: int,
    width: int,
    height: int,
    seed: int,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        num_inference_steps=max(1, steps),
        width=width,
        height=height,
        generator=generator,
    )
    return output.images[0]


if __name__ == "__main__":
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Dataset manifest not found: {args.manifest}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps(payload, indent=2))
        raise SystemExit(0)

    generated = 0
    started = time.perf_counter()
    records = payload.get("planned_records", [])
    for split_name, split in payload.get("splits", {}).items():
        split_records = [record for record in records if str(record["sample_id"]).startswith(f"{split_name}-")]
        if not split_records:
            continue
        pipe = load_pipeline(split["teacher_model"])
        try:
            for record in split_records:
                settings = record.get("generation_settings", {})
                width, height = parse_resolution(str(settings.get("resolution", "512x512")), args.max_side)
                steps = min(int(settings.get("steps", args.max_steps)), args.max_steps)
                seed = int(record.get("seed", 0))
                asset_paths = record.get("asset_paths", {})

                if "source_prompt" in record and "edit_instruction" in record:
                    source_prompt = str(record["source_prompt"])
                    edit_instruction = str(record["edit_instruction"])
                    source_image = render_image(pipe, source_prompt, steps, width, height, seed)
                    edited_prompt = f"{source_prompt}. Edit request: {edit_instruction}"
                    edited_image = render_image(pipe, edited_prompt, steps, width, height, seed + 1)
                    source_path = Path(asset_paths["source_image"])
                    edited_path = Path(asset_paths["edited_image"])
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    edited_path.parent.mkdir(parents=True, exist_ok=True)
                    source_image.save(source_path)
                    edited_image.save(edited_path)
                else:
                    prompt = str(record.get("prompt", ""))
                    image = render_image(pipe, prompt, steps, width, height, seed)
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
