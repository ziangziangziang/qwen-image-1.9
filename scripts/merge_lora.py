#!/usr/bin/env python3
"""Merge a LoRA adapter into the base model and save a self-contained checkpoint.

After merging, the output is a standard DiffusionPipeline checkpoint that can
be loaded without PEFT — no adapter files, no special loading code needed.

Usage:
    python scripts/merge_lora.py \
        --base   /scratch/training/slerp-selective-20260408/merge/merged-tri-capability-checkpoint \
        --lora   /scratch/training/slerp-selective-20260408/post_merge_train/lora-adapter/checkpoint-8000 \
        --output /scratch/training/slerp-selective-20260408/merged-with-lora-ckpt8000

The output directory is a complete DiffusionPipeline checkpoint:
    model_index.json  scheduler/  tokenizer/  transformer/  vae/  ...

Size: same as the base (~54 GB) — LoRA deltas are folded into the weight
matrices at full precision; no extra files or overhead.
"""
import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True,
                   help="Path to base pipeline checkpoint (e.g. merged-tri-capability-checkpoint)")
    p.add_argument("--lora", default=None,
                   help="Path to LoRA adapter directory (default: latest checkpoint-N under base/../lora-adapter)")
    p.add_argument("--output", required=True,
                   help="Output path for the merged standalone checkpoint")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="Save dtype (default: bfloat16)")
    p.add_argument("--adapter-name", default="default",
                   help="PEFT adapter name used when the adapter was saved (default: 'default')")
    return p.parse_args()


def latest_checkpoint(lora_dir: Path) -> Path:
    ckpts = sorted(
        [d for d in lora_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-N dirs found under {lora_dir}")
    return ckpts[-1]


def main():
    args = parse_args()
    import torch
    from diffusers import DiffusionPipeline
    from peft import PeftModel

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    save_dtype = dtype_map[args.dtype]

    base_path = Path(args.base)
    output_path = Path(args.output)

    # Resolve LoRA path
    if args.lora:
        lora_path = Path(args.lora)
    else:
        # Default: look for lora-adapter alongside the base checkpoint
        lora_dir = base_path.parent.parent / "post_merge_train" / "lora-adapter"
        lora_path = latest_checkpoint(lora_dir)
        print(f"[merge] auto-resolved LoRA: {lora_path}")

    print(f"[merge] base:   {base_path}", flush=True)
    print(f"[merge] lora:   {lora_path}", flush=True)
    print(f"[merge] output: {output_path}", flush=True)
    print(f"[merge] dtype:  {args.dtype}", flush=True)

    # ── Step 1: Load base pipeline ──
    print("[merge] loading base pipeline…", flush=True)
    pipe = DiffusionPipeline.from_pretrained(
        str(base_path),
        torch_dtype=save_dtype,
        local_files_only=True,
    )

    # ── Step 2: Wrap transformer with PEFT ──
    print(f"[merge] loading LoRA adapter from {lora_path}…", flush=True)
    pipe.transformer = PeftModel.from_pretrained(
        pipe.transformer,
        str(lora_path),
        adapter_name=args.adapter_name,
    )

    # ── Step 3: Merge LoRA deltas into base weights ──
    print("[merge] merging LoRA deltas into base weights (merge_and_unload)…", flush=True)
    pipe.transformer = pipe.transformer.merge_and_unload()
    print("[merge] merge complete — transformer is now a plain nn.Module", flush=True)

    # Cast to target dtype
    pipe.transformer = pipe.transformer.to(save_dtype)

    # ── Step 4: Save the fully merged pipeline ──
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged pipeline to {output_path}…", flush=True)
    pipe.save_pretrained(str(output_path))

    # Verify output
    saved_files = list(output_path.rglob("*.safetensors")) + list(output_path.rglob("*.bin"))
    total_size = sum(f.stat().st_size for f in saved_files) / 1e9
    print(f"[merge] saved {len(saved_files)} weight files  ({total_size:.1f} GB total)", flush=True)
    print(f"[merge] done → {output_path}", flush=True)


if __name__ == "__main__":
    main()
