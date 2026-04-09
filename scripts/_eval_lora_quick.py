#!/usr/bin/env python3
"""Quick mid-training LoRA eval: source | edit-model baseline | merged+LoRA (3-panel).

Usage:
    python scripts/_eval_lora_quick.py [--checkpoint <path>] [--steps 20] [--cases 3]

Loads the merged checkpoint + a LoRA adapter, then compares it against
the original edit model on a small set of cases. Non-destructive / read-only
relative to any ongoing training run.
"""
import argparse, gc, time, torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import time as _time

# ── defaults ────────────────────────────────────────────────────────
MERGED      = "/scratch/training/slerp-selective-20260408/merge/merged-tri-capability-checkpoint"
GEN_MODEL   = "/home/test/.cache/huggingface/hub/models--Qwen--Qwen-Image-2512/snapshots/25468b98e3276ca6700de15c6628e51b7de54a26"
EDIT_MODEL  = "/home/test/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
LORA_DIR    = "/scratch/training/slerp-selective-20260408/post_merge_train/lora-adapter"
OUT_DIR     = Path("/scratch/training/slerp-selective-20260408/lora_quick_eval")
SEED        = 42

EDIT_CASES = [
    ("street photo of a modern city block at noon",
     "change the sky to a dramatic sunset with orange and purple clouds"),
    ("japanese garden with stone lanterns and green moss",
     "add falling cherry blossom petals"),
    ("portrait of a young woman in soft natural light",
     "turn it into a detailed pencil sketch"),
    ("outdoor portrait of a hiker in front of green hills",
     "replace the background with a snowy mountain landscape"),
    ("a golden retriever sitting on a park bench",
     "make it look like an oil painting"),
    ("a modern living room with white walls",
     "change the lighting to a warm evening candlelight ambiance"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Path to LoRA checkpoint dir (default: latest under LORA_DIR)")
    p.add_argument("--steps", type=int, default=15,
                   help="Number of inference steps (default: 15)")
    p.add_argument("--cases", type=int, default=3,
                   help="Number of edit cases to eval (default: 3)")
    p.add_argument("--size", type=int, default=512,
                   help="Image resolution (default: 512 to fit alongside training VRAM usage)")
    p.add_argument("--device", default="offload", choices=["offload", "cpu"],
                   help="'offload': enable_model_cpu_offload (default); 'cpu': pure CPU inference")
    return p.parse_args()


def latest_checkpoint(lora_dir: str) -> str:
    base = Path(lora_dir)
    ckpts = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint-N dirs found under {lora_dir}")
    return str(ckpts[-1])


def unload(pipe):
    pipe.to("cpu")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


def make_panel(imgs, labels, pad=8, header_h=52):
    W = sum(i.width for i in imgs) + pad * (len(imgs) - 1)
    H = max(i.height for i in imgs) + header_h
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    x = 0
    for img in imgs:
        canvas.paste(img, (x, header_h))
        x += img.width + pad
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    x = 0
    for img, lbl in zip(imgs, labels):
        draw.text((x + 8, 8), lbl, fill=(240, 240, 240), font=font)
        x += img.width + pad
    return canvas


def main():
    args = parse_args()
    cases = EDIT_CASES[: args.cases]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.checkpoint or latest_checkpoint(LORA_DIR)
    print(f"[eval] LoRA checkpoint: {ckpt_path}", flush=True)
    print(f"[eval] inference_steps={args.steps}  cases={len(cases)}  size={args.size}x{args.size}", flush=True)

    gen_kwargs  = dict(num_inference_steps=args.steps, guidance_scale=4.5,
                       height=args.size, width=args.size)
    edit_kwargs = dict(num_inference_steps=args.steps, height=args.size, width=args.size)

    # ── Step 1: generate source images from the original gen model ──
    print("[eval] loading gen pipeline to generate source images…", flush=True)
    from diffusers import DiffusionPipeline
    gen_pipe = DiffusionPipeline.from_pretrained(
        GEN_MODEL, torch_dtype=torch.bfloat16, local_files_only=True
    )
    gen_pipe.set_progress_bar_config(disable=True)
    gen_pipe.enable_model_cpu_offload()  # stream layers on-demand to fit alongside training

    source_images = []
    for src_prompt, _ in cases:
        print(f"  gen: {src_prompt[:70]}…", flush=True)
        g = torch.Generator(device="cuda").manual_seed(SEED)
        img = gen_pipe(src_prompt, generator=g, **gen_kwargs).images[0]
        source_images.append(img)
    unload(gen_pipe)

    # ── Step 2: baseline edit model ──
    print("[eval] loading original edit model…", flush=True)
    orig_pipe = DiffusionPipeline.from_pretrained(
        EDIT_MODEL, torch_dtype=torch.bfloat16, local_files_only=True
    )
    orig_pipe.set_progress_bar_config(disable=True)
    orig_pipe.enable_model_cpu_offload()

    orig_imgs = []
    for i, ((_, instr), src) in enumerate(zip(cases, source_images)):
        print(f"  baseline edit [{i+1}/{len(cases)}]: {instr[:60]}…", flush=True)
        g = torch.Generator(device="cuda").manual_seed(SEED)
        img = orig_pipe(prompt=instr, image=src, generator=g, **edit_kwargs).images[0]
        orig_imgs.append(img)
    unload(orig_pipe)

    # ── Step 3: merged + LoRA edit ──
    print("[eval] loading merged pipeline + LoRA adapter…", flush=True)
    lora_pipe = DiffusionPipeline.from_pretrained(
        MERGED, torch_dtype=torch.bfloat16, local_files_only=True
    )
    lora_pipe.set_progress_bar_config(disable=True)

    # Load the LoRA adapter — saved via PEFT save_pretrained, load with PeftModel
    print(f"[eval] applying LoRA from {ckpt_path}…", flush=True)
    from peft import PeftModel
    lora_pipe.transformer = PeftModel.from_pretrained(
        lora_pipe.transformer, ckpt_path, adapter_name="magicbrush"
    )
    lora_pipe.transformer.set_adapter("magicbrush")
    lora_pipe.enable_model_cpu_offload()

    lora_imgs = []
    for i, ((_, instr), src) in enumerate(zip(cases, source_images)):
        print(f"  lora edit [{i+1}/{len(cases)}]: {instr[:60]}…", flush=True)
        g = torch.Generator(device="cuda").manual_seed(SEED)
        img = lora_pipe(prompt=instr, image=src, generator=g, **edit_kwargs).images[0]
        lora_imgs.append(img)
    unload(lora_pipe)

    # ── Save panels ──
    ckpt_label = Path(ckpt_path).name  # e.g. "checkpoint-7500"
    for i, ((src_prompt, instr), src, orig, lora) in enumerate(
        zip(cases, source_images, orig_imgs, lora_imgs)
    ):
        panel = make_panel(
            [src, orig, lora],
            [f"Source\n{src_prompt[:40]}", "Edit baseline", f"Merged+LoRA ({ckpt_label})"],
        )
        out_path = OUT_DIR / f"eval_{i+1:02d}_{ckpt_label}.png"
        panel.save(str(out_path))
        print(f"[eval] saved → {out_path}", flush=True)

    # Save source images too for reference
    for i, (img, (src_prompt, _)) in enumerate(zip(source_images, cases)):
        img.save(str(OUT_DIR / f"source_{i+1:02d}.png"))

    elapsed = time.time()
    print(f"[eval] done — {len(cases)} panels written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
