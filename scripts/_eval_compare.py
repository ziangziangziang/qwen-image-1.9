#!/usr/bin/env python3
"""GPU-side eval: original vs merged vs merged+LoRA three-panel comparison."""
import gc, json, time, torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

COMPARISON_DIR = Path('/scratch/training/slerp-selective-20260408/comparison')
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
FOUNDATION = '/home/test/.cache/huggingface/hub/models--Qwen--Qwen-Image-2512/snapshots/25468b98e3276ca6700de15c6628e51b7de54a26'
MERGED = '/scratch/training/slerp-selective-20260408/merge/merged-tri-capability-checkpoint'
LORA_ADAPTER = '/scratch/training/prod-20260408/post_merge_train/lora-adapter'
PROMPTS = [
    "a sweeping aerial photograph of a coastal city at golden hour",
    "a hyperrealistic macro shot of a dewdrop on a spider web",
    "an architectural visualization of a minimalist Japanese tea house surrounded by bamboo",
    "a cinematic portrait of an astronaut looking out a spacecraft window at Earth",
    "a watercolor painting of a busy Parisian café in the rain",
    "a product shot of a glass perfume bottle on black marble with dramatic lighting",
    "a dramatic storm over the Grand Canyon at sunset with lightning bolts",
    "a detailed pencil sketch of an old library filled with towering bookshelves",
]
EDIT_PROMPTS = [
    ["change the sky to a dramatic sunset", "street photo of a modern city block"],
    ["add falling cherry blossom petals", "japanese garden with stone lanterns"],
    ["make it a pencil sketch", "portrait of a woman in natural light"],
    ["replace the background with a snowy mountain", "outdoor portrait of a hiker"],
]
SEED = 42
GEN_KWARGS = dict(num_inference_steps=28, guidance_scale=4.5, height=1024, width=1024)

def load_pipe(ckpt, lora_path=None):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16, local_files_only=True,
    )
    if lora_path and Path(lora_path).exists():
        try:
            pipe.transformer.load_adapter(lora_path)
            print(f"[eval] loaded LoRA from {lora_path}", flush=True)
        except Exception as e:
            print(f"[eval] LoRA load failed: {e}", flush=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")

def unload(pipe):
    pipe.to("cpu")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

def gen(pipe, prompt, seed=SEED):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(prompt, generator=g, **GEN_KWARGS).images[0]

def three_panel(img_orig, img_merged, img_lora,
                label_l="Original", label_m="Merged", label_r="Merged+LoRA"):
    W = img_orig.width + img_merged.width + img_lora.width + 16
    H = max(img_orig.height, img_merged.height, img_lora.height) + 52
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    canvas.paste(img_orig,   (0, 52))
    canvas.paste(img_merged, (img_orig.width + 8, 52))
    canvas.paste(img_lora,   (img_orig.width + img_merged.width + 16, 52))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 14), label_l, fill=(240, 240, 240), font=font)
    draw.text((img_orig.width + 18, 14), label_m, fill=(240, 240, 240), font=font)
    draw.text((img_orig.width + img_merged.width + 26, 14), label_r, fill=(240, 240, 240), font=font)
    return canvas

all_prompts = [(p, "generation", None) for p in PROMPTS] + \
              [(f"{bp}. {ei}", "editing", ei) for ei, bp in EDIT_PROMPTS]
results = []
t_start = time.time()

# Pass 1: original
print("[eval] loading ORIGINAL pipeline…", flush=True)
pipe = load_pipe(FOUNDATION)
orig_imgs = {}
for prompt, kind, _ in all_prompts:
    print(f"[eval] original {kind}: {prompt[:60]}…", flush=True)
    orig_imgs[prompt] = gen(pipe, prompt)
print("[eval] unloading original…", flush=True)
unload(pipe)

# Pass 2: merged (no LoRA)
print("[eval] loading MERGED pipeline (no LoRA)…", flush=True)
pipe = load_pipe(MERGED)
merged_imgs = {}
for prompt, kind, _ in all_prompts:
    print(f"[eval] merged {kind}: {prompt[:60]}…", flush=True)
    merged_imgs[prompt] = gen(pipe, prompt)
print("[eval] unloading merged…", flush=True)
unload(pipe)

# Pass 3: merged + LoRA
print("[eval] loading MERGED+LoRA pipeline…", flush=True)
pipe = load_pipe(MERGED, LORA_ADAPTER)
for prompt, kind, edit_instr in all_prompts:
    print(f"[eval] lora {kind}: {prompt[:60]}…", flush=True)
    lora_img = gen(pipe, prompt)
    panel = three_panel(orig_imgs[prompt], merged_imgs[prompt], lora_img)
    prefix = "gen" if kind == "generation" else "edit"
    idx = sum(1 for r in results if r["type"] == kind) + 1
    slug = f"{prefix}_{idx:02d}_comparison.png"
    panel.save(str(COMPARISON_DIR / slug))
    entry = {"prompt": prompt, "type": kind, "file": slug}
    if edit_instr:
        entry["instruction"] = edit_instr
    results.append(entry)
    print(f"[eval]   saved {slug}", flush=True)
unload(pipe)

duration = time.time() - t_start
index = {
    "run_id": "prod-20260408",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "duration_seconds": round(duration, 1),
    "panels": ["Original", "Merged", "Merged+LoRA"],
    "foundation_ckpt": FOUNDATION,
    "merged_ckpt": MERGED,
    "lora_adapter": LORA_ADAPTER,
    "total_images": len(results),
    "comparisons": results,
}
(COMPARISON_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n")
print(f"[eval] done. {len(results)} three-panel comparisons saved to {COMPARISON_DIR}", flush=True)
