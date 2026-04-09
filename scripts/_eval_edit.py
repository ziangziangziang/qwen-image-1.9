#!/usr/bin/env python3
"""Edit eval: source image | original edit model | merged model (3-panel)."""
import gc, json, time, torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

EDIT_MODEL   = '/home/test/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9'
GEN_MODEL    = '/home/test/.cache/huggingface/hub/models--Qwen--Qwen-Image-2512/snapshots/25468b98e3276ca6700de15c6628e51b7de54a26'
MERGED       = '/scratch/training/slerp-selective-20260408/merge/merged-tri-capability-checkpoint'
OUT_DIR      = Path('/scratch/training/slerp-selective-20260408/edit_eval')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
GEN_KWARGS  = dict(num_inference_steps=28, guidance_scale=4.5, height=1024, width=1024)
EDIT_KWARGS = dict(num_inference_steps=28, height=1024, width=1024)

# (source_prompt, edit_instruction)
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

def load_gen_pipe(ckpt):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")

def load_edit_pipe(ckpt):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(ckpt, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")

def unload(pipe):
    pipe.to("cpu")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

def generate(pipe, prompt, seed=SEED):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(prompt, generator=g, **GEN_KWARGS).images[0]

def edit(pipe, image, instruction, seed=SEED):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(prompt=instruction, image=image, generator=g, **EDIT_KWARGS).images[0]

def three_panel(img_src, img_orig_edit, img_merged_edit,
                label_l="Source", label_m="Edit Model", label_r="Merged Model"):
    imgs = [img_src, img_orig_edit, img_merged_edit]
    W = sum(i.width for i in imgs) + 16
    H = max(i.height for i in imgs) + 72
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    x = 0
    for i, img in enumerate(imgs):
        canvas.paste(img, (x, 72))
        x += img.width + 8
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font
    labels = [label_l, label_m, label_r]
    x = 0
    for i, (img, lbl) in enumerate(zip(imgs, labels)):
        draw.text((x + 10, 8), lbl, fill=(240, 240, 240), font=font)
        x += img.width + 8
    return canvas

results = []
t_start = time.time()

# --- Step 1: Generate source images with the gen model ---
print("[eval] loading GEN pipeline to create source images…", flush=True)
gen_pipe = load_gen_pipe(GEN_MODEL)
source_images = []
for src_prompt, instr in EDIT_CASES:
    print(f"[eval] generating source: {src_prompt[:70]}…", flush=True)
    img = generate(gen_pipe, src_prompt)
    source_images.append(img)
print("[eval] unloading gen pipeline…", flush=True)
unload(gen_pipe)

# --- Step 2: Original edit model ---
print("[eval] loading ORIGINAL EDIT pipeline…", flush=True)
orig_edit_pipe = load_edit_pipe(EDIT_MODEL)
orig_edit_imgs = []
for i, ((src_prompt, instr), src_img) in enumerate(zip(EDIT_CASES, source_images)):
    print(f"[eval] original edit [{i+1}/{len(EDIT_CASES)}]: {instr[:70]}…", flush=True)
    result = edit(orig_edit_pipe, src_img, instr)
    orig_edit_imgs.append(result)
print("[eval] unloading original edit pipeline…", flush=True)
unload(orig_edit_pipe)

# --- Step 3: Merged model edit ---
print("[eval] loading MERGED pipeline for editing…", flush=True)
merged_edit_pipe = load_edit_pipe(MERGED)
for i, ((src_prompt, instr), src_img, orig_img) in enumerate(
    zip(EDIT_CASES, source_images, orig_edit_imgs)
):
    print(f"[eval] merged edit [{i+1}/{len(EDIT_CASES)}]: {instr[:70]}…", flush=True)
    merged_img = edit(merged_edit_pipe, src_img, instr)
    panel = three_panel(src_img, orig_img, merged_img)
    slug = f"edit_{i+1:02d}_comparison.png"
    panel.save(str(OUT_DIR / slug))
    results.append({
        "file": slug,
        "source_prompt": src_prompt,
        "edit_instruction": instr,
    })
    print(f"[eval]   saved {slug}", flush=True)
unload(merged_edit_pipe)

duration = time.time() - t_start
index = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "duration_seconds": round(duration, 1),
    "panels": ["Source (generated)", "Original Edit Model", "Merged Model"],
    "edit_model": EDIT_MODEL,
    "merged_ckpt": MERGED,
    "comparisons": results,
}
(OUT_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n")
print(f"[eval] done. {len(results)} edit comparisons saved to {OUT_DIR}", flush=True)
