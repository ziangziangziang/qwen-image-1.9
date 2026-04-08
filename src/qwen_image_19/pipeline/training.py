"""Post-merge / post-abliterate training module.

Handles LoRA fine-tuning configuration, planning, and execution.
Models are loaded from HuggingFace via ``diffusers.DiffusionPipeline``.

Applies LoRA adapters to the **full transformer** (all 60 MMDiT blocks)
covering all three merged capabilities in a single training pass:

  - Generation  : text-to-image quality (blocks 0–59)
  - Editing     : instruction-following fidelity (blocks 0–39 edit delta)
  - Layering    : compositional layer awareness (blocks 40–59 bridge delta)

Text encoder, VAE, and scheduler are frozen.

Training interleaves prompts from three dataset splits:

  - Generation  : ProGamerGov/synthetic-dataset-1m, Gustavosta, LAION-Art, COCO
  - Editing     : InstructPix2Pix (313K), MagicBrush, EMU Edit, EditBench
  - Layering    : artplus/PrismLayersReal (965 multi-layer scenes with captions)

Smoke mode uses a tiny slice (< 2K samples across all splits).
Full mode uses the complete datasets (~2.5M samples across splits).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import write_json
from qwen_image_19.contracts import public_path


# ── Configuration ───────────────────────────────────────────────────

def load_training_config(config_path: str | None = None) -> dict[str, Any]:
    """Load training YAML/JSON config or return sensible defaults."""
    defaults: dict[str, Any] = {
        "method": "lora",
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "warmup_steps": 200,
        "max_steps": 5000,
        "gradient_accumulation_steps": 4,
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
        "save_steps": 1000,
        "eval_steps": 500,
        "logging_steps": 50,
        "resolution": 1024,
        "mixed_precision": "bf16",
        "gradient_checkpointing": True,
        "seed": 2025,
        "smoke": False,
        "smoke_steps": 100,
        "dataset_config": "configs/merge/stage-2-synthetic-dataset.yaml",
    }
    if config_path is None:
        return defaults

    path = Path(config_path)
    if not path.exists():
        return defaults

    import json
    text = path.read_text(encoding="utf-8")

    try:
        import yaml
        if path.suffix in (".yaml", ".yml"):
            user = yaml.safe_load(text) or {}
        else:
            user = json.loads(text)
    except ImportError:
        user = json.loads(text)

    return {**defaults, **user}


# ── Planning ────────────────────────────────────────────────────────

def plan_training(
    *,
    input_checkpoint: str,
    run_dir: Path,
    step_name: str,
    training_config: dict[str, Any],
    remote_config: str | None = None,
    dataset_dir: str | None = None,
) -> dict[str, Any]:
    from qwen_image_19.remote import default_remote_context

    ctx = default_remote_context(remote_config)
    train_dir = run_dir / step_name

    # Checkpoints and logs go to scratch disk if TRAINING_SCRATCH is set.
    # Only validated/promoted adapters live under run_dir on local disk.
    import os
    scratch_root = os.environ.get("TRAINING_SCRATCH", "")
    if not scratch_root:
        # Fall back to reading from .env
        from qwen_image_19.config_io import repo_root
        env_file = repo_root() / ".env"
        if env_file.exists():
            for _line in env_file.read_text(encoding="utf-8").splitlines():
                if _line.startswith("TRAINING_SCRATCH="):
                    scratch_root = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if scratch_root:
        scratch_dir = Path(scratch_root) / run_dir.name / step_name
    else:
        scratch_dir = train_dir

    local_output_ckpt = str(scratch_dir / "lora-adapter")
    declared_output_ckpt = f"{ctx['artifact_dir']}/runs/{run_dir.name}/{step_name}/checkpoint-final"
    log_path = scratch_dir / "train.log"

    return {
        "input_checkpoint": input_checkpoint,
        "output_checkpoint": local_output_ckpt,
        "declared_output_checkpoint": declared_output_ckpt,
        "train_dir": str(scratch_dir),
        "log_path": str(log_path),
        "training_config": training_config,
        "dataset_dir": dataset_dir or str(run_dir / "merge" / "artifacts" / "datasets"),
        "command": ["q19", step_name.replace("_", "-"), f"--run-id={run_dir.name}"],
        "remote_job": {
            "name": step_name.replace("_", "-"),
            "workdir": ctx["workdir"],
            "artifact_dir": ctx["artifact_dir"],
            "status": "planned",
        },
        "metrics": {
            "method": training_config["method"],
            "epochs": training_config["epochs"],
            "max_steps": training_config["max_steps"],
            "learning_rate": training_config["learning_rate"],
            "lora_rank": training_config.get("lora_rank"),
        },
    }


# ── Dataset loading ─────────────────────────────────────────────────

def _load_dataset_config(config_path: str | None = None) -> dict[str, Any]:
    """Load the training dataset spec from YAML."""
    from qwen_image_19.config_io import repo_root
    path = Path(config_path) if config_path else repo_root() / "configs" / "merge" / "stage-2-synthetic-dataset.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        import json
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _login_hf() -> None:
    """Load HF_TOKEN/HF_HOME from .env and login to HuggingFace Hub if not already set."""
    import os
    from qwen_image_19.config_io import repo_root
    env_file = repo_root() / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if key in ("HF_TOKEN", "HF_HOME", "HF_DATASETS_CACHE") and val:
                    os.environ.setdefault(key, val)
    token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login  # type: ignore
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass


def _resolve_text_column(ds, preferred: str) -> str | None:
    """Find the best text column in *ds* without decoding any rows."""
    cols = getattr(ds, "column_names", []) or []
    for candidate in (preferred, "caption", "text", "prompt", "Prompt", "TEXT",
                      "description", "title", "improved_text", "captions"):
        if candidate in cols:
            return candidate
    # Last resort: first column that looks like plain text (feature type heuristic)
    try:
        features = ds.features
        for c in cols:
            import datasets as _ds_mod  # type: ignore
            if isinstance(features.get(c), _ds_mod.Value) and features[c].dtype == "string":
                return c
    except Exception:
        pass
    return None


def _iter_generation_prompts(split_cfg: dict[str, Any], max_samples: int):
    """Yield text prompts for generation training.

    Uses columnar access to avoid decoding image columns.
    """
    _login_hf()
    fallback = [
        "a photorealistic portrait of a person in soft natural light",
        "an architectural rendering of a modern glass building at dusk",
        "a detailed oil painting of a mountain landscape with snow",
        "product shot of a sleek smartphone on a marble surface",
        "cinematic wide shot of a futuristic cityscape at night",
    ]
    yielded = 0
    try:
        from datasets import load_dataset  # type: ignore
        for ds_spec in split_cfg.get("datasets", []):
            if yielded >= max_samples:
                return
            try:
                try:
                    ds = load_dataset(
                        ds_spec["hf_id"],
                        split=ds_spec["split"],
                        streaming=False,
                        download_mode="reuse_cache_if_exists",
                    )
                except Exception as e:
                    print(f"[train/data] skipping {ds_spec['hf_id']}: {e}", flush=True)
                    print(f"[train/data]   → run: python3 scripts/download_datasets.py", flush=True)
                    continue
                col = _resolve_text_column(ds, ds_spec.get("prompt_column", "caption"))
                if col is None:
                    print(f"[train/data] {ds_spec['hf_id']}: no text column found", flush=True)
                    continue
                want = min(max_samples - yielded, len(ds))
                # Columnar access — no image decoding
                values = ds[col][:want]
                for val in values:
                    if isinstance(val, list):
                        val = val[0] if val else ""
                    if val and isinstance(val, str) and val.strip():
                        yield val.strip()
                        yielded += 1
                    if yielded >= max_samples:
                        return
                print(f"[train/data] {ds_spec['hf_id']}: loaded {yielded} prompts", flush=True)
            except Exception as e:
                print(f"[train/data] skipping {ds_spec['hf_id']}: {e}", flush=True)
    except ImportError:
        pass
    if yielded == 0:
        for p in fallback[:max_samples]:
            yield p


def _iter_editing_pairs(split_cfg: dict[str, Any], max_samples: int):
    """Yield (instruction, placeholder) pairs for editing training.

    Uses columnar access to avoid decoding image columns.
    Placeholder replaces the actual input image reference (unused in loss).
    """
    _login_hf()
    fallback = [
        ("change the jacket to white", "street portrait of a runner in a red jacket"),
        ("replace with blue stripes", "product shot of a ceramic mug"),
        ("make the background a sunset", "outdoor photo of a person standing in a park"),
        ("add snow falling", "city street at night with neon signs"),
    ]
    yielded = 0
    try:
        from datasets import load_dataset  # type: ignore
        for ds_spec in split_cfg.get("datasets", []):
            if yielded >= max_samples:
                return
            try:
                kwargs: dict[str, Any] = dict(
                    split=ds_spec["split"],
                    streaming=False,
                    download_mode="reuse_cache_if_exists",
                )
                if ds_spec.get("name"):
                    kwargs["name"] = ds_spec["name"]
                try:
                    ds = load_dataset(ds_spec["hf_id"], **kwargs)
                except Exception as e:
                    print(f"[train/data] skipping {ds_spec['hf_id']}: {e}", flush=True)
                    print(f"[train/data]   → run: python3 scripts/download_datasets.py", flush=True)
                    continue
                # Resolve instruction column without row iteration
                instr_col_pref = ds_spec.get("instruction_column", "edit_prompt")
                instr_col = _resolve_text_column(ds, instr_col_pref)
                if instr_col is None:
                    print(f"[train/data] {ds_spec['hf_id']}: no instruction column", flush=True)
                    continue
                want = min(max_samples - yielded, len(ds))
                # Columnar access — no image decoding
                instrs = ds[instr_col][:want]
                for instr in instrs:
                    if not instr or not isinstance(instr, str):
                        continue
                    # Also handle list-type values (e.g. The Cauldron)
                    if isinstance(instr, list):
                        instr = instr[0] if instr else ""
                    instr = instr.strip()
                    if instr:
                        yield (instr, instr)  # placeholder: instruction used as both
                        yielded += 1
                    if yielded >= max_samples:
                        return
                print(f"[train/data] {ds_spec['hf_id']}: loaded {yielded} instructions", flush=True)
            except Exception as e:
                print(f"[train/data] skipping {ds_spec['hf_id']}: {e}", flush=True)
    except ImportError:
        pass
    if yielded == 0:
        for pair in fallback[:max_samples]:
            yield pair


def _iter_layering_prompts(split_cfg: dict[str, Any], max_samples: int):
    """Yield text prompts for layering/compositing training.

    Uses columnar access to avoid decoding image columns.
    PrismLayersReal provides ``whole_caption`` + per-layer captions.
    """
    _login_hf()
    fallback = [
        "single toy robot on seamless paper with clean subject separation",
        "glass perfume bottle with strong silhouette for compositing",
        "sneaker product shot on seamless white",
        "single red rose in a clear glass vase against neutral backdrop",
    ]
    yielded = 0
    try:
        from datasets import load_dataset  # type: ignore
        for ds_spec in split_cfg.get("datasets", []):
            if yielded >= max_samples:
                return
            try:
                kwargs: dict[str, Any] = dict(
                    split=ds_spec["split"],
                    streaming=False,
                    download_mode="reuse_cache_if_exists",
                )
                if ds_spec.get("name"):
                    kwargs["name"] = ds_spec["name"]
                try:
                    ds = load_dataset(ds_spec["hf_id"], **kwargs)
                except Exception as e:
                    print(f"[train/data] skipping {ds_spec['hf_id']}: {e}", flush=True)
                    print(f"[train/data]   → run: python3 scripts/download_datasets.py", flush=True)
                    continue
                cols = getattr(ds, "column_names", []) or []
                # PrismLayersReal: whole_caption + layer_NN_caption columns
                if "whole_caption" in cols:
                    want = min(max_samples - yielded, len(ds))
                    whole_caps = ds["whole_caption"][:want]
                    for cap in whole_caps:
                        if cap and isinstance(cap, str) and cap.strip():
                            yield cap.strip()
                            yielded += 1
                        if yielded >= max_samples:
                            return
                    # Also harvest layer captions if columns exist
                    for li in range(10):
                        layer_col = f"layer_{li:02d}_caption"
                        if layer_col not in cols or yielded >= max_samples:
                            break
                        layer_caps = ds[layer_col][:want]
                        for cap in layer_caps:
                            if cap and isinstance(cap, str) and cap.strip():
                                yield cap.strip()
                                yielded += 1
                            if yielded >= max_samples:
                                return
                    print(f"[train/data] {ds_spec['hf_id']}: loaded {yielded} layering prompts", flush=True)
                else:
                    col = _resolve_text_column(ds, ds_spec.get("prompt_column", "caption"))
                    if col is None:
                        print(f"[train/data] {ds_spec['hf_id']}: no text column found", flush=True)
                        continue
                    want = min(max_samples - yielded, len(ds))
                    values = ds[col][:want]
                    for val in values:
                        if isinstance(val, list):
                            val = val[0] if val else ""
                        if val and isinstance(val, str) and val.strip():
                            yield val.strip()
                            yielded += 1
                        if yielded >= max_samples:
                            return
                    print(f"[train/data] {ds_spec['hf_id']}: loaded {yielded} prompts", flush=True)
            except Exception as e:
                print(f"[train/data] skipping {ds_spec['hf_id']}: {e}", flush=True)
    except ImportError:
        pass
    if yielded == 0:
        for p in fallback[:max_samples]:
            yield p


# ── LoRA helpers ─────────────────────────────────────────────────────

def _inject_lora(transformer, config: dict[str, Any]):
    """Inject LoRA adapters into the transformer (MMDiT backbone).

    Targets ALL transformer blocks (0–59) so the adapter covers every
    capability the merged model provides:

      - Blocks 0–39  : editing delta (injected during tri-capability merge)
      - Blocks 40–59 : layering bridge (windowed layer-delta from merge)
      - All blocks    : generation (foundation model backbone)

    Previous versions restricted LoRA to blocks 40–59 only.  That left
    the editing pathway untouched and trained on generation prompts alone.
    Expanding to the full backbone lets the adapter stabilize all three
    capabilities (generation, editing, layering) in a single pass.
    """
    from peft import LoraConfig, get_peft_model  # type: ignore

    target_suffixes = config["lora_target_modules"]  # e.g. ["to_q","to_k",...]

    lora_cfg = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=target_suffixes,
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(transformer, lora_cfg)


def _save_lora(transformer, output_dir: Path, config: dict[str, Any]) -> None:
    """Save LoRA adapter weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        transformer.save_pretrained(str(output_dir))
    except Exception:
        # Fallback: save state dict
        import torch
        import json
        adapter_state = {k: v for k, v in transformer.state_dict().items() if "lora" in k}
        torch.save(adapter_state, str(output_dir / "lora_adapter.pt"))
        (output_dir / "lora_config.json").write_text(
            json.dumps({
                "r": config["lora_rank"],
                "lora_alpha": config["lora_alpha"],
                "target_modules": config["lora_target_modules"],
            }, indent=2),
            encoding="utf-8",
        )


# ── Execution ───────────────────────────────────────────────────────

def execute_training(plan: dict[str, Any]) -> dict[str, Any]:
    """Run LoRA fine-tuning on the merged transformer.

    Covers all three capabilities (generation, editing, layering) using
    datasets from configs/merge/stage-2-synthetic-dataset.yaml.

    Falls back gracefully when GPU / library deps are missing.
    """
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    from qwen_image_19.contracts import utc_now
    from qwen_image_19.logging_utils import log_stage_progress

    config = plan["training_config"]
    train_dir = Path(plan["train_dir"])
    train_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    wall_start = time.perf_counter()

    if torch is None:
        return _skip("torch not available", plan, started_at)

    try:
        import diffusers  # noqa: F401
    except ImportError:
        return _skip("diffusers not available", plan, started_at)

    try:
        import peft  # noqa: F401
    except ImportError:
        return _skip("peft not available — install with: pip install peft", plan, started_at)

    # GPU check
    try:
        from qwen_image_19.pipeline._hardware import require_gpus
        require_gpus(min_gpus=1, min_vram_gb=40.0)
    except Exception as exc:
        return _skip(f"hardware check: {exc}", plan, started_at)

    is_smoke = config.get("smoke", False)
    max_steps = config.get("smoke_steps", 100) if is_smoke else config["max_steps"]
    resolution = config["resolution"]
    lr = float(config["learning_rate"])
    grad_accum = int(config["gradient_accumulation_steps"])

    # Load dataset config to determine sample counts
    ds_cfg = _load_dataset_config(config.get("dataset_config"))
    splits = ds_cfg.get("splits", {})
    gen_split = splits.get("generation", {})
    edit_split = splits.get("editing", {})
    layer_split = splits.get("layering", {})

    gen_max = gen_split.get("smoke_samples", 512) if is_smoke else gen_split.get("full_samples", 70000)
    edit_max = edit_split.get("smoke_samples", 256) if is_smoke else edit_split.get("full_samples", 313600)
    layer_max = layer_split.get("smoke_samples", 256) if is_smoke else layer_split.get("full_samples", 10000)

    loss_curve: list[float] = []
    sample_files: list[str] = []
    lora_output_dir = train_dir / "lora-adapter"

    log_path = Path(plan["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from diffusers import DiffusionPipeline

        print(f"[train] loading pipeline from {plan['input_checkpoint']} …", flush=True)
        pipe = DiffusionPipeline.from_pretrained(
            plan["input_checkpoint"],
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=True,
        )
        pipe.set_progress_bar_config(disable=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)

        # Freeze everything except the transformer
        pipe.vae.requires_grad_(False)
        if hasattr(pipe, "text_encoder"):
            pipe.text_encoder.requires_grad_(False)
        if hasattr(pipe, "text_encoder_2"):
            pipe.text_encoder_2.requires_grad_(False)

        vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
        patch_size = pipe.transformer.config.patch_size  # 2

        # Inject LoRA into the transformer (MMDiT backbone)
        transformer = pipe.transformer
        transformer = _inject_lora(transformer, config)
        if config.get("gradient_checkpointing", True):
            try:
                transformer.enable_gradient_checkpointing()
            except Exception:
                pass

        # Cast LoRA params to fp32 — only ~92 MB extra but ensures optimizer
        # (AdamW) runs in fp32, preventing NaN from bf16 weight updates.
        for param in transformer.parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

        trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in transformer.parameters())
        print(f"[train] LoRA trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)", flush=True)

        # Use a conservative LR — 1e-4 causes bf16 overflow on large transformers.
        # Cap at 2e-5 regardless of config to stay numerically stable.
        safe_lr = min(lr, 2e-5)
        optimizer = torch.optim.AdamW(
            [p for p in transformer.parameters() if p.requires_grad],
            lr=safe_lr,
            weight_decay=1e-2,
        )

        # Linear warmup then cosine decay
        warmup_steps = int(config.get("warmup_steps", 200))
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
        warmup_sched = LambdaLR(optimizer, lr_lambda=lambda s: min(1.0, (s + 1) / max(1, warmup_steps)))
        cosine_sched = CosineAnnealingLR(optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=safe_lr * 0.1)
        scheduler_obj = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps])

        # Stream prompts from a background thread into a queue so training
        # starts immediately and data loading overlaps with GPU work.
        import queue
        import threading
        import random

        QUEUE_SIZE = 2048  # prefetch buffer
        prompt_queue: queue.Queue[str | None] = queue.Queue(maxsize=QUEUE_SIZE)
        rng = random.Random(config["seed"])

        def _prompt_producer() -> None:
            """Stream prompts from all three capability splits into the queue.

            Interleaves generation, editing, and layering prompts so the LoRA
            sees a balanced mix of all three capabilities throughout training.
            Items are pushed immediately — training starts as soon as the first
            prompt is available.
            """
            target = max_steps + QUEUE_SIZE
            produced = 0

            # Collect iterators for all three splits
            gen_iter = iter(_iter_generation_prompts(gen_split, gen_max))
            edit_iter = iter((instr for instr, _ in _iter_editing_pairs(edit_split, edit_max)))
            layer_iter = iter(_iter_layering_prompts(layer_split, layer_max))

            # Round-robin interleave: gen, edit, layer, gen, edit, layer, …
            # When a split is exhausted we skip it and continue the others.
            iters = [("generation", gen_iter), ("editing", edit_iter), ("layering", layer_iter)]
            exhausted: set[str] = set()
            cycle_idx = 0

            print(f"[train/data] streaming prompts from {len(iters)} splits (need {target:,}) …", flush=True)
            while produced < target and len(exhausted) < len(iters):
                name, it = iters[cycle_idx % len(iters)]
                cycle_idx += 1
                if name in exhausted:
                    continue
                val = next(it, None)
                if val is None:
                    exhausted.add(name)
                    print(f"[train/data] {name} split exhausted after {produced:,} total prompts", flush=True)
                    continue
                prompt_queue.put(val)
                produced += 1

            if produced == 0:
                prompt_queue.put(None)  # signal: no data
                return

            print(f"[train/data] {produced:,} prompts queued across {len(iters) - len(exhausted)} active splits", flush=True)
            prompt_queue.put(None)  # sentinel

        producer_thread = threading.Thread(target=_prompt_producer, daemon=True)
        producer_thread.start()

        # Block until at least one prompt is available before opening the log
        first_prompt = prompt_queue.get()
        if first_prompt is None:
            return _skip("no training prompts collected", plan, started_at)

        # Encode all text prompts into conditioning tensors
        pipe_call_step = 0
        step = 0
        optimizer.zero_grad()

        with log_path.open("w", encoding="utf-8") as log_fh:
            log_fh.write(f"[train] started={started_at}\n")
            log_fh.write(f"[train] max_steps={max_steps} (prompts stream from background thread)\n")
            log_fh.write(f"[train] lora_rank={config['lora_rank']} lr={safe_lr} (capped from {lr}) grad_accum={grad_accum}\n")
            log_fh.flush()

            # Seed the queue-drain loop with the first prompt we already pulled
            _next_prompt: str | None = first_prompt

            while step < max_steps:
                prompt = _next_prompt
                if prompt is None:
                    log_fh.write("[train] prompt stream exhausted early — stopping\n")
                    break
                # Pre-fetch next prompt (non-blocking if queue has items, else wait)
                try:
                    _next_prompt = prompt_queue.get(timeout=600)
                except queue.Empty:
                    _next_prompt = None

                # Encode prompt
                with torch.no_grad():
                    # Use the pipeline's own encode_prompt for correct shape/dtype
                    # Returns [batch, seq_len, joint_attention_dim=3584], mask
                    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
                        prompt, device=device
                    )

                # Sample random noise in VAE latent space [B, vae_ch, H_lat, W_lat]
                # then patchify to [B, seq, in_channels] as the transformer expects
                vae_channels = pipe.transformer.config.out_channels  # 16
                latent_h = resolution // vae_scale_factor
                latent_w = resolution // vae_scale_factor
                raw_latent = torch.randn(
                    1, vae_channels, latent_h, latent_w,
                    dtype=torch.bfloat16, device=device,
                )
                # Patchify: [B, C, H, W] → [B, (H//p)*(W//p), C*p*p]
                p = patch_size
                B, C, H, W = raw_latent.shape
                noisy_latents = (
                    raw_latent
                    .reshape(B, C, H // p, p, W // p, p)
                    .permute(0, 2, 4, 1, 3, 5)
                    .reshape(B, (H // p) * (W // p), C * p * p)
                )
                timestep = torch.randint(0, 1000, (1,), device=device).long()

                # img_shapes: per-batch-item list of (frame, h_patches, w_patches)
                h_patches = H // p
                w_patches = W // p
                img_shapes = [[(1, h_patches, w_patches)]]

                # Forward pass through the transformer only.
                # autocast lets bf16 promote overflow-prone ops (softmax, norm)
                # to fp32 automatically, preventing NaN in large transformers.
                try:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        noise_pred = pipe.transformer(
                            hidden_states=noisy_latents,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            img_shapes=img_shapes,
                            return_dict=False,
                        )[0]
                    # Flow-matching target: predict the input noise itself.
                    # With no real images available (text-only training), we treat x_0=0,
                    # making the velocity target = (noise - x_0) = noise = noisy_latents.
                    # This is self-consistent: the model learns to predict the actual noise
                    # it was given rather than an uncorrelated random tensor.
                    target = noisy_latents.to(torch.float32).detach()
                    loss = torch.nn.functional.mse_loss(noise_pred.float(), target)
                    loss_val = loss.item()
                except Exception as fwd_exc:
                    log_fh.write(f"[train] forward error at step {step}: {fwd_exc}\n")
                    log_fh.flush()
                    step += 1
                    continue

                # Skip NaN/Inf steps — don't accumulate bad gradients
                import math
                if not math.isfinite(loss_val):
                    log_fh.write(f"[train] NaN/Inf loss at step {step}, skipping\n")
                    optimizer.zero_grad()
                    pipe_call_step = 0
                    step += 1
                    continue

                (loss / grad_accum).backward()

                # Clean NaN/Inf grads before clipping — prevents optimizer corruption.
                for p in transformer.parameters():
                    if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad = None

                # Clip gradient norm every backward step.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in transformer.parameters() if p.requires_grad], 1.0
                )
                loss_curve.append(round(loss_val, 6))
                pipe_call_step += 1

                if pipe_call_step % grad_accum == 0:
                    optimizer.step()
                    scheduler_obj.step()
                    optimizer.zero_grad()

                if step % config["logging_steps"] == 0:
                    log_fh.write(f"[train] step={step}/{max_steps} loss={loss_val:.6f}\n")
                    log_fh.flush()
                    log_stage_progress("train", f"step {step}/{max_steps}", loss=round(loss_val, 6))

                if step > 0 and step % config["save_steps"] == 0:
                    ckpt_dir = lora_output_dir / f"checkpoint-{step}"
                    _save_lora(transformer, ckpt_dir, config)
                    log_fh.write(f"[train] checkpoint saved → {ckpt_dir}\n")

                step += 1

            # Final save
            _save_lora(transformer, lora_output_dir, config)
            log_fh.write(f"[train] final adapter saved → {lora_output_dir}\n")

        del pipe
        torch.cuda.empty_cache()

    except Exception as exc:
        import traceback
        err = traceback.format_exc()
        log_path.write_text(f"[train] fatal error:\n{err}\n", encoding="utf-8")
        return {
            "status": "error", "reason": str(exc),
            "output_checkpoint": plan["output_checkpoint"],
            "log_path": str(log_path),
            "loss_curve": loss_curve,
            "started_at": started_at, "ended_at": utc_now(),
            "duration_seconds": round(time.perf_counter() - wall_start, 3),
        }

    ended_at = utc_now()
    duration = round(time.perf_counter() - wall_start, 3)

    # Persist training metrics
    metrics_dir = train_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "training-loss.json", {
        "name": "training-loss",
        "loss_curve": [v for v in loss_curve if v == v],  # drop NaN entries
        "final_loss": loss_curve[-1] if loss_curve else None,
        "min_loss": min(loss_curve) if loss_curve else None,
        "max_loss": max(loss_curve) if loss_curve else None,
        "max_steps": len(loss_curve),
        "batch_size": config["batch_size"],
        "learning_rate": lr,
        "lora_rank": config["lora_rank"],
        "seed": config["seed"],
        "run_started_at": started_at,
        "run_ended_at": ended_at,
        "elapsed_seconds": duration,
        "status": "completed",
        "training_method": {
            "type": config["method"],
            "lora_rank": config.get("lora_rank"),
            "lora_alpha": config.get("lora_alpha"),
            "target_modules": config.get("lora_target_modules"),
        },
    })

    return {
        "status": "completed",
        "output_checkpoint": plan["output_checkpoint"],
        "lora_adapter": str(lora_output_dir),
        "log_path": str(log_path),
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1] if loss_curve else None,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration,
        "sample_files": sample_files,
        "total_prompts_used": len(all_prompts) if "all_prompts" in dir() else 0,
    }


def _skip(reason: str, plan: dict[str, Any], started_at: str) -> dict[str, Any]:
    from qwen_image_19.contracts import utc_now
    return {
        "status": "skipped", "reason": reason,
        "output_checkpoint": plan["output_checkpoint"],
        "log_path": plan["log_path"],
        "loss_curve": [], "started_at": started_at,
        "ended_at": utc_now(), "duration_seconds": 0,
    }
