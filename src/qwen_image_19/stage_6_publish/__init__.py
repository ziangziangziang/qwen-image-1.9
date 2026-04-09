"""Stage 6 — Publish to HuggingFace.

Uploads quantized artifacts and a model card to the target HF repo.
Token is read from .env (HF_TOKEN) or environment.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root
from qwen_image_19.contracts import public_path, utc_now


class PublishError(RuntimeError):
    """Raised when HF publish fails."""


# ── Token resolution ──────────────────────────────────────────────────

def _load_hf_token() -> str:
    """Load HF token from environment or .env file."""
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    env_path = repo_root() / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                token = line[len("HF_TOKEN="):].strip().strip('"').strip("'")
                if token:
                    return token
    raise PublishError(
        "HF_TOKEN not found. Set it in the environment or in .env at the repo root."
    )


# ── Model card generation ──────────────────────────────────────────────

def _shorten(path: str) -> str:
    return Path(path).name if path else "—"


def build_model_card(
    *,
    repo_id: str,
    run_manifest: dict[str, Any],
    eval_samples: list[Path] | None = None,
) -> str:
    """Generate a HuggingFace model card README.md from the run manifest."""
    steps = run_manifest.get("steps", {})
    source_models = run_manifest.get("source_models", {})
    run_id = run_manifest.get("run_id", "unknown")
    created_at = run_manifest.get("created_at", utc_now())

    # Source model table
    model_rows = "\n".join(
        f"| `{alias}` | [{meta.get('model_id', alias)}](https://huggingface.co/{meta.get('model_id', alias)}) "
        f"| {meta.get('role', '—')} | {meta.get('license', 'Apache-2.0')} |"
        for alias, meta in source_models.items()
    )

    # Merge step section
    merge = steps.get("merge", {})
    merge_metrics = merge.get("metrics", {})
    merge_method = merge_metrics.get("merge_method", "delta-edit")
    merge_coeff = merge_metrics.get("coefficient", "0.35")
    foundation = merge_metrics.get("foundation_model", "Qwen/Qwen-Image-2512")

    # Abliterate section
    ablate = steps.get("abliterate", {})
    ablate_artifacts = ablate.get("artifacts", [])
    recipe_artifact = next(
        (a for a in ablate_artifacts if a.get("kind") == "recipe_config"), None
    )
    recipe_note = (
        f"Recipe: `{_shorten(recipe_artifact['path_or_uri'])}`"
        if recipe_artifact else "Recipe: `stage-3-abliteration.example.yaml` (default)"
    )

    # Quantize section
    quant = steps.get("quantize", {})
    quant_metrics = quant.get("metrics", {})
    quant_method = quant_metrics.get("quant_method", "gguf + exl2")
    quant_artifacts = quant.get("artifacts", [])
    quant_artifact_rows = "\n".join(
        f"| `{a.get('kind', '')}` | `{_shorten(a.get('path_or_uri', ''))}` |"
        for a in quant_artifacts
    )

    # Sample images section
    sample_section = ""
    if eval_samples:
        sample_lines = "\n".join(
            f"![sample {i}]({s.name})"
            for i, s in enumerate(eval_samples[:6])
        )
        sample_section = f"""
## Sample Outputs

{sample_lines}
"""

    card = f"""---
license: apache-2.0
base_model:
  - Qwen/Qwen-Image-2512
  - Qwen/Qwen-Image-Edit-2511
  - Qwen/Qwen-Image
tags:
  - image-generation
  - qwen
  - mmdit
  - abliterated
  - quantized
  - rocm
language:
  - en
library_name: diffusers
pipeline_tag: text-to-image
---

# {repo_id.split("/")[-1]}

A merged, abliterated, and quantized derivative of the Qwen-Image 20B MMDiT family.

> **Run ID:** `{run_id}`
> **Created:** {created_at}

## Architecture

| Property | Value |
| --- | --- |
| Base family | Qwen-Image (MMDiT 20B) |
| Text encoder | Qwen2.5-VL |
| VAE | RGB-VAE |
| RoPE | 2D |
| Backbone parameters | ~20B |
| License | Apache-2.0 |

## Source Models

| Alias | Model | Role | License |
| --- | --- | --- | --- |
{model_rows}

## Research Method

### 1. Delta-Edit Merge

The edit capability is transferred to the foundation model via a controlled
delta injection:

```
edit_delta = Qwen-Image-Edit-2511 − Qwen-Image (delta base)
merged     = Qwen-Image-2512 + {merge_coeff} × edit_delta
```

Only MMDiT backbone tensors are blended. Text encoder, VAE, and RoPE
components are passed through from the foundation checkpoint unchanged.

- **Strategy:** `{merge_method}`
- **Blend coefficient:** `{merge_coeff}`
- **Foundation:** `{foundation}`
- **Excluded subsystems:** text_encoder, vae, rope

### 2. Abliteration (Refusal-Direction Removal)

Refusal-direction vectors are identified in the residual stream and
projected out of target weight matrices using a norm-preserving
orthogonal projection:

```
W′ = W − scale × (W @ r̂) ⊗ r̂    (norm-preserving variant)
```

- **Target layers:** 18+ (attention o_proj + MLP down_proj)
- **Scale:** 1.0
- **Mode:** norm-preserving (preserves weight magnitude distribution)
- {recipe_note}

### 3. Quantization

| Kind | Path |
| --- | --- |
{quant_artifact_rows if quant_artifact_rows else "| — | Quantization artifacts not yet available |"}

- **GGUF targets:** Q4_K_M, IQ4_XS (with importance-matrix)
- **EXL2 target:** 4.0 bpw
- **Runtime:** vLLM-Omni (ROCm), ExLlamaV2

## Hardware

- **GPU:** AMD Instinct MI300X — 192 GB HBM3 VRAM
- **ROCm:** 7.2.0
- **Precision:** bf16 (merge + abliterate), quantized (deployment)
{sample_section}
## Usage

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "{repo_id}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
pipe = pipe.to("cuda")

image = pipe(
    "a photorealistic portrait of an astronaut on Mars at sunrise",
    num_inference_steps=30,
    guidance_scale=4.0,
).images[0]
image.save("output.png")
```

## License

Apache-2.0 — inherited from all source models.
"""
    return card


# ── Upload helpers ─────────────────────────────────────────────────────

def _upload_with_retry(api: Any, *, max_retries: int = 3, **kwargs: Any) -> Any:
    for attempt in range(max_retries):
        try:
            return api.upload_file(**kwargs)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise PublishError(
                    f"Upload failed after {max_retries} attempts: {exc}"
                ) from exc
            wait = 2 ** attempt
            time.sleep(wait)


def _collect_artifacts(run_dir: Path) -> list[Path]:
    """Collect all publishable artifacts from the run directory."""
    artifacts: list[Path] = []
    # GGUF files
    for p in sorted((run_dir / "quantize").rglob("*.gguf")):
        artifacts.append(p)
    # EXL2 files
    for p in sorted((run_dir / "quantize").rglob("*.safetensors")):
        artifacts.append(p)
    for p in sorted((run_dir / "quantize").rglob("*.json")):
        if "step-result" not in p.name and "eval-summary" not in p.name:
            artifacts.append(p)
    # Sample images
    for p in sorted(run_dir.rglob("samples/*.png")):
        artifacts.append(p)
    return artifacts


# ── Main publish function ──────────────────────────────────────────────

def publish(
    *,
    repo_id: str,
    run_dir: Path,
    run_manifest: dict[str, Any],
    dry_run: bool = False,
    private: bool = False,
    token: str | None = None,
) -> dict[str, Any]:
    """Upload artifacts and model card to HuggingFace.

    Args:
        repo_id: Target HF repository (e.g. "ThirdMiddle/Qwen-Image-1.9").
        run_dir: Local run directory with all artifacts.
        run_manifest: The run manifest dict (for model card generation).
        dry_run: If True, return the plan without uploading.
        private: Create repo as private.
        token: HF token override. Falls back to .env / HF_TOKEN env var.
    """
    resolved_token = token or _load_hf_token()
    artifacts = _collect_artifacts(run_dir)
    eval_samples = [a for a in artifacts if a.suffix == ".png"]

    # Build model card
    model_card = build_model_card(
        repo_id=repo_id,
        run_manifest=run_manifest,
        eval_samples=eval_samples,
    )

    plan = {
        "repo_id": repo_id,
        "run_dir": str(run_dir),
        "artifact_count": len(artifacts),
        "artifacts": [str(a) for a in artifacts],
        "model_card_preview": model_card[:500] + "…",
        "private": private,
    }

    if dry_run:
        return {
            "status": "planned",
            "plan": plan,
            "model_card_preview": model_card,
        }

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise PublishError(
            "huggingface_hub is required for publishing. "
            "Install with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=resolved_token)
    started_at = utc_now()
    t0 = time.perf_counter()

    # Ensure repo exists
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    except Exception as exc:
        raise PublishError(f"Failed to create/access repo `{repo_id}`: {exc}") from exc

    uploaded: list[str] = []
    failed: list[str] = []

    # Upload model card
    try:
        api.upload_file(
            path_or_fileobj=model_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add model card (run {run_manifest.get('run_id', 'unknown')})",
        )
        uploaded.append("README.md")
    except Exception as exc:
        failed.append(f"README.md: {exc}")

    # Upload artifacts
    for artifact in artifacts:
        # Determine in-repo path
        try:
            rel = artifact.relative_to(run_dir)
        except ValueError:
            rel = Path(artifact.name)
        repo_path = str(rel)

        try:
            _upload_with_retry(
                api,
                path_or_fileobj=str(artifact),
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {artifact.name}",
            )
            uploaded.append(repo_path)
        except PublishError as exc:
            failed.append(f"{repo_path}: {exc}")

    duration = time.perf_counter() - t0
    repo_url = f"https://huggingface.co/{repo_id}"
    status = "succeeded" if not failed else ("partial" if uploaded else "failed")

    return {
        "status": status,
        "repo_id": repo_id,
        "repo_url": repo_url,
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": round(duration, 2),
        "uploaded": uploaded,
        "failed": failed,
        "artifact_count": len(uploaded),
    }
