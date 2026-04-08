"""Stage 3 evaluation worker — runs generation eval for a given checkpoint.

Produces a set of sample images and quantitative metrics via the
Qwen3.5-35B-A3B quality judge (prompt_adherence, visual_quality,
aesthetic_score, overall 1-10 scale).

Validation and test prompts are loaded from:
  configs/eval/stage-3-eval-datasets.yaml

Falls back to hardcoded prompts if the config or datasets are unavailable.
Requires torch + diffusers + transformers.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root
from qwen_image_19.contracts import utc_now


class EvalWorkerError(RuntimeError):
    """Raised when eval worker encounters a fatal error."""


# ── Dataset config loading ────────────────────────────────────────────────────

def _load_eval_config() -> dict[str, Any]:
    path = repo_root() / "configs" / "eval" / "stage-3-eval-datasets.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _regression_suite_prompts() -> list[dict[str, Any]]:
    cfg = _load_eval_config()
    return cfg.get("splits", {}).get("regression_suite", {}).get("fixed_prompts", [])


def _resolve_judge_model_id() -> str:
    """Return the abliterated judge checkpoint when available, else the default model ID.

    Reads ``judge.checkpoint`` from the eval config. If that path exists on
    disk it takes precedence over ``judge.model_id`` so that post-prepare-judge
    runs automatically benefit from the unfiltered judge without any flag change.
    """
    from qwen_image_19.quality_judge import JUDGE_MODEL_ID
    cfg = _load_eval_config()
    judge_cfg = cfg.get("judge", {})
    checkpoint = judge_cfg.get("checkpoint", "")
    if checkpoint and Path(checkpoint).exists():
        return checkpoint
    return judge_cfg.get("model_id", JUDGE_MODEL_ID)


def _val_prompts_for_step(step: str) -> list[str]:
    cfg = _load_eval_config()
    split_map = {
        "merge": "generation_val",
        "post_merge_train": "generation_val",
        "abliterate": "generation_val",
        "post_abliterate_train": "generation_val",
        "quantize": "generation_val",
    }
    split_key = split_map.get(step, "generation_val")
    split = cfg.get("splits", {}).get(split_key, {})
    prompts: list[str] = []
    try:
        from datasets import load_dataset  # type: ignore
        for ds_spec in split.get("datasets", []):
            want = min(ds_spec.get("sample_count", 50), 50)  # cap per dataset in worker
            try:
                ds = load_dataset(ds_spec["hf_id"], split=ds_spec["split"], streaming=True)
                col = ds_spec.get("prompt_column", "prompt")
                for row in ds:
                    val = row.get(col)
                    if val and isinstance(val, str) and val.strip():
                        prompts.append(val.strip())
                    if len(prompts) >= want:
                        break
            except Exception:
                pass
    except ImportError:
        pass
    # Always fall back to hardcoded set if we got nothing
    if not prompts:
        prompts = _EVAL_PROMPTS_FALLBACK
    return prompts


_EVAL_PROMPTS_FALLBACK = [
    "a photorealistic portrait of an astronaut on Mars at sunrise",
    "a watercolor painting of a rainy Tokyo street at night",
    "a detailed pencil sketch of a Victorian clockwork machine",
    "a vibrant oil painting of a tropical coral reef",
    "an architectural render of a futuristic glass skyscraper",
    "a fantasy illustration of a dragon guarding a mountain fortress",
    "a macro photograph of a dewdrop on a leaf",
    "a minimalist line drawing of a cat sitting on a windowsill",
]


# ── Generation eval ───────────────────────────────────────────────────────────

def run_generation_eval(
    *,
    model_id: str,
    sample_dir: Path,
    num_prompts: int = 8,
    seed: int = 2025,
    device: str = "cuda",
    guidance_scale: float = 4.0,
    num_inference_steps: int = 30,
    height: int = 512,
    width: int = 512,
    use_judge: bool = True,
    step: str = "merge",
) -> dict[str, Any]:
    """Run image generation eval on *model_id* and write sample PNGs.

    When *use_judge* is True, scores each image via Qwen3.5-35B-A3B and
    includes aggregate judge metrics in the result.

    Returns a result dict with:
        status, sample_files, metrics, judge_scores, started_at, ended_at
    """
    try:
        import torch
        from diffusers import DiffusionPipeline  # noqa: F401
    except ImportError as exc:
        raise EvalWorkerError(
            "Eval worker requires torch and diffusers. "
            f"Install them with: pip install torch diffusers\n{exc}"
        ) from exc

    sample_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    t0 = time.perf_counter()

    # Load validation prompts from dataset config
    all_prompts = _val_prompts_for_step(step)
    prompts = all_prompts[:num_prompts]

    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        generator = torch.Generator(device=device).manual_seed(seed)
        sample_files: list[str] = []
        generated_images: list[Any] = []

        for i, prompt in enumerate(prompts):
            out = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            img = out.images[0]
            fname = f"sample_{i:03d}.png"
            img.save(sample_dir / fname)
            sample_files.append(fname)
            generated_images.append((img, prompt))

        del pipe
        torch.cuda.empty_cache()

    except Exception as exc:
        raise EvalWorkerError(f"Eval generation failed: {exc}") from exc

    duration = time.perf_counter() - t0

    # ── Judge scoring ──────────────────────────────────────────────────────
    judge_scores: list[dict[str, Any]] = []
    judge_aggregate: dict[str, Any] = {}

    if use_judge and generated_images:
        try:
            from qwen_image_19.quality_judge import QualityJudge, aggregate_scores
            judge = QualityJudge(model_id=_resolve_judge_model_id(), device=device)
            for img, prompt in generated_images:
                score = judge.score_generation(img, prompt)
                judge_scores.append(score)
            judge_aggregate = aggregate_scores(judge_scores)
            judge.unload()
        except Exception as exc:
            judge_aggregate = {"error": str(exc), "note": "judge scoring failed — using raw metrics only"}

    return {
        "status": "completed",
        "started_at": started_at,
        "ended_at": utc_now(),
        "duration_seconds": round(duration, 2),
        "sample_files": sample_files,
        "judge_scores": judge_scores,
        "judge_aggregate": judge_aggregate,
        "metrics": {
            "num_prompts": num_prompts,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "samples_written": len(sample_files),
            **{k: v for k, v in judge_aggregate.items() if not isinstance(v, dict)},
        },
    }


# ── Regression suite eval ─────────────────────────────────────────────────────

def run_regression_eval(
    *,
    candidate_checkpoint: str,
    reference_cache_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    steps: int = 30,
    side: int = 1024,
) -> dict[str, Any]:
    """Run the fixed 9-prompt regression suite with judge scoring.

    Generates images from *candidate_checkpoint*, compares to cached
    reference images, scores with Qwen3.5-35B-A3B, returns gate result.
    """
    suite = _regression_suite_prompts()
    if not suite:
        return {"gate_passed": True, "note": "regression suite config not found — skipped"}

    try:
        import torch
        from diffusers import DiffusionPipeline
        from qwen_image_19.quality_judge import QualityJudge, run_regression_suite

        output_dir.mkdir(parents=True, exist_ok=True)

        pipe = DiffusionPipeline.from_pretrained(
            candidate_checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        def candidate_model_fn():
            return pipe

        judge = QualityJudge(model_id=_resolve_judge_model_id(), device=device)
        result = run_regression_suite(
            judge=judge,
            suite_config=suite,
            candidate_model_fn=candidate_model_fn,
            reference_cache_dir=reference_cache_dir,
            output_dir=output_dir,
            device=device,
            steps=steps,
            side=side,
        )
        judge.unload()
        del pipe
        torch.cuda.empty_cache()
        return result
    except Exception as exc:
        return {"gate_passed": False, "error": str(exc)}


# ── Step dispatcher ───────────────────────────────────────────────────────────

def run_eval_for_step(
    *,
    step: str,
    checkpoint: str,
    run_dir: Path,
    num_prompts: int = 8,
    device: str = "cuda",
    use_judge: bool = True,
) -> dict[str, Any]:
    """Run eval for a specific pipeline step and return structured result."""
    sample_dir = run_dir / step / "samples"
    return run_generation_eval(
        model_id=checkpoint,
        sample_dir=sample_dir,
        num_prompts=num_prompts,
        device=device,
        use_judge=use_judge,
        step=step,
    )

