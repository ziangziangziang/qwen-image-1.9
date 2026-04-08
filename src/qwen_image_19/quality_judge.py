"""Quality judge using Qwen/Qwen3.5-35B-A3B as a vision-language evaluator.

Scores generated images on a 1-10 scale across four dimensions:
  - prompt_adherence   : does the image faithfully represent the prompt?
  - visual_quality     : sharpness, coherence, absence of artifacts
  - detail_fidelity    : fine-grained detail preservation (esp. for editing/layering)
  - aesthetic_score    : overall compositional and stylistic quality

For editing tasks, also scores:
  - instruction_adherence  : was the edit instruction followed correctly?
  - non_edit_preservation  : are non-edited regions unchanged?

For layering tasks, also scores:
  - alpha_quality          : is the alpha channel clean and accurate?
  - foreground_fidelity    : foreground subject quality
  - background_cleanness   : background isolation quality

Usage:
    judge = QualityJudge()
    result = judge.score_generation(image_path, prompt)
    result = judge.score_edit(image_path, source_prompt, instruction)
    result = judge.score_layer(image_path, prompt)
    batch  = judge.score_batch([...])
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

from qwen_image_19.contracts import utc_now

# ── Constants ─────────────────────────────────────────────────────────────────

JUDGE_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
JUDGE_DTYPE = "bfloat16"
SCORE_SCALE = (1, 10)
PASS_THRESHOLD = 6.5
REGRESSION_TOLERANCE = 0.5

_GENERATION_SYSTEM = """You are an expert image quality evaluator. You will be shown a generated image and the text prompt that was used to create it. Score the image on each dimension from 1 (very poor) to 10 (excellent).

Return ONLY valid JSON with this exact structure:
{
  "prompt_adherence": <1-10>,
  "visual_quality": <1-10>,
  "aesthetic_score": <1-10>,
  "overall": <1-10>,
  "reasoning": "<one sentence per dimension, semicolon-separated>"
}"""

_EDITING_SYSTEM = """You are an expert image editing evaluator. You will be shown an edited image, the original description, and the edit instruction. Score on each dimension from 1 to 10.

Return ONLY valid JSON with this exact structure:
{
  "instruction_adherence": <1-10>,
  "non_edit_preservation": <1-10>,
  "visual_quality": <1-10>,
  "aesthetic_score": <1-10>,
  "overall": <1-10>,
  "reasoning": "<one sentence per dimension, semicolon-separated>"
}"""

_LAYERING_SYSTEM = """You are an expert in image compositing and alpha channel evaluation. You will be shown a layered/RGBA image or a composite alongside the prompt. Score on each dimension from 1 to 10.

Return ONLY valid JSON with this exact structure:
{
  "alpha_quality": <1-10>,
  "foreground_fidelity": <1-10>,
  "background_cleanness": <1-10>,
  "composite_naturalness": <1-10>,
  "overall": <1-10>,
  "reasoning": "<one sentence per dimension, semicolon-separated>"
}"""

_REGRESSION_SYSTEM = """You are comparing two images generated for the same prompt: a REFERENCE (baseline model) and a CANDIDATE (merged/modified model). Score the CANDIDATE relative to the REFERENCE on a 1-10 scale. 10 = candidate is indistinguishable from or better than reference.

Return ONLY valid JSON:
{
  "prompt_adherence": <1-10>,
  "visual_quality": <1-10>,
  "aesthetic_score": <1-10>,
  "regression_delta": <float, positive means candidate is better>,
  "overall": <1-10>,
  "reasoning": "<brief comparison>"
}"""


# ── Image encoding ────────────────────────────────────────────────────────────

def _encode_image(image_path: str | Path) -> str:
    """Return base64-encoded image for multimodal input."""
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _pil_to_b64(image) -> str:
    """Convert a PIL image to base64 PNG."""
    import io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Model loading ─────────────────────────────────────────────────────────────

class QualityJudge:
    """Qwen3.5-35B-A3B quality judge for image evaluation.

    Loaded lazily on first use to avoid importing heavy deps at module load time.

    Args:
        model_id: HuggingFace model ID (defaults to Qwen/Qwen3.5-35B-A3B).
        device: Torch device to load on (defaults to "cuda").
        dtype: Torch dtype string ("bfloat16", "float16", "float32").
        max_new_tokens: Maximum tokens the judge can generate per response.
    """

    def __init__(
        self,
        model_id: str = JUDGE_MODEL_ID,
        device: str = "cuda",
        dtype: str = JUDGE_DTYPE,
        max_new_tokens: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        print(f"[judge] loading {self.model_id} ({self.dtype}) …", flush=True)
        t0 = time.time()
        self._processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()
        print(f"[judge] loaded in {time.time() - t0:.1f}s", flush=True)

    def _infer(self, system_prompt: str, user_message: str, images: list) -> dict[str, Any]:
        """Run a multimodal judge inference and parse JSON response."""
        import torch

        self._load()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": []},
        ]
        # Add images to user content
        for img in images:
            messages[1]["content"].append({"type": "image", "image": img})
        messages[1]["content"].append({"type": "text", "text": user_message})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=False,
            )
        # Trim the prompt tokens
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        response_text = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        # Parse JSON from response
        return _parse_json_response(response_text)

    # ── Public scoring API ────────────────────────────────────────────────────

    def score_generation(
        self,
        image: str | Path | Any,
        prompt: str,
    ) -> dict[str, Any]:
        """Score a generated image against its text prompt.

        Args:
            image: Path to a PNG/JPEG file, or a PIL Image.
            prompt: The text prompt used to generate the image.

        Returns:
            Dict with keys: prompt_adherence, visual_quality, aesthetic_score,
            overall, reasoning, passed (bool), scored_at.
        """
        pil_img = _load_pil(image)
        user_msg = f'Prompt: "{prompt}"\n\nPlease evaluate the generated image.'
        scores = self._infer(_GENERATION_SYSTEM, user_msg, [pil_img])
        scores["passed"] = scores.get("overall", 0) >= PASS_THRESHOLD
        scores["scored_at"] = utc_now()
        scores["task"] = "generation"
        scores["prompt"] = prompt
        return scores

    def score_edit(
        self,
        image: str | Path | Any,
        source_description: str,
        instruction: str,
    ) -> dict[str, Any]:
        """Score an edited image against the edit instruction.

        Args:
            image: Path or PIL Image of the edited output.
            source_description: Description of the original/input image.
            instruction: The edit instruction that was applied.

        Returns:
            Dict with keys: instruction_adherence, non_edit_preservation,
            visual_quality, aesthetic_score, overall, reasoning, passed.
        """
        pil_img = _load_pil(image)
        user_msg = (
            f'Original image description: "{source_description}"\n'
            f'Edit instruction: "{instruction}"\n\n'
            "Please evaluate how well the edit was applied."
        )
        scores = self._infer(_EDITING_SYSTEM, user_msg, [pil_img])
        scores["passed"] = scores.get("overall", 0) >= PASS_THRESHOLD
        scores["scored_at"] = utc_now()
        scores["task"] = "editing"
        scores["instruction"] = instruction
        return scores

    def score_layer(
        self,
        image: str | Path | Any,
        prompt: str,
    ) -> dict[str, Any]:
        """Score a layered/RGBA image for compositing quality.

        Args:
            image: Path or PIL Image (RGB composite or RGBA rendered as RGBA preview).
            prompt: The prompt or description.

        Returns:
            Dict with keys: alpha_quality, foreground_fidelity, background_cleanness,
            composite_naturalness, overall, reasoning, passed.
        """
        pil_img = _load_pil(image)
        user_msg = (
            f'Prompt: "{prompt}"\n\n'
            "Evaluate the quality of subject isolation, alpha channel, and compositing."
        )
        scores = self._infer(_LAYERING_SYSTEM, user_msg, [pil_img])
        scores["passed"] = scores.get("overall", 0) >= PASS_THRESHOLD
        scores["scored_at"] = utc_now()
        scores["task"] = "layering"
        scores["prompt"] = prompt
        return scores

    def score_regression(
        self,
        candidate_image: str | Path | Any,
        reference_image: str | Path | Any,
        prompt: str,
    ) -> dict[str, Any]:
        """Compare a candidate image to a reference baseline image.

        Used for regression gates after merge / abliterate / quantize.

        Returns:
            Dict including regression_delta (positive = candidate is better),
            overall, passed (True if regression_delta >= -REGRESSION_TOLERANCE).
        """
        cand_pil = _load_pil(candidate_image)
        ref_pil = _load_pil(reference_image)
        user_msg = (
            f'Prompt: "{prompt}"\n\n'
            "The FIRST image is the REFERENCE (baseline). "
            "The SECOND image is the CANDIDATE (merged/modified model). "
            "Score the candidate relative to the reference."
        )
        scores = self._infer(_REGRESSION_SYSTEM, user_msg, [ref_pil, cand_pil])
        delta = scores.get("regression_delta", 0.0)
        scores["passed"] = float(delta) >= -REGRESSION_TOLERANCE
        scores["scored_at"] = utc_now()
        scores["task"] = "regression"
        scores["prompt"] = prompt
        return scores

    def score_batch(
        self,
        items: list[dict[str, Any]],
        on_result: "Any | None" = None,
    ) -> list[dict[str, Any]]:
        """Score a batch of items.

        Each item must have:
          - "task": "generation" | "editing" | "layering" | "regression"
          - "image": path or PIL
          - "prompt": text prompt (generation/layering/regression)
          - "source_description" + "instruction" (editing)
          - "reference_image" (regression)

        Args:
            items: List of item dicts.
            on_result: Optional callback called with each result as it's produced.

        Returns:
            List of score dicts in the same order as *items*.
        """
        results: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            task = item.get("task", "generation")
            try:
                if task == "generation":
                    r = self.score_generation(item["image"], item["prompt"])
                elif task == "editing":
                    r = self.score_edit(
                        item["image"],
                        item.get("source_description", ""),
                        item["instruction"],
                    )
                elif task == "layering":
                    r = self.score_layer(item["image"], item["prompt"])
                elif task == "regression":
                    r = self.score_regression(
                        item["image"],
                        item["reference_image"],
                        item["prompt"],
                    )
                else:
                    r = {"error": f"unknown task '{task}'", "task": task}
            except Exception as exc:
                r = {"error": str(exc), "task": task, "index": i}
            results.append(r)
            if on_result is not None:
                on_result(i, r)
        return results

    def unload(self) -> None:
        """Release GPU memory."""
        import torch
        self._model = None
        self._processor = None
        torch.cuda.empty_cache()


# ── Batch eval helpers ────────────────────────────────────────────────────────

def run_regression_suite(
    *,
    judge: QualityJudge,
    suite_config: list[dict[str, Any]],
    candidate_model_fn: "Any",
    reference_cache_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    steps: int = 30,
    side: int = 1024,
) -> dict[str, Any]:
    """Run the fixed regression suite from the eval dataset config.

    For each entry in *suite_config*:
      1. Generate from the candidate model (or load from cache if --resume).
      2. Load/generate reference image from reference_cache_dir.
      3. Score with the judge.
      4. Aggregate pass/fail.

    Returns a summary dict with per-item scores and aggregate gate result.
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    gate_passed = True

    for item in suite_config:
        item_id = item.get("id", "unknown")
        task = item.get("task", "generation")
        prompt = item.get("prompt") or item.get("source_prompt", "")
        seed = item.get("seed", 2025)

        # Generate candidate image
        cand_path = output_dir / f"{item_id}-candidate.png"
        if not cand_path.exists():
            try:
                from diffusers import DiffusionPipeline
                cand_pipe = candidate_model_fn()
                gen = torch.Generator(device=device).manual_seed(seed)
                out = cand_pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=4.0,
                    height=side,
                    width=side,
                    generator=gen,
                )
                out.images[0].save(str(cand_path))
            except Exception as e:
                results.append({"id": item_id, "error": str(e), "passed": False})
                gate_passed = False
                continue

        # Load or generate reference image
        ref_path = reference_cache_dir / f"{item_id}-reference.png"

        # Score
        try:
            score = judge.score_regression(cand_path, ref_path, prompt)
            score["id"] = item_id
            results.append(score)
            if not score.get("passed", True):
                gate_passed = False
        except Exception as e:
            results.append({"id": item_id, "error": str(e), "passed": False})
            gate_passed = False

    # Aggregate
    scored = [r for r in results if "overall" in r]
    avg_overall = sum(r["overall"] for r in scored) / len(scored) if scored else 0.0
    avg_delta = sum(r.get("regression_delta", 0) for r in scored) / len(scored) if scored else 0.0

    summary = {
        "gate_passed": gate_passed,
        "total": len(results),
        "passed": sum(1 for r in results if r.get("passed", False)),
        "failed": sum(1 for r in results if not r.get("passed", True)),
        "avg_overall": round(avg_overall, 3),
        "avg_regression_delta": round(avg_delta, 3),
        "items": results,
        "scored_at": utc_now(),
    }

    # Write report
    (output_dir / "regression-suite-results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def aggregate_scores(score_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-dimension means from a list of judge score dicts."""
    if not score_list:
        return {"error": "no scores to aggregate"}

    all_keys = set()
    for s in score_list:
        all_keys.update(k for k, v in s.items() if isinstance(v, (int, float)))

    agg: dict[str, Any] = {}
    for key in sorted(all_keys):
        vals = [s[key] for s in score_list if key in s and isinstance(s[key], (int, float))]
        if vals:
            agg[f"{key}_mean"] = round(sum(vals) / len(vals), 3)
            agg[f"{key}_min"] = round(min(vals), 3)
            agg[f"{key}_max"] = round(max(vals), 3)

    agg["n"] = len(score_list)
    agg["pass_rate"] = round(sum(1 for s in score_list if s.get("passed", False)) / len(score_list), 3)
    return agg


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_pil(image: Any) -> Any:
    """Coerce a path / PIL image / bytes to a PIL RGB Image."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow required: pip install Pillow") from exc

    if isinstance(image, (str, Path)):
        return Image.open(str(image)).convert("RGB")
    # Already PIL
    if hasattr(image, "save"):
        return image.convert("RGB")
    # Bytes
    import io
    return Image.open(io.BytesIO(image)).convert("RGB")


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract the first JSON object from the judge's text response."""
    text = text.strip()
    # Find the JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {"error": "no JSON in response", "raw_response": text[:500]}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse error: {exc}", "raw_response": text[:500]}
