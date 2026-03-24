from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import re
import sys
from pathlib import Path
from typing import Any

from qwen_image_19.config_io import load_json, repo_root


STAGE1_ARTIFACT_DIR = Path("reports") / "stage-1"
DEFAULT_STAGE2_ARTIFACT_DIR = Path("reports") / "stage-2"
DEFAULT_STAGE2_RUN_STATUS = Path("stage-2") / "run-status.json"
CORE_CANDIDATE_DEFAULT_WEIGHT = 0.35
DEFAULT_RUN_PROFILE = "full"
SUPPORTED_RUN_PROFILES = {"smoke", "full", "quality"}

def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, ss = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {ss:02d}s"
    return f"{ss}s"


def _describe_job(job_name: str, manifest: dict[str, Any]) -> list[str]:
    """Return a list of detail lines describing what a job will do, drawn from the manifest."""
    limits = manifest.get("limits", {})
    poc_steps = int(limits.get("poc_steps", 6))
    poc_side = int(limits.get("poc_side", 512))
    cfg = float(limits.get("poc_true_cfg_scale", 4.0))
    guidance = float(limits.get("poc_guidance_scale", 1.0))
    _d = "  "  # indent prefix for detail lines

    if job_name == "core_delta_sweep":
        recipe = manifest.get("core_delta_recipe", {})
        candidates = manifest.get("core_delta_candidates", [])
        lines = [
            f"{_d}task       : build edit-delta checkpoints via coefficient sweep",
            f"{_d}foundation : {recipe.get('foundation_model', '?')}",
            f"{_d}edit src   : {recipe.get('delta_source_model', '?')}",
            f"{_d}candidates : {len(candidates)}",
        ]
        for c in candidates:
            lines.append(f"{_d}             {c['candidate_id']}  blend={c['blend_weight']}  -> {c['output_checkpoint']}")
        lines.append(f"{_d}diffusion  : steps={poc_steps}  side={poc_side}  cfg={cfg:g}  guidance={guidance:g}")
        return lines

    if job_name == "core_smoke_eval":
        sel = manifest.get("selected_core_candidate", {})
        recipe = manifest.get("core_delta_recipe", {})
        n_prompts = int(limits.get("eval_prompt_count", 6))
        return [
            f"{_d}task       : smoke quality check on best core delta checkpoint",
            f"{_d}model      : {recipe.get('foundation_model', '?')}",
            f"{_d}checkpoint : {sel.get('output_checkpoint', '?')}",
            f"{_d}prompts    : {n_prompts}",
            f"{_d}diffusion  : steps={poc_steps}  side={poc_side}  cfg={cfg:g}  guidance={guidance:g}",
            f"{_d}output     : {sel.get('smoke_report', '?')}",
        ]

    if job_name == "teacher_dataset_generation":
        dataset = manifest.get("dataset", {})
        split_counts: dict[str, int] = dataset.get("split_counts", {})
        total_samples = sum(split_counts.values())
        lines = [
            f"{_d}task       : generate synthetic teacher dataset",
            f"{_d}manifest   : {dataset.get('manifest_path', '?')}",
            f"{_d}output dir : {dataset.get('output_root', '?')}",
            f"{_d}splits     : {len(split_counts)}  ({total_samples} samples total)",
        ]
        for split_name, count in split_counts.items():
            lines.append(f"{_d}             {split_name}  {count} samples")
        lines.append(f"{_d}diffusion  : steps={poc_steps}  side={poc_side}  cfg={cfg:g}  guidance={guidance:g}")
        return lines

    if job_name == "layered_bridge_train":
        recipe = manifest.get("layered_bridge_recipe", {})
        training = recipe.get("training_limits", {})
        max_steps = int(training.get("max_steps", 500))
        batch_size = int(training.get("batch_size", 1))
        block_window = recipe.get("bridge_block_window", "?")
        freeze_policy = recipe.get("freeze_policy", "?")
        trainable = recipe.get("trainable_modules", [])
        dataset = manifest.get("dataset", {})
        return [
            f"{_d}task       : train layered bridge adapter",
            f"{_d}strategy   : {recipe.get('strategy', '?')}",
            f"{_d}donor      : {recipe.get('donor_model', '?')}",
            f"{_d}base ckpt  : {recipe.get('base_core_checkpoint', '?')}",
            f"{_d}dataset    : {dataset.get('output_root', '?')}",
            f"{_d}training   : steps={max_steps}  batch={batch_size}",
            f"{_d}blocks     : {block_window}",
            f"{_d}freeze     : {freeze_policy}",
            f"{_d}trainable  : {', '.join(trainable) if trainable else '(none)'}",
            f"{_d}adapter out: {recipe.get('output_adapter', '?')}",
            f"{_d}ckpt out   : {recipe.get('output_checkpoint', '?')}",
        ]

    if job_name == "experimental_smoke_eval":
        recipe = manifest.get("layered_bridge_recipe", {})
        n_prompts = int(limits.get("eval_prompt_count", 6))
        return [
            f"{_d}task       : smoke eval on layered bridge checkpoint",
            f"{_d}foundation : {recipe.get('foundation_model', '?')}  (dispatched via QwenImageLayeredPipeline)",
            f"{_d}bridge ckpt: {recipe.get('output_checkpoint', '?')}",
            f"{_d}prompts    : {n_prompts}",
            f"{_d}diffusion  : steps={poc_steps}  side={poc_side}  cfg={cfg:g}  guidance={guidance:g}",
        ]

    if job_name == "core_edit_eval":
        sel = manifest.get("selected_core_candidate", {})
        recipe = manifest.get("core_delta_recipe", {})
        n_pairs = int(limits.get("eval_edit_prompt_count", 3))
        return [
            f"{_d}task       : edit capability eval — before/after image pairs",
            f"{_d}merged ckpt: {sel.get('output_checkpoint', '?')}",
            f"{_d}foundation : {recipe.get('foundation_model', '?')}  (generates 'before' images)",
            f"{_d}merged     : {recipe.get('foundation_model', '?')}  (generates 'after' images)",
            f"{_d}pairs      : {n_pairs}",
            f"{_d}output     : stage-2/evals/core-edit/edit-summary.json",
            f"{_d}diffusion  : steps={poc_steps}  side={poc_side}  cfg={cfg:g}  guidance={guidance:g}",
        ]

    if job_name == "consistency_eval":
        sel = manifest.get("selected_core_candidate", {})
        recipe = manifest.get("core_delta_recipe", {})
        n_prompts = int(limits.get("consistency_eval_prompt_count", 4))
        return [
            f"{_d}task       : consistency eval — pixel-L2 drift between foundation and merged",
            f"{_d}baseline   : {recipe.get('foundation_model', '?')}",
            f"{_d}merged     : {recipe.get('foundation_model', '?')}  + checkpoint {sel.get('output_checkpoint', '?')}",
            f"{_d}prompts    : {n_prompts}  (fixed seeds for reproducibility)",
            f"{_d}output     : stage-2/evals/consistency/consistency-summary.json",
            f"{_d}diffusion  : steps={poc_steps}  side={poc_side}  cfg={cfg:g}  guidance={guidance:g}",
        ]

    return []


def _emit_progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Stage2FusionError(RuntimeError):
    """Raised when Stage 2 does not have enough evidence to build a fusion plan."""


def stage2_remote_path(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part]
    return "/".join(cleaned)


def repo_relative_path(path: Path) -> str:
    try:
        return path.relative_to(repo_root()).as_posix()
    except ValueError:
        return path.name


def stage2_artifact_paths(target_dir: Path) -> dict[str, Path]:
    return {
        "artifact_dir": target_dir,
        "report_readme": target_dir / "README.md",
        "merge_manifest_json": target_dir / "merge-manifest.json",
        "dataset_manifest_json": target_dir / "dataset-manifest.json",
        "run_status_json": repo_root() / DEFAULT_STAGE2_RUN_STATUS,
    }


def validate_run_options(dry_run: bool, smoke_run: bool, run_profile: str | None) -> str:
    if dry_run and smoke_run:
        raise Stage2FusionError("`--dry-run` and `--smoke-run` cannot be used together.")
    if run_profile and run_profile not in SUPPORTED_RUN_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_RUN_PROFILES))
        raise Stage2FusionError(f"Unsupported run profile `{run_profile}`. Choose from: {supported}.")
    if smoke_run:
        return "smoke"
    return run_profile or DEFAULT_RUN_PROFILE


def load_run_profiles(config_dir: Path) -> dict[str, Any]:
    payload = load_json(config_dir / "stage-2-run-profiles.yaml")
    profiles = payload.get("profiles", {})
    for name in ("smoke", "full", "quality"):
        if name not in profiles:
            raise Stage2FusionError(
                "Stage 2 run profile config must define `smoke`, `full`, and `quality` profiles."
            )
    return profiles


def parse_candidate_weight(candidate_id: str) -> float:
    match = re.fullmatch(r"core-delta-w(\d+)", candidate_id)
    if not match:
        raise Stage2FusionError(f"Invalid core candidate id `{candidate_id}`.")
    return int(match.group(1)) / 100.0


def build_stage2_compatibility_shims(target_dir: Path) -> dict[str, Path]:
    if target_dir.name == "stage-2":
        compat_dir = target_dir.parent
        return {
            "legacy_report_md": compat_dir / "stage-2-fusion-report.md",
            "legacy_merge_manifest_json": compat_dir / "stage-2-merge-manifest.json",
        }
    return {}


def render_stage2_compatibility_stub(target_dir: Path) -> str:
    return f"""# Stage 2 Fusion Report

Canonical Stage 2 report: [stage-2/README.md](stage-2/README.md)

This file is kept as a compatibility shim. Open `{target_dir / 'README.md'}` for the full Stage 2 report.
"""


def require_stage1_artifacts() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    matrix_path = repo_root() / STAGE1_ARTIFACT_DIR / "compatibility-matrix.json"
    weight_path = repo_root() / STAGE1_ARTIFACT_DIR / "weight-analysis.json"
    missing = [repo_relative_path(path) for path in (matrix_path, weight_path) if not path.exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise Stage2FusionError(
            "Stage 2 requires canonical Stage 1 artifacts. "
            f"Missing: {missing_list}. Run `q19 stage1 analyze` first."
        )
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    weight_analysis = json.loads(weight_path.read_text(encoding="utf-8"))
    return (
        matrix,
        weight_analysis,
        {
            "compatibility_matrix": repo_relative_path(matrix_path),
            "weight_analysis": repo_relative_path(weight_path),
        },
    )


def find_subsystem(matrix: dict[str, Any], subsystem: str) -> dict[str, Any]:
    for item in matrix.get("subsystems", []):
        if item.get("subsystem") == subsystem:
            return item
    raise Stage2FusionError(f"Stage 1 matrix is missing subsystem evidence for `{subsystem}`.")


def extract_transformer_block_ids(payload: dict[str, Any]) -> list[int]:
    ids: set[int] = set()
    for item in payload.get("top_divergent_blocks", []):
        layer_id = str(item.get("layer_id", ""))
        match = re.fullmatch(r"mmdit_backbone:transformer_blocks:(\d+)", layer_id)
        if match:
            ids.add(int(match.group(1)))
    return sorted(ids)


def top_mismatch_prefixes(matrix: dict[str, Any], pair_name: str, prefix: str) -> list[str]:
    pairwise = matrix.get("pairwise_comparisons", {}).get(pair_name, {})
    matches = []
    for item in pairwise.get("top_mismatching_prefixes", []):
        item_prefix = str(item.get("prefix", ""))
        if item_prefix.startswith(prefix):
            matches.append(item_prefix)
    return matches


def build_stage1_evidence(
    matrix: dict[str, Any],
    weight_analysis: dict[str, Any],
    stage1_refs: dict[str, str],
) -> dict[str, Any]:
    foundation_vs_edit = weight_analysis["foundation_vs_edit"]
    foundation_vs_layered = weight_analysis["foundation_vs_layered"]
    base_vs_layered = weight_analysis["base_vs_layered"]
    vae_subsystem = find_subsystem(matrix, "vae")
    rope_subsystem = find_subsystem(matrix, "rope")
    mmdit_subsystem = find_subsystem(matrix, "mmdit_backbone")
    foundation_hot_blocks = extract_transformer_block_ids(foundation_vs_edit)
    layered_hot_blocks = extract_transformer_block_ids(foundation_vs_layered)
    return {
        "artifact_refs": stage1_refs,
        "foundation_vs_edit": {
            "pair": list(foundation_vs_edit["models"]),
            "shared_key_count": foundation_vs_edit["shared_key_count"],
            "exact_equal_tensor_ratio": foundation_vs_edit["exact_equal_tensor_ratio"],
            "transformer_structural_compatibility": mmdit_subsystem["structural_compatibility"],
            "transformer_merge_strategy": mmdit_subsystem["recommended_merge_strategy"],
            "text_encoder_exact_match": foundation_vs_edit["by_subsystem"]["text_encoder"]["mean_exact_tensor_match_ratio"] == 1.0,
            "vae_exact_match": foundation_vs_edit["by_subsystem"]["vae"]["mean_exact_tensor_match_ratio"] == 1.0,
            "transformer_mean_block_relative_l2_delta": foundation_vs_edit["by_subsystem"]["mmdit_backbone"]["mean_block_relative_l2_delta"],
            "observed_hot_blocks": foundation_hot_blocks,
        },
        "layered_vs_core": {
            "pair": list(foundation_vs_layered["models"]),
            "shared_key_count": foundation_vs_layered["shared_key_count"],
            "exact_equal_tensor_ratio": foundation_vs_layered["exact_equal_tensor_ratio"],
            "text_encoder_exact_match": foundation_vs_layered["by_subsystem"]["text_encoder"]["mean_exact_tensor_match_ratio"] == 1.0,
            "transformer_mean_block_relative_l2_delta": foundation_vs_layered["by_subsystem"]["mmdit_backbone"]["mean_block_relative_l2_delta"],
            "observed_hot_blocks": layered_hot_blocks,
            "layered_transformer_mismatch_prefixes": top_mismatch_prefixes(matrix, "foundation_vs_layered", "transformer."),
            "layered_conditioning_hint": "transformer.time_text_embed.addition_t_embedding.weight",
        },
        "layered_conflicts": {
            "base_pair": list(base_vs_layered["models"]),
            "vae": {
                "structural_compatibility": vae_subsystem["structural_compatibility"],
                "base_label": matrix["model_summaries"]["qwen-image-base"]["vae"]["label"],
                "layered_label": matrix["model_summaries"]["qwen-image-layered"]["vae"]["label"],
                "shape_mismatch_count": vae_subsystem["evidence"]["shape_mismatch_count"],
            },
            "rope": {
                "structural_compatibility": rope_subsystem["structural_compatibility"],
                "foundation_label": matrix["model_summaries"]["qwen-image-2512"]["rope"]["label"],
                "layered_label": matrix["model_summaries"]["qwen-image-layered"]["rope"]["label"],
            },
        },
    }


def build_exclusions(matrix: dict[str, Any], weight_analysis: dict[str, Any]) -> dict[str, Any]:
    foundation_vs_edit = weight_analysis["foundation_vs_edit"]
    foundation_vs_layered = weight_analysis["foundation_vs_layered"]
    vae_subsystem = find_subsystem(matrix, "vae")
    rope_subsystem = find_subsystem(matrix, "rope")
    return {
        "text_encoder": {
            "decision": "skip-direct-merge",
            "reason": "Stage 1 value analysis shows exact text-encoder parity, so Stage 2 does not spend merge budget on a no-op subsystem.",
            "evidence": {
                "foundation_vs_edit_exact_ratio": foundation_vs_edit["by_subsystem"]["text_encoder"]["mean_exact_tensor_match_ratio"],
                "foundation_vs_layered_exact_ratio": foundation_vs_layered["by_subsystem"]["text_encoder"]["mean_exact_tensor_match_ratio"],
                "foundation_vs_layered_mean_relative_l2_delta": foundation_vs_layered["by_subsystem"]["text_encoder"]["mean_block_relative_l2_delta"],
            },
        },
        "vae": {
            "decision": "skip-direct-merge",
            "reason": "Layered keeps RGBA semantics while the core stack is RGB, so the VAE stays out of Stage 2 fusion.",
            "evidence": {
                "structural_compatibility": vae_subsystem["structural_compatibility"],
                "shape_mismatch_count": vae_subsystem["evidence"]["shape_mismatch_count"],
                "base_label": matrix["model_summaries"]["qwen-image-base"]["vae"]["label"],
                "layered_label": matrix["model_summaries"]["qwen-image-layered"]["vae"]["label"],
            },
        },
        "rope": {
            "decision": "skip-direct-merge",
            "reason": "Layer3D positional behavior does not directly align with the 2D foundation, so Stage 2 keeps rope changes behind the bridge experiment.",
            "evidence": {
                "structural_compatibility": rope_subsystem["structural_compatibility"],
                "foundation_label": matrix["model_summaries"]["qwen-image-2512"]["rope"]["label"],
                "layered_label": matrix["model_summaries"]["qwen-image-layered"]["rope"]["label"],
            },
        },
    }


def build_core_delta_recipe(
    models: dict[str, dict[str, Any]],
    config_dir: Path,
    stage1_evidence: dict[str, Any],
    exclusions: dict[str, Any],
) -> dict[str, Any]:
    recipe = load_json(config_dir / "stage-2-delta-edit.yaml")
    return {
        "name": recipe["recipe_name"],
        "foundation_alias": recipe["foundation_model"],
        "foundation_model": models[recipe["foundation_model"]]["model_id"],
        "delta_source_alias": recipe["delta_source"],
        "delta_source_model": models[recipe["delta_source"]]["model_id"],
        "delta_base_candidate_alias": recipe["delta_base_candidate"],
        "delta_base_candidate_model": models[recipe["delta_base_candidate"]]["model_id"],
        "target_components": recipe["target_components"],
        "target_subsystems": recipe["target_subsystems"],
        "excluded_subsystems": recipe["excluded_subsystems"],
        "coefficient_sweep": [float(value) for value in recipe["coefficient_sweep"]],
        "selection_rule": recipe["selection_rule"],
        "observed_hot_blocks": stage1_evidence["foundation_vs_edit"]["observed_hot_blocks"],
        "exclusion_refs": {key: exclusions[key]["decision"] for key in ("text_encoder", "vae", "rope")},
    }



def core_candidate_id(weight: float) -> str:
    return f"core-delta-w{int(round(weight * 100)):03d}"


def build_core_delta_candidates(core_delta_recipe: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for weight in core_delta_recipe["coefficient_sweep"]:
        suffix = core_candidate_id(weight)
        candidates.append(
            {
                "candidate_id": suffix,
                "blend_weight": weight,
                "selection_status": "planned",
                "output_checkpoint": stage2_remote_path(
                    "stage-2",
                    "artifacts",
                    "core-candidates",
                    suffix,
                    "qwen-image-1.9-core-bf16.safetensors",
                ),
                "smoke_report": stage2_remote_path(
                    "stage-2",
                    "evals",
                    "core-candidates",
                    suffix,
                    "smoke-summary.json",
                ),
            }
        )
    return candidates


def select_core_candidate(
    candidates: list[dict[str, Any]],
    preferred_weight: float = CORE_CANDIDATE_DEFAULT_WEIGHT,
) -> dict[str, Any]:
    for candidate in candidates:
        if abs(float(candidate["blend_weight"]) - preferred_weight) < 1e-9:
            return {
                **candidate,
                "selection_status": "provisional-default",
                "selection_note": "Promote this candidate to stable once the remote smoke suite confirms edit retention without visible generation regression.",
            }
    raise Stage2FusionError("Stage 2 core coefficient sweep did not contain the provisional default candidate.")


def build_layered_bridge_recipe(
    models: dict[str, dict[str, Any]],
    config_dir: Path,
    selected_core_candidate: dict[str, Any],
) -> dict[str, Any]:
    recipe = load_json(config_dir / "stage-2-layered-bridge.yaml")
    return {
        "name": recipe["recipe_name"],
        "strategy": recipe["strategy"],
        "foundation_alias": recipe["foundation_model"],
        "foundation_model": models[recipe["foundation_model"]]["model_id"],
        "donor_alias": recipe["donor_model"],
        "donor_model": models[recipe["donor_model"]]["model_id"],
        "base_core_candidate_id": selected_core_candidate["candidate_id"],
        "base_core_checkpoint": selected_core_candidate["output_checkpoint"],
        "target_component": recipe["target_component"],
        "target_subsystems": recipe["target_subsystems"],
        "bridge_block_window": recipe["bridge_block_window"],
        "extra_parameter_paths": recipe["extra_parameter_paths"],
        "freeze_policy": recipe["freeze_policy"],
        "trainable_modules": recipe["trainable_modules"],
        "distillation_target": recipe["distillation_target"],
        "layered_output_adapter": recipe["layered_output_adapter"],
        "output_adapter": stage2_remote_path("stage-2", "artifacts", "experimental", "layered-bridge-adapter.safetensors"),
        "output_checkpoint": stage2_remote_path(
            "stage-2",
            "artifacts",
            "experimental",
            "qwen-image-1.9-layered-bridge-bf16.safetensors",
        ),
        "experimental": True,
    }


def build_planned_dataset_records(
    split_name: str,
    split_payload: dict[str, Any],
    output_root: str,
) -> list[dict[str, Any]]:
    prompts = split_payload["prompt_bank"]
    seeds = split_payload["seed_schedule"]
    sample_count = int(split_payload["planned_sample_count"])
    records = []
    for index in range(sample_count):
        prompt_entry = prompts[index % len(prompts)]
        seed = seeds[index % len(seeds)]
        sample_id = f"{split_name}-{index + 1:04d}"
        asset_root = stage2_remote_path(output_root, split_name)
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "teacher_model_alias": split_payload["teacher_model_alias"],
            "teacher_model": split_payload["teacher_model"],
            "seed": seed,
            "generation_settings": split_payload["generation_settings"],
            "asset_paths": {
                "metadata": stage2_remote_path(asset_root, "metadata", f"{sample_id}.json"),
            },
            "output_metadata": {
                "task": split_payload["task"],
                "flattened_to_rgb": split_name == "layered_teacher",
            },
        }
        if isinstance(prompt_entry, dict):
            record.update(prompt_entry)
        else:
            record["prompt"] = prompt_entry
        if split_name == "edit_teacher":
            record["asset_paths"]["source_image"] = stage2_remote_path(asset_root, "source", f"{sample_id}.png")
            record["asset_paths"]["edited_image"] = stage2_remote_path(asset_root, "edited", f"{sample_id}.png")
        else:
            record["asset_paths"]["image"] = stage2_remote_path(asset_root, "images", f"{sample_id}.png")
        records.append(record)
    return records


def build_dataset_manifest(
    models: dict[str, dict[str, Any]],
    config_dir: Path,
) -> dict[str, Any]:
    recipe = load_json(config_dir / "stage-2-synthetic-dataset.yaml")
    splits: dict[str, Any] = {}
    planned_records: list[dict[str, Any]] = []
    output_root = recipe["output_root"]
    for split_name, split in recipe["splits"].items():
        split_payload = {
            "teacher_model_alias": split["teacher_model"],
            "teacher_model": models[split["teacher_model"]]["model_id"],
            "task": split["task"],
            "planned_sample_count": split["planned_sample_count"],
            "seed_schedule": split["seed_schedule"],
            "generation_settings": split["generation_settings"],
            "prompt_bank": split["prompt_bank"],
        }
        splits[split_name] = {
            **split_payload,
            "asset_root": stage2_remote_path(output_root, split_name),
        }
        planned_records.extend(build_planned_dataset_records(split_name, split_payload, output_root))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "stage2",
        "name": recipe["recipe_name"],
        "relative_paths_only": True,
        "output_root": output_root,
        "flatten_layered_rgba": recipe["flatten_layered_rgba"],
        "record_schema": recipe["record_schema"],
        "splits": splits,
        "planned_records": planned_records,
    }


def build_remote_jobs(
    core_delta_candidates: list[dict[str, Any]],
    dataset_manifest: dict[str, Any],
    selected_core_candidate: dict[str, Any],
    layered_bridge_recipe: dict[str, Any],
) -> dict[str, Any]:
    return {
        "core_delta_sweep": {
            "status": "planned",
            "entrypoint": "scripts/stage-2-build-edit-delta.py",
            "workdir": stage2_remote_path("stage-2", "jobs", "core-delta-sweep"),
            "log_path": stage2_remote_path("stage-2", "logs", "core-delta-sweep.log"),
            "outputs": [candidate["output_checkpoint"] for candidate in core_delta_candidates],
        },
        "core_smoke_eval": {
            "status": "planned",
            "entrypoint": "scripts/stage-2-compose-bf16-checkpoint.py",
            "workdir": stage2_remote_path("stage-2", "jobs", "core-smoke-eval"),
            "log_path": stage2_remote_path("stage-2", "logs", "core-smoke-eval.log"),
            "outputs": [selected_core_candidate["smoke_report"]],
        },
        "teacher_dataset_generation": {
            "status": "planned",
            "entrypoint": "scripts/stage-2-generate-teacher-dataset.py",
            "workdir": stage2_remote_path("stage-2", "jobs", "teacher-dataset"),
            "log_path": stage2_remote_path("stage-2", "logs", "teacher-dataset.log"),
            "outputs": [dataset_manifest["output_root"]],
        },
        "layered_bridge_train": {
            "status": "planned",
            "entrypoint": "scripts/stage-2-build-layered-bridge.py",
            "workdir": stage2_remote_path("stage-2", "jobs", "layered-bridge-train"),
            "log_path": stage2_remote_path("stage-2", "logs", "layered-bridge-train.log"),
            "outputs": [layered_bridge_recipe["output_adapter"], layered_bridge_recipe["output_checkpoint"]],
        },
        "experimental_smoke_eval": {
            "status": "planned",
            "entrypoint": "scripts/stage-2-compose-bf16-checkpoint.py",
            "workdir": stage2_remote_path("stage-2", "jobs", "experimental-smoke-eval"),
            "log_path": stage2_remote_path("stage-2", "logs", "experimental-smoke-eval.log"),
            "outputs": [
                stage2_remote_path("stage-2", "evals", "experimental", "smoke-summary.json"),
            ],
        },
        "core_edit_eval": {
            "status": "planned",
            "entrypoint": "scripts/stage-2-compose-bf16-checkpoint.py",
            "workdir": stage2_remote_path("stage-2", "jobs", "core-edit-eval"),
            "log_path": stage2_remote_path("stage-2", "logs", "core-edit-eval.log"),
            "outputs": [
                stage2_remote_path("stage-2", "evals", "core-edit", "edit-summary.json"),
            ],
        },
        "consistency_eval": {
            "status": "planned",
            "entrypoint": "scripts/stage-2-compose-bf16-checkpoint.py",
            "workdir": stage2_remote_path("stage-2", "jobs", "consistency-eval"),
            "log_path": stage2_remote_path("stage-2", "logs", "consistency-eval.log"),
            "outputs": [
                stage2_remote_path("stage-2", "evals", "consistency", "consistency-summary.json"),
            ],
        },
    }


def build_artifacts_section(
    artifact_paths: dict[str, Path],
    selected_core_candidate: dict[str, Any],
    layered_bridge_recipe: dict[str, Any],
    target_dir: Path,
) -> dict[str, str]:
    return {
        "report_readme": repo_relative_path(artifact_paths["report_readme"]),
        "merge_manifest": repo_relative_path(artifact_paths["merge_manifest_json"]),
        "dataset_manifest": repo_relative_path(artifact_paths["dataset_manifest_json"]),
        "training_report": repo_relative_path(target_dir / "training-report.md"),
        "training_figures_dir": repo_relative_path(target_dir / "figures"),
        "stable_core_checkpoint": selected_core_candidate["output_checkpoint"],
        "experimental_bridge_adapter": layered_bridge_recipe["output_adapter"],
        "experimental_bridge_checkpoint": layered_bridge_recipe["output_checkpoint"],
        "artifact_dir": repo_relative_path(target_dir),
    }


def build_fusion_manifest(
    matrix: dict[str, Any],
    weight_analysis: dict[str, Any],
    models: dict[str, dict[str, Any]],
    remote_context: dict[str, Any],
    config_dir: Path,
    artifact_paths: dict[str, Path],
    target_dir: Path,
    stage1_refs: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1_evidence = build_stage1_evidence(matrix, weight_analysis, stage1_refs)
    exclusions = build_exclusions(matrix, weight_analysis)
    core_delta_recipe = build_core_delta_recipe(models, config_dir, stage1_evidence, exclusions)
    core_delta_candidates = build_core_delta_candidates(core_delta_recipe)
    selected_core_candidate = select_core_candidate(core_delta_candidates)
    layered_bridge_recipe = build_layered_bridge_recipe(models, config_dir, selected_core_candidate)
    dataset_manifest = build_dataset_manifest(models, config_dir)
    remote_jobs = build_remote_jobs(
        core_delta_candidates,
        dataset_manifest,
        selected_core_candidate,
        layered_bridge_recipe,
    )
    artifacts = build_artifacts_section(
        artifact_paths,
        selected_core_candidate,
        layered_bridge_recipe,
        target_dir,
    )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage": "stage2",
        "path_policy": {
            "absolute_paths_allowed": False,
            "local_paths": "repo-relative",
            "remote_paths": "relative-to-remote-workdir-or-remote-artifact-root",
        },
        "remote": {
            "name": remote_context["name"],
            "python": remote_context["python"],
            "path_policy": "manifest paths stay relative even when remote roots are absolute in operator config",
        },
        "foundation_model": models["qwen-image-2512"]["model_id"],
        "source_models": {
            alias: models[alias]["model_id"] for alias in (
                "qwen-image-base",
                "qwen-image-2512",
                "qwen-image-edit-2511",
                "qwen-image-layered",
            )
        },
        "compatibility_summary": matrix["summary"],
        "stage1_evidence": stage1_evidence,
        "exclusions": exclusions,
        "core_delta_recipe": core_delta_recipe,
        "core_delta_candidates": core_delta_candidates,
        "selected_core_candidate": selected_core_candidate,
        "layered_bridge_recipe": layered_bridge_recipe,
        "dataset": {
            "manifest_path": artifacts["dataset_manifest"],
            "output_root": dataset_manifest["output_root"],
            "split_counts": {
                split_name: payload["planned_sample_count"]
                for split_name, payload in dataset_manifest["splits"].items()
            },
            "relative_paths_only": dataset_manifest["relative_paths_only"],
        },
        "remote_jobs": remote_jobs,
        "artifacts": artifacts,
    }
    return manifest, dataset_manifest


def rebuild_dataset_records(dataset_manifest: dict[str, Any]) -> None:
    planned_records: list[dict[str, Any]] = []
    output_root = dataset_manifest["output_root"]
    for split_name, split_payload in dataset_manifest["splits"].items():
        payload = {
            "teacher_model_alias": split_payload["teacher_model_alias"],
            "teacher_model": split_payload["teacher_model"],
            "task": split_payload["task"],
            "planned_sample_count": split_payload["planned_sample_count"],
            "seed_schedule": split_payload["seed_schedule"],
            "generation_settings": split_payload["generation_settings"],
            "prompt_bank": split_payload["prompt_bank"],
        }
        planned_records.extend(build_planned_dataset_records(split_name, payload, output_root))
    dataset_manifest["planned_records"] = planned_records


def apply_run_profile(
    manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    profiles: dict[str, Any],
    run_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_profile = profiles[run_profile]
    profile_manifest = copy.deepcopy(manifest)
    profile_dataset_manifest = copy.deepcopy(dataset_manifest)
    limits = active_profile.get("limits", {})

    if run_profile == "smoke":
        target_candidate_id = str(limits.get("core_candidate_id", profile_manifest["selected_core_candidate"]["candidate_id"]))
        candidates = [
            candidate
            for candidate in profile_manifest["core_delta_candidates"]
            if candidate["candidate_id"] == target_candidate_id
        ]
        if not candidates:
            raise Stage2FusionError(
                f"Smoke profile requested candidate `{target_candidate_id}` but it is not part of the core sweep."
            )
        profile_manifest["core_delta_candidates"] = candidates
        profile_manifest["selected_core_candidate"] = {
            **candidates[0],
            "selection_status": "smoke-only",
            "selection_note": "Smoke profile runs one candidate to validate pipeline wiring before the full sweep.",
        }
        profile_manifest["core_delta_recipe"]["coefficient_sweep"] = [
            parse_candidate_weight(target_candidate_id)
        ]
        profile_manifest["layered_bridge_recipe"]["base_core_candidate_id"] = target_candidate_id
        profile_manifest["layered_bridge_recipe"]["base_core_checkpoint"] = candidates[0]["output_checkpoint"]
        profile_manifest["layered_bridge_recipe"]["training_limits"] = {
            "max_steps": int(limits.get("bridge_train_steps", 64)),
            "batch_size": int(limits.get("bridge_batch_size", 1)),
            "eval_prompt_count": int(limits.get("eval_prompt_count", 4)),
        }

        smoke_count = int(limits.get("dataset_samples_per_split", 2))
        for split_name, split_payload in profile_dataset_manifest["splits"].items():
            split_payload["planned_sample_count"] = smoke_count
            split_payload["seed_schedule"] = split_payload["seed_schedule"][:smoke_count]
            split_payload["prompt_bank"] = split_payload["prompt_bank"][:smoke_count]
            profile_manifest["dataset"]["split_counts"][split_name] = smoke_count
        rebuild_dataset_records(profile_dataset_manifest)

    profile_manifest["remote_jobs"] = build_remote_jobs(
        profile_manifest["core_delta_candidates"],
        profile_dataset_manifest,
        profile_manifest["selected_core_candidate"],
        profile_manifest["layered_bridge_recipe"],
    )
    profile_manifest["run_profile"] = run_profile
    profile_manifest["resource_profile"] = active_profile.get("resource_profile", {})
    profile_manifest["limits"] = limits
    return profile_manifest, profile_dataset_manifest


