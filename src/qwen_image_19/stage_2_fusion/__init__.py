from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

from qwen_image_19.config_io import load_json_yaml, repo_root, write_json, write_text
from qwen_image_19.remote import default_remote_context
from qwen_image_19.stage_1_analysis import load_model_inventory


STAGE1_ARTIFACT_DIR = Path("reports") / "stage-1"
DEFAULT_STAGE2_ARTIFACT_DIR = Path("reports") / "stage-2"
DEFAULT_STAGE2_RUN_STATUS = Path("stage-2") / "run-status.json"
CORE_CANDIDATE_DEFAULT_WEIGHT = 0.35
DEFAULT_RUN_PROFILE = "full"
SUPPORTED_RUN_PROFILES = {"smoke", "full"}


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
    payload = load_json_yaml(config_dir / "stage-2-run-profiles.yaml")
    profiles = payload.get("profiles", {})
    for name in ("smoke", "full"):
        if name not in profiles:
            raise Stage2FusionError(
                "Stage 2 run profile config must define both `smoke` and `full` profiles."
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
    recipe = load_json_yaml(config_dir / "stage-2-delta-edit.yaml")
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


def build_edit_delta_recipe(
    models: dict[str, dict[str, Any]],
    config_dir: Path,
    stage1_evidence: dict[str, Any],
    exclusions: dict[str, Any],
) -> dict[str, Any]:
    return build_core_delta_recipe(models, config_dir, stage1_evidence, exclusions)


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
    recipe = load_json_yaml(config_dir / "stage-2-layered-bridge.yaml")
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
    recipe = load_json_yaml(config_dir / "stage-2-synthetic-dataset.yaml")
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


def ensure_outputs_exist(paths: list[str]) -> list[str]:
    missing = []
    for relative in paths:
        path = repo_root() / relative
        if not path.exists():
            missing.append(relative)
            continue
        if path.is_file() and path.stat().st_size == 0:
            missing.append(relative)
            continue
        if path.is_dir() and not any(path.iterdir()):
            missing.append(relative)
    return missing


def clear_target_path(relative_path: str) -> bool:
    path = repo_root() / relative_path
    if not path.exists():
        return False
    if path.is_dir():
        shutil.rmtree(path)
        return True
    path.unlink()
    return True


def clear_output_targets(paths: list[str]) -> int:
    removed = 0
    for relative_path in paths:
        if clear_target_path(relative_path):
            removed += 1
    return removed


def run_subprocess_job(
    command: list[str],
    log_path: str,
    workdir: str,
) -> tuple[int, float]:
    log_file = repo_root() / log_path
    log_file.parent.mkdir(parents=True, exist_ok=True)
    (repo_root() / workdir).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=repo_root(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = time.perf_counter() - started
    return process.returncode, duration


def build_job_command(
    job_name: str,
    job_payload: dict[str, Any],
    manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    remote_context: dict[str, Any],
) -> list[str]:
    python_cmd = str(remote_context.get("python") or "python3")
    if job_name == "core_delta_sweep":
        candidate = manifest["selected_core_candidate"] if manifest["run_profile"] == "smoke" else manifest["core_delta_candidates"][0]
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-build-edit-delta.py"),
            "--execute",
            "--candidate-id",
            candidate["candidate_id"],
            "--output-checkpoint",
            candidate["output_checkpoint"],
        ]
    if job_name == "core_smoke_eval":
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-compose-bf16-checkpoint.py"),
            "--execute",
            "--task",
            "core-smoke",
            "--model-ref",
            manifest["selected_core_candidate"]["output_checkpoint"],
            "--output",
            manifest["selected_core_candidate"]["smoke_report"],
            "--num-prompts",
            str(int(manifest["limits"].get("eval_prompt_count", 6))),
        ]
    if job_name == "teacher_dataset_generation":
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-generate-teacher-dataset.py"),
            "--execute",
            "--manifest",
            manifest["dataset"]["manifest_path"],
        ]
    if job_name == "layered_bridge_train":
        limits = manifest["layered_bridge_recipe"].get("training_limits", {})
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-build-layered-bridge.py"),
            "--execute",
            "--output-adapter",
            manifest["layered_bridge_recipe"]["output_adapter"],
            "--output-checkpoint",
            manifest["layered_bridge_recipe"]["output_checkpoint"],
            "--metrics-output",
            stage2_remote_path("stage-2", "metrics", "layered-bridge-train.json"),
            "--max-steps",
            str(int(limits.get("max_steps", 500))),
            "--batch-size",
            str(int(limits.get("batch_size", 1))),
        ]
    if job_name == "experimental_smoke_eval":
        return [
            python_cmd,
            str(repo_root() / "scripts" / "stage-2-compose-bf16-checkpoint.py"),
            "--execute",
            "--task",
            "experimental-smoke",
            "--model-ref",
            manifest["layered_bridge_recipe"]["output_checkpoint"],
            "--output",
            stage2_remote_path("stage-2", "evals", "experimental", "smoke-summary.json"),
            "--num-prompts",
            str(int(manifest["limits"].get("eval_prompt_count", 6))),
        ]
    raise Stage2FusionError(f"Unsupported remote job `{job_name}` in Stage 2 executor.")


def run_stage2_jobs(
    manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    artifact_paths: dict[str, Path],
    remote_context: dict[str, Any],
    resume: bool = False,
) -> dict[str, Any]:
    cleanup_performed = False
    status_payload = {
        "stage": "stage2",
        "run_profile": manifest["run_profile"],
        "execution_policy": "resume" if resume else "overwrite",
        "cleanup_performed": cleanup_performed,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "jobs": {},
        "summary": {
            "failed_job": None,
            "resume_hint": None,
        },
    }
    existing_status = artifact_paths["run_status_json"]
    if resume and existing_status.exists():
        try:
            status_payload = json.loads(existing_status.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
        status_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        status_payload["run_profile"] = manifest["run_profile"]
        status_payload["execution_policy"] = "resume"
        status_payload.setdefault("jobs", {})
        status_payload.setdefault("summary", {"failed_job": None, "resume_hint": None})
    elif not resume and existing_status.exists():
        existing_status.unlink()
        cleanup_performed = True
        status_payload["cleanup_performed"] = cleanup_performed

    job_names = list(manifest["remote_jobs"].keys())
    for job_name in job_names:
        payload = manifest["remote_jobs"][job_name]
        existing = status_payload["jobs"].get(job_name, {})
        if resume and existing.get("status") == "succeeded":
            missing = ensure_outputs_exist(payload["outputs"])
            if not missing:
                status_payload["jobs"][job_name] = {
                    **existing,
                    "status": "skipped",
                    "skip_reason": "already_succeeded",
                }
                write_json(artifact_paths["run_status_json"], status_payload)
                continue

        if not resume:
            removed_count = clear_output_targets(payload["outputs"])
            if removed_count > 0:
                cleanup_performed = True
                status_payload["cleanup_performed"] = True

        if job_name == "core_delta_sweep":
            commands = []
            started_iso = datetime.now(timezone.utc).isoformat()
            status_payload["jobs"][job_name] = {
                "status": "running",
                "commands": [],
                "workdir": payload["workdir"],
                "stdout_stderr_log": payload["log_path"],
                "started_at": started_iso,
            }
            write_json(artifact_paths["run_status_json"], status_payload)
            started = time.perf_counter()
            for candidate in manifest["core_delta_candidates"]:
                command = [
                    str(remote_context.get("python") or "python3"),
                    str(repo_root() / "scripts" / "stage-2-build-edit-delta.py"),
                    "--execute",
                    "--candidate-id",
                    candidate["candidate_id"],
                    "--output-checkpoint",
                    candidate["output_checkpoint"],
                ]
                commands.append(command)
                exit_code, _ = run_subprocess_job(command, payload["log_path"], payload["workdir"])
                if exit_code != 0:
                    duration = time.perf_counter() - started
                    status_payload["jobs"][job_name] = {
                        "status": "failed",
                        "commands": commands,
                        "workdir": payload["workdir"],
                        "stdout_stderr_log": payload["log_path"],
                        "started_at": started_iso,
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "duration_seconds": round(duration, 4),
                        "exit_code": exit_code,
                        "outputs": payload["outputs"],
                        "missing_outputs": ensure_outputs_exist(payload["outputs"]),
                        "failure_reason": f"core sweep candidate command failed with exit code {exit_code}",
                    }
                    status_payload["summary"] = {
                        "failed_job": job_name,
                        "resume_hint": f"Re-run `q19 stage2 fuse --run-profile {manifest['run_profile']} --execute --resume` after fixing `{job_name}`.",
                    }
                    write_json(artifact_paths["run_status_json"], status_payload)
                    raise Stage2FusionError(
                        f"Stage 2 execution failed at `{job_name}`. "
                        f"Check `{repo_relative_path(artifact_paths['run_status_json'])}` for details."
                    )
            duration = time.perf_counter() - started
            missing_outputs = ensure_outputs_exist(payload["outputs"])
            failed = bool(missing_outputs)
            status_payload["jobs"][job_name] = {
                "status": "failed" if failed else "succeeded",
                "commands": commands,
                "workdir": payload["workdir"],
                "stdout_stderr_log": payload["log_path"],
                "started_at": started_iso,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(duration, 4),
                "exit_code": 0,
                "outputs": payload["outputs"],
                "missing_outputs": missing_outputs,
                "failure_reason": f"Missing expected outputs: {', '.join(missing_outputs)}" if failed else None,
            }
            if failed:
                status_payload["summary"] = {
                    "failed_job": job_name,
                    "resume_hint": f"Re-run `q19 stage2 fuse --run-profile {manifest['run_profile']} --execute --resume` after fixing `{job_name}`.",
                }
                write_json(artifact_paths["run_status_json"], status_payload)
                raise Stage2FusionError(
                    f"Stage 2 execution failed at `{job_name}`. "
                    f"Check `{repo_relative_path(artifact_paths['run_status_json'])}` for details."
                )
            write_json(artifact_paths["run_status_json"], status_payload)
            continue

        command = build_job_command(job_name, payload, manifest, dataset_manifest, remote_context)
        started_iso = datetime.now(timezone.utc).isoformat()
        status_payload["jobs"][job_name] = {
            "status": "running",
            "command": command,
            "workdir": payload["workdir"],
            "stdout_stderr_log": payload["log_path"],
            "started_at": started_iso,
        }
        write_json(artifact_paths["run_status_json"], status_payload)
        exit_code, duration = run_subprocess_job(command, payload["log_path"], payload["workdir"])
        ended_iso = datetime.now(timezone.utc).isoformat()
        missing_outputs = ensure_outputs_exist(payload["outputs"])
        failed = exit_code != 0 or bool(missing_outputs)
        job_status = {
            "status": "failed" if failed else "succeeded",
            "command": command,
            "workdir": payload["workdir"],
            "stdout_stderr_log": payload["log_path"],
            "started_at": started_iso,
            "ended_at": ended_iso,
            "duration_seconds": round(duration, 4),
            "exit_code": exit_code,
            "outputs": payload["outputs"],
            "missing_outputs": missing_outputs,
        }
        if job_name == "layered_bridge_train":
            metrics_path = repo_root() / stage2_remote_path("stage-2", "metrics", "layered-bridge-train.json")
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if any(not math.isfinite(float(value)) for value in metrics.get("loss_curve", [])):
                    failed = True
                    job_status["status"] = "failed"
                    job_status["failure_reason"] = "Detected non-finite training metric."
        if failed and "failure_reason" not in job_status:
            job_status["failure_reason"] = (
                f"Job returned exit code {exit_code}."
                if exit_code != 0
                else f"Missing expected outputs: {', '.join(missing_outputs)}"
            )
        status_payload["jobs"][job_name] = job_status
        if failed:
            status_payload["summary"] = {
                "failed_job": job_name,
                "resume_hint": f"Re-run `q19 stage2 fuse --run-profile {manifest['run_profile']} --execute --resume` after fixing `{job_name}`.",
            }
            write_json(artifact_paths["run_status_json"], status_payload)
            raise Stage2FusionError(
                f"Stage 2 execution failed at `{job_name}`. "
                f"Check `{repo_relative_path(artifact_paths['run_status_json'])}` for details."
            )
        write_json(artifact_paths["run_status_json"], status_payload)

    status_payload["cleanup_performed"] = cleanup_performed
    status_payload["summary"] = {
        "failed_job": None,
        "resume_hint": None,
    }
    write_json(artifact_paths["run_status_json"], status_payload)
    return status_payload


def render_candidate_rows(candidates: list[dict[str, Any]], selected_candidate_id: str) -> str:
    rows = []
    for candidate in candidates:
        selected = "selected" if candidate["candidate_id"] == selected_candidate_id else "candidate"
        rows.append(
            f"| `{candidate['candidate_id']}` | `{candidate['blend_weight']}` | `{selected}` | `{candidate['output_checkpoint']}` | `{candidate['smoke_report']}` |"
        )
    return "\n".join(rows)


def render_dataset_rows(dataset_manifest: dict[str, Any]) -> str:
    rows = []
    for split_name, payload in dataset_manifest["splits"].items():
        rows.append(
            f"| `{split_name}` | `{payload['teacher_model']}` | `{payload['task']}` | `{payload['planned_sample_count']}` | `{payload['asset_root']}` |"
        )
    return "\n".join(rows)


def render_job_rows(remote_jobs: dict[str, Any]) -> str:
    rows = []
    for name, payload in remote_jobs.items():
        rows.append(
            f"| `{name}` | `{payload['status']}` | `{payload['entrypoint']}` | `{payload['workdir']}` | `{payload['log_path']}` |"
        )
    return "\n".join(rows)


def render_artifact_rows(artifacts: dict[str, str]) -> str:
    return "\n".join(
        f"| `{name}` | `{path}` |" for name, path in artifacts.items()
    )


def render_fusion_report(manifest: dict[str, Any], dataset_manifest: dict[str, Any]) -> str:
    core = manifest["core_delta_recipe"]
    selected = manifest["selected_core_candidate"]
    layered = manifest["layered_bridge_recipe"]
    evidence = manifest["stage1_evidence"]
    exclusions = manifest["exclusions"]
    return f"""# Stage 2 Fusion

## Mission
Stage 2 now builds two tracks from the Stage 1 evidence: a stable BF16 core based on `2512 + transformer-only edit delta`, and an experimental Layered bridge branch that learns RGB behavior from synthetic teacher data instead of pretending `vae` and `rope` mismatches will disappear.

## Run Mode
- Run mode: `{manifest['run_mode']}`
- Run profile: `{manifest['run_profile']}`
- Execution enabled: `{manifest['execution_enabled']}`
- Execution policy: `{manifest['execution_policy']}`
- Cleanup performed: `{manifest['cleanup_performed']}`
- Resource profile: `num_gpus={manifest['resource_profile'].get('num_gpus')}`, `vram_target_gb={manifest['resource_profile'].get('vram_target_gb')}`
- Limits: `{json.dumps(manifest['limits'], sort_keys=True)}`

## Stage 1 Evidence
- Foundation vs Edit transformer path is the one clean merge lane: `shared={evidence['foundation_vs_edit']['shared_key_count']}`, `exact={evidence['foundation_vs_edit']['exact_equal_tensor_ratio']}`, `strategy={evidence['foundation_vs_edit']['transformer_merge_strategy']}`.
- Edit deltas cluster in late MMDiT blocks: `{', '.join(str(value) for value in evidence['foundation_vs_edit']['observed_hot_blocks']) or 'none captured'}`.
- Layered text encoder is a no-op donor in practice: `exact={evidence['layered_vs_core']['text_encoder_exact_match']}`.
- Layered conflicts remain real: VAE `{evidence['layered_conflicts']['vae']['base_label']} -> {evidence['layered_conflicts']['vae']['layered_label']}`, rope `{evidence['layered_conflicts']['rope']['foundation_label']} -> {evidence['layered_conflicts']['rope']['layered_label']}`.

## No-Go List
- `text_encoder`: {exclusions['text_encoder']['reason']}
- `vae`: {exclusions['vae']['reason']}
- `rope`: {exclusions['rope']['reason']}

## Stable Core Track
- Foundation: `{core['foundation_model']}`
- Delta source: `{core['delta_source_model']}`
- Delta base candidate: `{core['delta_base_candidate_model']}`
- Target scope: `{', '.join(core['target_components'])}` / `{', '.join(core['target_subsystems'])}`
- Selection rule: `{core['selection_rule']}`

| Candidate | Blend weight | Status | Planned checkpoint | Planned smoke report |
| --- | --- | --- | --- | --- |
{render_candidate_rows(manifest['core_delta_candidates'], selected['candidate_id'])}

## Experimental Layered Bridge Track
- Donor: `{layered['donor_model']}`
- Strategy: `{layered['strategy']}`
- Base core candidate: `{layered['base_core_candidate_id']}`
- Bridge scope: `transformer_blocks.{layered['bridge_block_window']['start']}:{layered['bridge_block_window']['end']}`
- Extra parameter paths: {', '.join(layered['extra_parameter_paths'])}
- Trainable modules: {', '.join(layered['trainable_modules'])}
- Freeze policy: {', '.join(layered['freeze_policy'])}
- Distillation target: `{layered['distillation_target']}`
- Output adapter: `{layered['output_adapter']}`
- Output checkpoint: `{layered['output_checkpoint']}`

## Teacher Dataset
- Dataset manifest: `{manifest['dataset']['manifest_path']}`
- Output root: `{dataset_manifest['output_root']}`
- Layered flattening: `{dataset_manifest['flatten_layered_rgba']}`

| Split | Teacher model | Task | Planned samples | Asset root |
| --- | --- | --- | --- | --- |
{render_dataset_rows(dataset_manifest)}

## Remote Jobs
| Job | Status | Entry point | Workdir | Log |
| --- | --- | --- | --- | --- |
{render_job_rows(manifest['remote_jobs'])}

## Artifacts
| Artifact | Path |
| --- | --- |
{render_artifact_rows(manifest['artifacts'])}

## Limitations
- Stage 2 does not attempt true RGBA decomposition support. Layered supervision is flattened back into RGB composites.
- The stable core winner is provisional until the remote coefficient sweep and smoke suite complete.
- The Layered branch is experimental and should be treated as a bridge adapter, not a drop-in replacement for the core checkpoint.
"""


def fuse(
    artifact_dir: str | Path | None = None,
    remote_config: str | None = None,
    cache_dir: str | None = None,
    dry_run: bool = False,
    smoke_run: bool = False,
    run_profile: str | None = None,
    execute: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    selected_run_profile = validate_run_options(dry_run=dry_run, smoke_run=smoke_run, run_profile=run_profile)
    models = load_model_inventory()
    matrix, weight_analysis, stage1_refs = require_stage1_artifacts()
    remote_context = default_remote_context(remote_config)
    if cache_dir:
        remote_context["cache_dir"] = cache_dir
    config_dir = repo_root() / "configs" / "merge"
    run_profiles = load_run_profiles(config_dir)
    target_dir = Path(artifact_dir) if artifact_dir else repo_root() / DEFAULT_STAGE2_ARTIFACT_DIR
    artifact_paths = stage2_artifact_paths(target_dir)
    compatibility_shims = build_stage2_compatibility_shims(target_dir)
    manifest, dataset_manifest = build_fusion_manifest(
        matrix,
        weight_analysis,
        models,
        remote_context,
        config_dir,
        artifact_paths,
        target_dir,
        stage1_refs,
    )
    manifest, dataset_manifest = apply_run_profile(
        manifest,
        dataset_manifest,
        run_profiles,
        selected_run_profile,
    )
    manifest["run_mode"] = "dry-run" if dry_run else "write"
    manifest["execution_enabled"] = bool(execute and not dry_run)
    manifest["execution_policy"] = "resume" if resume else "overwrite"
    manifest["cleanup_performed"] = False
    manifest["artifacts"]["run_status"] = repo_relative_path(artifact_paths["run_status_json"])

    report = render_fusion_report(manifest, dataset_manifest)
    result = {
        "stage": "stage2",
        "mode": "dry-run" if dry_run else "write",
        "run_profile": selected_run_profile,
        "execution_enabled": bool(execute and not dry_run),
        "execution_policy": manifest["execution_policy"],
        "manifest": manifest,
        "dataset_manifest": dataset_manifest,
        "report_preview": report if dry_run else report.splitlines()[:12],
        "artifact_dir": repo_relative_path(target_dir),
        "artifact_paths": {
            key: repo_relative_path(path) for key, path in artifact_paths.items() if key != "artifact_dir"
        },
        "compatibility_shims": {
            key: repo_relative_path(path) for key, path in compatibility_shims.items()
        },
    }
    if dry_run:
        return result
    write_json(artifact_paths["merge_manifest_json"], manifest)
    write_json(artifact_paths["dataset_manifest_json"], dataset_manifest)
    write_text(artifact_paths["report_readme"], report)
    written = [
        repo_relative_path(artifact_paths["merge_manifest_json"]),
        repo_relative_path(artifact_paths["dataset_manifest_json"]),
        repo_relative_path(artifact_paths["report_readme"]),
    ]
    if compatibility_shims:
        write_json(compatibility_shims["legacy_merge_manifest_json"], manifest)
        write_text(compatibility_shims["legacy_report_md"], render_stage2_compatibility_stub(target_dir))
        written.extend(repo_relative_path(path) for path in compatibility_shims.values())
    if execute:
        run_status = run_stage2_jobs(
            manifest,
            dataset_manifest,
            artifact_paths,
            remote_context,
            resume=resume,
        )
        manifest["cleanup_performed"] = bool(run_status.get("cleanup_performed"))
        result["execution_policy"] = manifest["execution_policy"]
        result["run_status"] = run_status
        refreshed_report = render_fusion_report(manifest, dataset_manifest)
        write_json(artifact_paths["merge_manifest_json"], manifest)
        write_text(artifact_paths["report_readme"], refreshed_report)
        if compatibility_shims:
            write_json(compatibility_shims["legacy_merge_manifest_json"], manifest)
        written.append(repo_relative_path(artifact_paths["run_status_json"]))
    result["written"] = written
    return result
