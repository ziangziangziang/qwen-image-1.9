from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_image_19.config_io import repo_root, write_json, write_text
from qwen_image_19.remote import default_remote_context
from qwen_image_19.stage_1_analysis import load_model_inventory

from qwen_image_19.stage_2_fusion._manifest import (
    STAGE1_ARTIFACT_DIR,
    DEFAULT_STAGE2_ARTIFACT_DIR,
    DEFAULT_STAGE2_RUN_STATUS,
    CORE_CANDIDATE_DEFAULT_WEIGHT,
    DEFAULT_RUN_PROFILE,
    SUPPORTED_RUN_PROFILES,
    _fmt_duration,
    _describe_job,
    _emit_progress,
    Stage2FusionError,
    stage2_remote_path,
    repo_relative_path,
    stage2_artifact_paths,
    validate_run_options,
    load_run_profiles,
    parse_candidate_weight,
    build_stage2_compatibility_shims,
    render_stage2_compatibility_stub,
    require_stage1_artifacts,
    find_subsystem,
    extract_transformer_block_ids,
    top_mismatch_prefixes,
    build_stage1_evidence,
    build_exclusions,
    build_core_delta_recipe,
    core_candidate_id,
    build_core_delta_candidates,
    select_core_candidate,
    build_layered_bridge_recipe,
    build_planned_dataset_records,
    build_dataset_manifest,
    build_remote_jobs,
    build_artifacts_section,
    build_fusion_manifest,
    rebuild_dataset_records,
    apply_run_profile,
)
from qwen_image_19.stage_2_fusion._jobs import (
    ensure_outputs_exist,
    clear_target_path,
    clear_output_targets,
    run_subprocess_job,
    diffusion_resource_args,
    diffusion_sampling_args,
    build_job_command,
    run_stage2_jobs,
)
from qwen_image_19.stage_2_fusion._report import (
    render_candidate_rows,
    render_dataset_rows,
    render_job_rows,
    render_artifact_rows,
    render_fusion_report,
)


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
    effective_execute = bool((smoke_run and not dry_run) or (execute and not dry_run))
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
    manifest["execution_enabled"] = effective_execute
    manifest["execution_policy"] = "resume" if resume else "overwrite"
    manifest["cleanup_performed"] = False
    manifest["artifacts"]["run_status"] = repo_relative_path(artifact_paths["run_status_json"])

    report = render_fusion_report(manifest, dataset_manifest)
    result = {
        "stage": "stage2",
        "mode": "dry-run" if dry_run else "write",
        "run_profile": selected_run_profile,
        "execution_enabled": effective_execute,
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
    if effective_execute:
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

        if manifest.get("run_profile") != "smoke":
            training_report_command = [
                str(remote_context.get("python") or "python3"),
                "-m",
                "qwen_image_19.stage_2_fusion._worker_training_report",
                "--run-status",
                repo_relative_path(artifact_paths["run_status_json"]),
                "--merge-manifest",
                repo_relative_path(artifact_paths["merge_manifest_json"]),
                "--dataset-manifest",
                repo_relative_path(artifact_paths["dataset_manifest_json"]),
                "--metrics",
                stage2_remote_path("stage-2", "metrics", "layered-bridge-train.json"),
                "--output-md",
                manifest["artifacts"]["training_report"],
                "--figures-dir",
                manifest["artifacts"]["training_figures_dir"],
            ]
            training_report_log = stage2_remote_path("stage-2", "logs", "training-report.log")
            exit_code, _ = run_subprocess_job(
                training_report_command,
                training_report_log,
                stage2_remote_path("stage-2", "jobs", "training-report"),
            )
            if exit_code != 0:
                raise Stage2FusionError(
                    "Stage 2 execution succeeded but training report generation failed. "
                    f"Check `{training_report_log}`."
                )

        refreshed_report = render_fusion_report(manifest, dataset_manifest, run_status=run_status)
        write_json(artifact_paths["merge_manifest_json"], manifest)
        write_text(artifact_paths["report_readme"], refreshed_report)
        if compatibility_shims:
            write_json(compatibility_shims["legacy_merge_manifest_json"], manifest)
        written.append(repo_relative_path(artifact_paths["run_status_json"]))
    result["written"] = written
    return result
